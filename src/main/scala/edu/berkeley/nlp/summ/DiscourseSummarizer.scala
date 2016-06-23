package edu.berkeley.nlp.summ

import scala.collection.JavaConverters.asScalaSetConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import org.gnu.glpk.GLPK
import org.gnu.glpk.GLPKConstants
import org.gnu.glpk.glp_iocp
import org.gnu.glpk.glp_prob
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.summ.data.DiscourseTree
import edu.berkeley.nlp.summ.data.DiscourseNode
import edu.berkeley.nlp.summ.data.StopwordDict
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed

/**
 * Generic discourse-aware summarizer
 */
trait DiscourseDepExSummarizer extends Serializable {
  def summarize(ex: DiscourseDepExProcessed, budget: Int, cleanUpForHumans: Boolean = true): Seq[String]
  def summarizeOracle(ex: DiscourseDepExProcessed, budget: Int): Seq[String]
  def display(ex: DiscourseDepExProcessed, budget: Int);
  def printStatistics();
}

object DiscourseDepExSummarizer {
  
  // Used for tie-breaking in the ILP scores. This seems to be the best way to do it; works better than a similar
  // version which biases towards earlier content but generally including more content
  def biasTowardsEarlier(leafScores: Seq[Double]) = {
    (0 until leafScores.size).map(i => leafScores(i) - (i+1) * 0.000001)
  }
}

/**
 * Tree knapsack system of Hirao et al. (2013) and Yoshida et al. (2014), depending on whether
 * it is instantiated over gold or predicted discourse dependency trees. EDUs are scored heuristically
 * and an EDU can only be included if its parent is included as well.
 */
object TreeKnapsackSummarizer {
  
  def computeEduValuesUseStopwordSet(leafWordss: Seq[Seq[String]], parents: Seq[Int]): Array[Double] = {
    computeEduValues(leafWordss, parents, (word: String) => StopwordDict.stopwords.contains(word))
  }
  
  def computeEduValuesUsePoss(leafWordss: Seq[Seq[String]], leafPoss: Seq[Seq[String]], parents: Seq[Int]): Array[Double] = {
    val leafPossDict = new HashMap[String,String]
    for (i <- 0 until leafWordss.size) {
      for (j <- 0 until leafWordss(i).size) {
        leafPossDict += leafWordss(i)(j) -> leafPoss(i)(j)
      }
    }
    computeEduValues(leafWordss, parents, (word: String) => StopwordDict.stopwordTags.contains(leafPossDict(word)))
  }
  
  def computeEduValues(leafWordss: Seq[Seq[String]], parents: Seq[Int], stopwordTest: (String => Boolean)): Array[Double] = {
    val depths = DiscourseTree.computeDepths(parents, Array.fill(parents.size)(""), true)
    val wordFreqs = new Counter[String]
    for (leafWords <- leafWordss) {
      for (word <- leafWords) {
        wordFreqs.incrementCount(word, 1.0)
      }
    }
    // Use log counts
    for (word <- wordFreqs.keySet.asScala) {
      val containsLetter = word.map(c => Character.isLetter(c)).foldLeft(false)(_ || _)
      val isStopword = stopwordTest(word)
      val isWordValid = containsLetter && !isStopword
      if (isWordValid) {
        wordFreqs.setCount(word, Math.log(1 + wordFreqs.getCount(word))/Math.log(2))
      } else {
        wordFreqs.setCount(word, 0.0)
      }
    }
    Array.tabulate(leafWordss.size)(i => {
      val wordSet = leafWordss(i).toSet
      var totalCount = 0.0
      for (word <- wordSet) {
        totalCount += wordFreqs.getCount(word)
      }
      totalCount.toDouble / depths(i)
    })
  }
  
  def scoreHypothesis(tree: DiscourseTree, budget: Int, eduScores: Array[Double], edusOn: Seq[Int]) = {
    var totalScore = 0.0
    var budgetUsed = 0
    var constraintsViolated = 0
    for (edu <- edusOn) {
      totalScore += eduScores(edu)
      budgetUsed += tree.leaves(edu).leafWords.size
      if (tree.parents(edu) != -1 && !edusOn.contains(tree.parents(edu))) {
        Logger.logss("Violated constraint! Included " + edu + " without the parent " + tree.parents(edu))
        constraintsViolated += 1
      }
    }
    Logger.logss("Score: " + totalScore + " with budget " + budgetUsed + "/" + budget + "; " + constraintsViolated + " constraints violated")
  }
  
  def summarizeFirstK(leaves: Seq[DiscourseNode], budget: Int): Seq[String] = {
    val sents = new ArrayBuffer[String]
    var budgetUsed = 0
    var leafIdx = 0
    var done = false
    while (!done) {
      val currLeafWords = leaves(leafIdx).leafWords
      if (budgetUsed + currLeafWords.size < budget) {
        sents += currLeafWords.reduce(_ + " " + _)
        budgetUsed += currLeafWords.size
      } else {
        done = true
        // Comment or uncomment this to take partial EDUs
//        sents += currLeafWords.slice(0, budget - budgetUsed).reduce(_ + " " + _)
        budgetUsed = budget
      }
      leafIdx += 1
    }
    sents
  }
  
  def summarizeILP(leafSizes: Seq[Int], parents: Seq[Int], budget: Int, eduScores: Seq[Double], useGurobi: Boolean): (Seq[Int], Double) = {
    summarizeILP(leafSizes, parents, Array.fill(parents.size)(""), budget, eduScores, 1, 1)
  }
  
  def summarizeILP(leafSizes: Seq[Int], parents: Seq[Int], labels: Seq[String], budget: Int, eduScores: Seq[Double], numEqualsConstraints: Int, numParentConstraints: Int): (Seq[Int], Double) = {
    summarizeILPWithGLPK(leafSizes, budget, eduScores, makeConstraints(parents, labels, numEqualsConstraints, numParentConstraints))
  }
  
  def makeConstraints(parents: Seq[Int], labels: Seq[String], numEqualsConstraints: Int, numParentConstraints: Int): ArrayBuffer[(Int,Int)] = {
    val constraints = new ArrayBuffer[(Int,Int)]
    for (i <- 0 until parents.size) {
      val isEq = labels(i).startsWith("=")
      val isList = labels(i) == "=List"
      if (parents(i) != -1 && labels(i).startsWith("=")) {
        if (numEqualsConstraints == 1) {
          constraints += (parents(i) -> i)
        } else if (numEqualsConstraints == 2) {
          constraints += (parents(i) -> i)
          constraints += (i -> parents(i))
        }
      } else {
        if (parents(i) != -1) {
          if (numParentConstraints == 1) {
            constraints += (parents(i) -> i)
          }
        }
      }
    }
    constraints
  }
  
  // constraints = sequence of (a, b) pairs where b's inclusion implies a's as well (a = parent, b = child in the standard case)
  def summarizeILPWithGLPK(leafSizes: Seq[Int], budget: Int, eduScores: Seq[Double], constraints: Seq[(Int,Int)]): (Seq[Int], Double) = {
    val debug = false
    // Turn off output
    GLPK.glp_term_out(GLPKConstants.GLP_OFF)
    val lp = GLPK.glp_create_prob();
    GLPK.glp_set_prob_name(lp, "myProblem");
    
    // Variables
    val numVariables = leafSizes.size
    GLPK.glp_add_cols(lp, numVariables);
    for (i <- 0 until numVariables) {
      GLPK.glp_set_col_name(lp, i+1, "x" + i);
      GLPK.glp_set_col_kind(lp, i+1, GLPKConstants.GLP_BV);
    }
    // Objective weights
    GLPK.glp_set_obj_name(lp, "obj");
    GLPK.glp_set_obj_dir(lp, GLPKConstants.GLP_MAX);
    for (i <- 0 until leafSizes.size) {
      GLPK.glp_set_obj_coef(lp, i+1, eduScores(i));
    }
    // Constraints
    val numConstraints = constraints.size
    GLPK.glp_add_rows(lp, numConstraints+1);
    val ind = GLPK.new_intArray(numVariables+1);
    val values = GLPK.new_doubleArray(numVariables+1);
    // Binary constraints, usually from parent-child relationships
    for (i <- 0 until constraints.size) {
      val parent = constraints(i)._1
      val child = constraints(i)._2
      GLPK.intArray_setitem(ind, 1, parent+1);
      GLPK.intArray_setitem(ind, 2, child+1);
      GLPK.doubleArray_setitem(values, 1, 1);
      GLPK.doubleArray_setitem(values, 2, -1);
      GLPK.glp_set_row_name(lp, i+1, "c" + (i+1));
      GLPK.glp_set_row_bnds(lp, i+1, GLPKConstants.GLP_LO, 0, 0);
      GLPK.glp_set_mat_row(lp, i+1, 2, ind, values);
    }
    // Length constraint
    for (j <- 0 until leafSizes.size) {
      GLPK.intArray_setitem(ind, j+1, j+1);
      GLPK.doubleArray_setitem(values, j+1, leafSizes(j));
    }
    GLPK.glp_set_row_name(lp, numConstraints + 1, "clen");
    GLPK.glp_set_row_bnds(lp, numConstraints + 1, GLPKConstants.GLP_UP, 0, budget);
    GLPK.glp_set_mat_row(lp, numConstraints + 1, leafSizes.size, ind, values);
    GLPK.delete_doubleArray(values);
    GLPK.delete_intArray(ind);
    
    val (soln, score) = solveILPAndReport(lp, leafSizes.size)
    GLPK.glp_delete_prob(lp)
    (soln, score)
  }
  
  def solveILPAndReport(lp: glp_prob, numEdus: Int): (Seq[Int],Double) = {
    val iocp = new glp_iocp();
    GLPK.glp_init_iocp(iocp);
    iocp.setPresolve(GLPKConstants.GLP_ON);
    val edusChosen = new ArrayBuffer[Int]
    val ret = GLPK.glp_intopt(lp, iocp);
    if (ret == 0) {
      for (i <- 0 until numEdus) {
        val colValue = GLPK.glp_mip_col_val(lp, i+1)
        if (colValue == 1) {
          edusChosen += i
        }
      }
    } else {
      throw new RuntimeException("Couldn't solve!")
    }
    edusChosen.toSeq -> GLPK.glp_mip_obj_val(lp)
  }
  
  def extractSummary(leafWords: Seq[Seq[String]], selected: Seq[Int]) = {
    selected.map(leafIdx => leafWords(leafIdx).reduce(_ + " " + _))
  }
  
}
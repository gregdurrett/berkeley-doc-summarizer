package edu.berkeley.nlp.summ

import scala.collection.mutable.ArrayBuffer
import org.gnu.glpk.GLPK
import org.gnu.glpk.GLPKConstants
import edu.berkeley.nlp.summ.data.FragilePronoun
import edu.berkeley.nlp.summ.data.PronounReplacement
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed

/**
 * ILP implementation for the main summarizer.
 */
object CompressiveAnaphoraSummarizerILP {

  def summarizeILPWithGLPK(ex: DiscourseDepExProcessed,
                           parents: Seq[Int],
                           eduScores: Seq[Double],
                           bigramScores: Seq[Double],
                           pronReplacements: Seq[PronounReplacement],
                           pronReplacementScores: Seq[Double],
                           budget: Int,
                           numEqualsConstraints: Int,
                           numParentConstraints: Int,
                           doPronounConstraints: Boolean,
                           fragilePronouns: Seq[FragilePronoun],
                           useUnigrams: Boolean): (Seq[Int], Seq[Int], Seq[Int], Seq[Int], Seq[Int], Double) = {
    require(pronReplacements.isEmpty || useUnigrams, "Can't do pronoun replacement correctly with bigrams right now")
    val constraints = TreeKnapsackSummarizer.makeConstraints(parents, ex.parentLabels, numEqualsConstraints, numParentConstraints)
    val leafSizes = ex.leafSizes
    val parentLabels = ex.parentLabels
    
//    val docBigramsSeq = ex.getEduBigramsMap.reduce(_ ++ _).toSet.toSeq.sorted
    val docBigramsSeq = ex.getDocBigramsSeq(useUnigrams)
    
    val edusContainingBigram = docBigramsSeq.map(bigram => {
      (0 until ex.eduAlignments.size).filter(eduIdx => {
        val words = ex.getEduWords(eduIdx)
        val poss = ex.getEduPoss(eduIdx)
        val bigrams = if (useUnigrams) RougeComputer.getUnigramsNoStopwords(words, poss) else RougeComputer.getBigramsNoStopwords(words, poss)
        bigrams.contains(bigram)
      })
    })
    val pronRepsContainingBigram = docBigramsSeq.map(bigram => new ArrayBuffer[Int])
    for (pronReplacementIdx <- 0 until pronReplacements.size) {
      val pronReplacement = pronReplacements(pronReplacementIdx)
      // N.B. We currently don't handle deleted bigrams; it's assumed there really won't
      // be any because the pronoun replacement should only replace pronouns
      val addedDeletedBigrams = ex.getBigramDelta(pronReplacement, useUnigrams)
      for (bigram <- addedDeletedBigrams._1.toSeq.sorted) {
        val bigramIdx = if (pronReplacement.addedGenitive) {
          // This should always be fine because 's is its own token and is a stopword, so
          // the only time it appears in a token should be if we made a genitive alteration...
          docBigramsSeq.indexOf(bigram._1.replace("'s", "") -> bigram._2.replace("'s", ""))
        } else {
          docBigramsSeq.indexOf(bigram)
        }
        // Very occasional non-unique things are possible due to the genitive modification...
        // it did cause a crash at one point
        if (bigramIdx >= 0 && !pronRepsContainingBigram(bigramIdx).contains(pronReplacementIdx)) {
          pronRepsContainingBigram(bigramIdx) += pronReplacementIdx
        }
      }
    }
    
    val debug = false
    
    // Turn off output
    GLPK.glp_term_out(GLPKConstants.GLP_OFF)
    val lp = GLPK.glp_create_prob();
    GLPK.glp_set_prob_name(lp, "myProblem");
    
    // Variables
    val numEdus = leafSizes.size
    val numBigrams = docBigramsSeq.size
    val numProns = pronReplacements.size
    val numVariables = numEdus + numBigrams + numProns 
    val bigramOffset = numEdus
    val pronOffset = numEdus + numBigrams
    val cutOffset = numEdus + numBigrams + numProns
    val restartOffset = numEdus + numBigrams + numProns
    GLPK.glp_add_cols(lp, numVariables);
    for (i <- 0 until numVariables) {
      GLPK.glp_set_col_name(lp, i+1, "x" + i);
      GLPK.glp_set_col_kind(lp, i+1, GLPKConstants.GLP_BV);
    }
    // Objective weights
    GLPK.glp_set_obj_name(lp, "obj");
    GLPK.glp_set_obj_dir(lp, GLPKConstants.GLP_MAX);
    for (i <- 0 until numEdus) {
      GLPK.glp_set_obj_coef(lp, i+1, eduScores(i));
    }
    for (i <- 0 until numBigrams) {
      GLPK.glp_set_obj_coef(lp, bigramOffset+i+1, bigramScores(i));
    }
    for (i <- 0 until numProns) {
      GLPK.glp_set_obj_coef(lp, pronOffset+i+1, pronReplacementScores(i));
    }
    // Constraints
//    val numBigramConstraints = numBigrams + edusContainingBigram.map(_.size).foldLeft(0)(_ + _)
//    val numPronConstraints = pronReplacements.size * 2 + pronReplacements.map(_.prevEDUsContainingEntity.size).foldLeft(0)(_ + _)
//    val numConstraints = constraints.size + 1 + numBigramConstraints + numPronConstraints
//    GLPK.glp_add_rows(lp, numConstraints+1);
    var constraintIdx = 0
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
      GLPK.glp_add_rows(lp, 1);
      GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_LO, 0, 0);
      GLPK.glp_set_mat_row(lp, constraintIdx+1, 2, ind, values);
      constraintIdx += 1
    }
    // BIGRAMS
    // Constraints representing bigrams being in or out
    // bigram >= edu for each edu containing that bigram AND for each pron rep implying it
    // bigram - edu/pronrep >= 0
    for (i <- 0 until numBigrams) {
      val edusContaining = edusContainingBigram(i)
      val pronRepsContaining = pronRepsContainingBigram(i)
//      val pronRepsContaining = Seq[Int]()
      for (eduContaining <- edusContaining) {
        GLPK.intArray_setitem(ind, 1, bigramOffset+i+1);
        GLPK.intArray_setitem(ind, 2, eduContaining+1);
        GLPK.doubleArray_setitem(values, 1, 1);
        GLPK.doubleArray_setitem(values, 2, -1);
        GLPK.glp_add_rows(lp, 1);
        GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_LO, 0, 0);
        GLPK.glp_set_mat_row(lp, constraintIdx+1, 2, ind, values);
        constraintIdx += 1
      }
      for (pronRepContaining <- pronRepsContainingBigram(i)) {
        GLPK.intArray_setitem(ind, 1, bigramOffset+i+1);
        GLPK.intArray_setitem(ind, 2, pronOffset+pronRepContaining+1);
        GLPK.doubleArray_setitem(values, 1, 1);
        GLPK.doubleArray_setitem(values, 2, -1);
        GLPK.glp_add_rows(lp, 1);
        GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_LO, 0, 0);
        GLPK.glp_set_mat_row(lp, constraintIdx+1, 2, ind, values);
        constraintIdx += 1
      }
      // bigram <= sum of all edus containing the bigram AND sum of all pron replacements containing bigram
      // sum - bigram >= 0
      for (eduContainingIdx <- 0 until edusContaining.size) {
        GLPK.intArray_setitem(ind, eduContainingIdx+1, edusContaining(eduContainingIdx)+1);
        GLPK.doubleArray_setitem(values, eduContainingIdx+1, 1);
      }
      for (pronRepContainingIdx <- 0 until pronRepsContaining.size) {
        GLPK.intArray_setitem(ind, edusContaining.size + pronRepContainingIdx + 1, pronOffset + pronRepsContaining(pronRepContainingIdx)+1);
        GLPK.doubleArray_setitem(values, edusContaining.size + pronRepContainingIdx + 1, 1);
      }
      GLPK.intArray_setitem(ind, edusContaining.size + pronRepsContaining.size + 1, bigramOffset+i+1);
      GLPK.doubleArray_setitem(values, edusContaining.size + pronRepsContaining.size + 1, -1);
      GLPK.glp_add_rows(lp, 1);
      GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_LO, 0, 0);
      GLPK.glp_set_mat_row(lp, constraintIdx+1, edusContaining.size + pronRepsContaining.size + 1, ind, values);
      constraintIdx += 1
    }
    // PRONOUNS
    // Pronoun constraints
    for (pronReplacementIdx <- 0 until pronReplacements.size) {
      val pronReplacement = pronReplacements(pronReplacementIdx)
      // Pronoun is only on if the EDU is on: edu - pron >= 0 (pron <= edu)
      GLPK.intArray_setitem(ind, 1, pronReplacement.eduIdx+1);
      GLPK.intArray_setitem(ind, 2, pronOffset + pronReplacementIdx + 1);
      GLPK.doubleArray_setitem(values, 1, 1);
      GLPK.doubleArray_setitem(values, 2, -1);
      GLPK.glp_add_rows(lp, 1);
      GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_LO, 0, 0);
      GLPK.glp_set_mat_row(lp, constraintIdx+1, 2, ind, values);
      constraintIdx += 1
      // If the current edu is on, then the pronoun is on if none of the previous
      // instantiations of it are included.
      // pron >= 1 - sum(prev)
      // sum(prev_edu) + var >= 1 IF edu is on
      // sum(prev_edu) + var >= 0 otherwise
      // => var - edu + sum(prev_edu) >= 0
      val prevEdus = pronReplacement.prevEDUsContainingEntity
      GLPK.intArray_setitem(ind, 1, pronOffset + pronReplacementIdx + 1);
      GLPK.doubleArray_setitem(values, 1, 1);
      GLPK.intArray_setitem(ind, 2, pronReplacement.eduIdx+1);
      GLPK.doubleArray_setitem(values, 2, -1);
      for (i <- 0 until prevEdus.size) {
        GLPK.intArray_setitem(ind, i+3, prevEdus(i)+1);
        GLPK.doubleArray_setitem(values, i+3, 1);
      }
      GLPK.glp_add_rows(lp, 1);
      GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_LO, 0, 0);
      GLPK.glp_set_mat_row(lp, constraintIdx+1, prevEdus.size+2, ind, values);
      constraintIdx += 1
      // Pronoun is off if any previous instantiation is included
      // pron <= (1 - edu)
      // pron + edu <= 1
      for (i <- 0 until prevEdus.size) {
        GLPK.intArray_setitem(ind, 1, pronOffset + pronReplacementIdx + 1);
        GLPK.intArray_setitem(ind, 2, prevEdus(i) + 1);
        GLPK.doubleArray_setitem(values, 1, 1);
        GLPK.doubleArray_setitem(values, 2, 1);
        GLPK.glp_add_rows(lp, 1);
        GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_UP, 0, 1);
        GLPK.glp_set_mat_row(lp, constraintIdx+1, 2, ind, values);
        constraintIdx += 1
      }
      // For the version where pronouns are constraint, set pronoun <= 0 so we can't 
      // use pronoun replacements at all (just avoid including pronouns with bad anaphora)
      if (doPronounConstraints) {
        GLPK.intArray_setitem(ind, 1, pronOffset + pronReplacementIdx + 1);
        GLPK.doubleArray_setitem(values, 1, 1);
        GLPK.glp_add_rows(lp, 1);
        GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_UP, 0, 0);
        GLPK.glp_set_mat_row(lp, constraintIdx+1, 1, ind, values);
        constraintIdx += 1
      }
    }
    // FRAGILE PRONOUN CONSTRAINTS
    for (fragilePronoun <- fragilePronouns) {
      // eduIdx <= 1/(n-0.001)(prev_edu_sum) + pronRep
      // eduIdx - stuff - pronRep <= 0
      GLPK.intArray_setitem(ind, 1, fragilePronoun.eduIdx + 1);
      GLPK.doubleArray_setitem(values, 1, 1);
      var currConstIdx = 2
      // Subtract from the denominator so the constraint doesn't have floating point issues
      for (pastEduIdx <- fragilePronoun.antecedentEdus) {
        GLPK.intArray_setitem(ind, currConstIdx, pastEduIdx + 1);
        GLPK.doubleArray_setitem(values, currConstIdx, -1.0/(fragilePronoun.antecedentEdus.size - 0.01));
        currConstIdx += 1
      }
      val correspondingPronRepIndices = pronReplacements.zipWithIndex.filter(_._1.mentIdx == fragilePronoun.mentIdx).map(_._2)
      if (!correspondingPronRepIndices.isEmpty) {
        GLPK.intArray_setitem(ind, currConstIdx, pronOffset + correspondingPronRepIndices.head + 1);
        GLPK.doubleArray_setitem(values, currConstIdx, -1.0);
        correspondingPronRepIndices.head
        currConstIdx += 1
      }
      GLPK.glp_add_rows(lp, 1);
      GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_UP, 0, 0);
      GLPK.glp_set_mat_row(lp, constraintIdx+1, currConstIdx-1, ind, values);
      constraintIdx += 1
    }
    // CUTS AND RESTARTS
    val leq = true
    val geq = false
    def addConstraint(vars: Seq[Int], coeffs: Seq[Double], isLeq: Boolean, result: Int) {
      for (i <- 0 until vars.size) {
        GLPK.intArray_setitem(ind, i+1, vars(i));
        GLPK.doubleArray_setitem(values, i+1, coeffs(i));
      }
      GLPK.glp_add_rows(lp, 1);
      GLPK.glp_set_row_bnds(lp, constraintIdx+1, if (isLeq) GLPKConstants.GLP_UP else GLPKConstants.GLP_LO, if (isLeq) 0 else result, if (isLeq) result else 0);
      GLPK.glp_set_mat_row(lp, constraintIdx+1, vars.size, ind, values);
      constraintIdx += 1
    }
    // LENGTH
    for (i <- 0 until numEdus) {
      GLPK.intArray_setitem(ind, i+1, i+1);
      GLPK.doubleArray_setitem(values, i+1, leafSizes(i));
    }
    for (i <- 0 until numProns) {
      val colIdx = numEdus+i+1
      GLPK.intArray_setitem(ind, colIdx, pronOffset+i+1);
//      Logger.logss("Additional words: " + (pronReplacements(i-numEdus).replacementWords.size - 1))
      GLPK.doubleArray_setitem(values, colIdx, pronReplacements(i).replacementWords.size - 1);
    }
    GLPK.glp_add_rows(lp, 1);
    GLPK.glp_set_row_bnds(lp, constraintIdx+1, GLPKConstants.GLP_UP, 0, budget);
    GLPK.glp_set_mat_row(lp, constraintIdx+1, numEdus + numProns, ind, values);
    GLPK.delete_doubleArray(values);
    GLPK.delete_intArray(ind);
//    require(constraintIdx+1 == numConstraints, constraintIdx+1 + " " + numConstraints)
    
    val (soln, score) = TreeKnapsackSummarizer.solveILPAndReport(lp, numVariables)
    GLPK.glp_delete_prob(lp)
//    (soln.filter(_ < numEdus),
//     soln.filter(idx => idx >= pronOffset && idx < cutOffset).map(idx => idx - pronOffset),
//     soln.filter(idx => idx >= bigramOffset && idx < pronOffset).map(idx => idx - bigramOffset),
//     score)
    (soln.filter(_ < numEdus),
     soln.filter(idx => idx >= pronOffset && idx < cutOffset).map(idx => idx - pronOffset),
     soln.filter(idx => idx >= bigramOffset && idx < pronOffset).map(idx => idx - bigramOffset),
     soln.filter(idx => idx >= cutOffset && idx < restartOffset).map(idx => idx - cutOffset),
     soln.filter(idx => idx >= restartOffset).map(idx => idx - restartOffset),
     score)
  }
}
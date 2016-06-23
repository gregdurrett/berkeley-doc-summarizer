package edu.berkeley.nlp.summ.preprocess

import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import mstparser.KsummDependencyDecoder
import edu.berkeley.nlp.futile.classify.ClassifyUtils
import mstparser.FeatureVector
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.LightRunner
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.IntCounter
import edu.berkeley.nlp.futile.util.Counter
import scala.collection.JavaConverters._
import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.entity.coref.NumberGenderComputer
import edu.berkeley.nlp.entity.DepConstTree
import edu.berkeley.nlp.summ.CorefUtils
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed
import edu.berkeley.nlp.summ.data.DiscourseTree
import edu.berkeley.nlp.summ.data.DiscourseDepEx
import edu.berkeley.nlp.summ.TreeKnapsackSummarizer
import edu.berkeley.nlp.summ.data.DiscourseDepExNoGold
import edu.berkeley.nlp.summ.data.DiscourseTreeReader
import edu.berkeley.nlp.summ.GeneralTrainer
import edu.berkeley.nlp.summ.AdagradWeightVector
import edu.berkeley.nlp.summ.LikelihoodAndGradientComputerSparse
import edu.berkeley.nlp.summ.RougeComputer

/**
 * Training harness for the discourse dependency parser (requires segmented EDUs -- see
 * edu.berkeley.nlp.summ.preprocess.EDUSegmenter).
 */
object DiscourseDependencyParser {
  
  val preprocDocsPath = "../ksumm/data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES-PROC2/"
  val discourseTreesPath = "../ksumm/data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES/"
  val rstSummsPath = "../ksumm/data/RSTDiscourse/data/summaries-processed/"
  val numberGenderPath = "data/gender.data"
  
  val parallel = false
  val numItrs = 6
  val eta = 0.1
  val reg = 1e-8
  val batchSize = 1
  
  val withLoss = true
//  val withLoss = false
  
  val modelPath = "discoursedep.ser.gz"
  
  val isLabeled = true
  val useMultinuclearLinks = true
  
  val holdOutEvalSet = true
  
  val summDocNames = Seq("file1.out.dis", "wsj_1105.out.dis", "wsj_1111.out.dis", "wsj_1128.out.dis", "wsj_1142.out.dis", "wsj_1154.out.dis", "wsj_1162.out.dis", "wsj_1322.out.dis", "wsj_2309.out.dis", "wsj_2317.out.dis",
                         "wsj_1102.out.dis", "wsj_1107.out.dis", "wsj_1121.out.dis", "wsj_1131.out.dis", "wsj_1143.out.dis", "wsj_1157.out.dis", "wsj_1302.out.dis", "wsj_1331.out.dis", "wsj_2313.out.dis", "wsj_2322.out.dis",
                         "wsj_1103.out.dis", "wsj_1109.out.dis", "wsj_1122.out.dis", "wsj_1140.out.dis", "wsj_1146.out.dis", "wsj_1161.out.dis", "wsj_1317.out.dis", "wsj_1394.out.dis", "wsj_2315.out.dis", "wsj_2326.out.dis")
                         
  def main(args: Array[String]) {
    LightRunner.initializeOutput(DiscourseDependencyParser.getClass())
    LightRunner.populateScala(DiscourseDependencyParser.getClass(), args)
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(numberGenderPath);
    val mpc = new MentionPropertyComputer(Some(numberGenderComputer))
    val exs = DiscourseTreeReader.readAllAlignAndFilter(preprocDocsPath, discourseTreesPath, mpc)
    
    val labelIdx = new Indexer[String]
    if (isLabeled) {
      for (ex <- exs) {
        for (label <- ex.getLabels(useMultinuclearLinks)) {
          if (label != null) {
            labelIdx.getIndex(label)
          }
        }
      }
    } else {
      labelIdx.getIndex("default-label")
    }
    
    
    // Make sure we don't train on anything in the summ test set
    val summExs = exs.filter(ex => summDocNames.contains(ex.discourseTreeFileName))
    val trainExsRaw = if (holdOutEvalSet) {
      new ArrayBuffer[DiscourseDepEx] ++ exs.slice(0, (exs.size * 0.8).toInt)
    } else {
      new ArrayBuffer[DiscourseDepEx] ++ exs
    }
    val trainExs = new scala.util.Random(0).shuffle(trainExsRaw)
    Logger.logss(trainExs.size + " train exs before removing test exes")
    trainExs --= summExs
    Logger.logss(trainExs.size + " train exs after removing")
    
    val testExs = exs.slice((exs.size * 0.8).toInt, exs.size)
    
    Logger.logss("Extracting training set features")
    val featIdx = new Indexer[String]
    val featurizer = new DiscourseDependencyFeaturizer(featIdx)
    for (i <- 0 until trainExs.size) {
      if (trainExs.size < 10 || (i % (trainExs.size/10) == 0)) {
        Logger.logss("On example " + i + " / " + trainExs.size)
      }
      featurizer.extractFeaturesMSTCached(trainExs(i), true)
    }
    
    val computer = new DiscourseDependencyComputer(featurizer, labelIdx, withLoss, useMultinuclearLinks)
    val initialWeights = computer.getInitialWeights(0)
    Logger.logss(initialWeights.size + " total surface features; " + initialWeights.size + " total features because there are " + labelIdx.size + " labels")
    val weights = new GeneralTrainer(true).trainAdagradSparse(trainExs, computer, eta, reg, batchSize, numItrs, initialWeights, verbose = true);
    Logger.logss(computer.featMillis + " featurization millis")
    Logger.logss(computer.dpMillis + " DP millis")
    val model = new DiscourseDependencyParser(computer, weights)
    if (modelPath != "") {
      IOUtils.writeObjFileHard(modelPath, model)
    }
    
    if (holdOutEvalSet) {
      Logger.logss("TRAIN ACCURACY")
      eval(model, trainExs)
      Logger.logss("TEST ACCURACY")
      eval(model, testExs)
    }
    LightRunner.finalizeOutput()
  }
  
  def analyzeLabelCounts(exs: Seq[DiscourseDepEx]) {
    val labelCounter = new Counter[String]
    for (ex <- exs) {
      for (label <- ex.getLabels(useMultinuclearLinks)) {
        if (label != null) {
          labelCounter.incrementCount(label, 1.0)
        }
      }
    }
    val pq = labelCounter.asPriorityQueue()
    val labelCounterCondensed = new Counter[String]
    while (pq.hasNext()) {
      val key = pq.next
      Logger.logss(key + ": " + labelCounter.getCount(key))
      val compressedKey = (if (key.contains("-")) key.substring(0, key.indexOf("-")) else key)
      labelCounterCondensed.incrementCount(compressedKey, labelCounter.getCount(key))
    }
    Logger.logss("CONDENSED")
    val pqc = labelCounterCondensed.asPriorityQueue()
    while (pqc.hasNext()) {
      val key = pqc.next
      Logger.logss(key + ": " + labelCounterCondensed.getCount(key))
    }
  }
  
  
  def eval(depParser: DiscourseDependencyParser, testExs: Seq[DiscourseDepEx]) {
    var numCorrect = 0
    var numLabeledCorrect = 0
    var numRootsCorrect = 0
    var numRootsCorrectTrivial = 0
    // "Trivial" is a pure right-branching tree
    var numCorrectTrivial = 0
    var numCorrectSameAsTrivial = 0
    var numTotal = 0
    val numCorrectEachDepth = new Counter[String]
    val numTotalEachDepth = new Counter[String]
    val labelsCorrect = new Counter[String]
    val labelsCounts = new Counter[String]
    val labelsCountsCorrectEdge = new Counter[String]
    val labelConfusions = new Counter[String]
    val distTotalsEachDepth = new Counter[String]
    Logger.logss("Decoding test data")
    for (i <- 0 until testExs.size) {
      if (testExs.size < 10 || (i % (testExs.size/10) == 0)) {
        Logger.logss("On example " + i + " / " + testExs.size)
      }
      val testEx = testExs(i)
      val goldParents = testEx.getParents(useMultinuclearLinks)
      val goldLabels = testEx.getLabels(useMultinuclearLinks)
      val goldDepths = DiscourseTree.computeDepths(goldParents, goldLabels, false)
      val (parents, labels) = depParser.decode(testEx.stripGold)
      for (i <- 0 until parents.size) {
        val trivialCorrect = goldParents(i) == i-1
        val isRoot = goldParents(i) == -1
        if (trivialCorrect) {
          numCorrectTrivial += 1
          if (isRoot) numRootsCorrectTrivial += 1 
        }
        numTotalEachDepth.incrementCount(goldDepths(i) + "", 1.0)
        if (goldParents(i) == parents(i)) {
          numCorrect += 1
          numCorrectEachDepth.incrementCount(goldDepths(i) + "", 1.0)
          if (goldLabels(i) == labels(i)) {
            labelsCorrect.incrementCount(goldLabels(i), 1.0)
            numLabeledCorrect += 1
          } else {
            labelConfusions.incrementCount("G:" + goldLabels(i) + "--P:" + labels(i), 1.0) 
          }
          if (isRoot) numRootsCorrect += 1
          if (trivialCorrect) numCorrectSameAsTrivial += 1
          labelsCountsCorrectEdge.incrementCount(goldLabels(i), 1.0)
        }
        labelsCounts.incrementCount(goldLabels(i), 1.0)
        numTotal += 1
        distTotalsEachDepth.incrementCount("" + goldDepths(i), Math.abs(i) - goldParents(i))
      }
    }
    Logger.logss("Accuracy: " + ClassifyUtils.renderNumerDenom(numCorrect, numTotal))
    Logger.logss("Labeled accuracy: " + ClassifyUtils.renderNumerDenom(numLabeledCorrect, numTotal))
    Logger.logss("Root accuracy: " + ClassifyUtils.renderNumerDenom(numRootsCorrect, testExs.size))
    Logger.logss("Trivial accuracy: " + ClassifyUtils.renderNumerDenom(numCorrectTrivial, numTotal))
    Logger.logss("Trivial root accuracy: " + ClassifyUtils.renderNumerDenom(numRootsCorrectTrivial, testExs.size))
    Logger.logss("Num same as trivial: " + numCorrectSameAsTrivial)
    for (i <- 0 until 7) {
      Logger.logss("Accuracy at depth " + i + ": " + ClassifyUtils.renderNumerDenom(numCorrectEachDepth.getCount("" + i), numTotalEachDepth.getCount("" + i)))
      Logger.logss("Avg dist at depth " + i + ": " + ClassifyUtils.renderNumerDenom(distTotalsEachDepth.getCount("" + i), numTotalEachDepth.getCount("" + i)))
    }
    Logger.logss("Accuracy per label type:")
    for (key <- labelsCounts.keySet.asScala.toSeq.sorted) {
      Logger.logss(key + ": " + ClassifyUtils.renderNumerDenom(labelsCorrect.getCount(key), labelsCounts.getCount(key)) + "; when edge is correct: " + ClassifyUtils.renderNumerDenom(labelsCorrect.getCount(key), labelsCountsCorrectEdge.getCount(key)))
    }
    Logger.logss(labelConfusions.totalCount() + " total confusions:")
    val pq = labelConfusions.asPriorityQueue()
    while (pq.hasNext) {
      val key = pq.next
      Logger.logss(key + ": " + labelConfusions.getCount(key))
    }
  }
}

@SerialVersionUID(1L)
class DiscourseDependencyParser(computer: DiscourseDependencyComputer,
                                weights: Array[Double]) extends Serializable {
  val wrappedWeights = new AdagradWeightVector(weights, 0, 0)
  def decode(ex: DiscourseDepExNoGold) = computer.decode(ex, wrappedWeights)
}

@SerialVersionUID(1L)
class DiscourseDependencyComputer(val featurizer: DiscourseDependencyFeaturizer,
                                  val labelIdx: Indexer[String],
                                  val withLoss: Boolean,
                                  val useMultinuclearLinks: Boolean) extends LikelihoodAndGradientComputerSparse[DiscourseDepEx] with Serializable {
  val numSurfaceFeats = featurizer.featIdx.size()
  var featMillis = 0L
  var dpMillis = 0L
  
  val lossScalingFactor = 5.0
  
  def getInitialWeights(initialWeightsScale: Double): Array[Double] = Array.tabulate(featurizer.featIdx.size * labelIdx.size)(i => 0.0)
  
  def accumulateGradientAndComputeObjective(ex: DiscourseDepEx, weights: AdagradWeightVector, gradient: IntCounter): Double = {
    val feats = featurizer.extractFeaturesMSTCached(ex, false)
    val goldParents = ex.getParents(useMultinuclearLinks);
    val goldLabels = if (labelIdx.size == 1) Array.fill(goldParents.size)(0) else ex.getLabels(useMultinuclearLinks).map(labelIdx.indexOf(_))
    val (predParents, predLabels, predScore) = decode(ex.stripGold, feats, weights, 1.0, Some(goldParents));
//    val recomputedPredScore = scoreParse(ex, weights, predParents, 1.0)
    val goldScore = scoreParse(ex, weights, goldParents, goldLabels, 1.0)
//    Logger.logss("Pred score: " + predScore + ", recomputed pred score: " + recomputedPredScore + ", gold score: " + goldScore)
    for (i <- 0 until goldParents.size) {
      val goldFeats = ex.cachedFeatures(Math.min(i,goldParents(i))+1)(Math.max(i,goldParents(i))+1)(if (goldParents(i) < i) 0 else 1)
      require(goldFeats.size != 0, "Should never have zero gold feats")
      for (feat <- goldFeats) {
        gradient.incrementCount(offsetWeight(feat, goldLabels(i)), 1)
      }
      val predFeats = ex.cachedFeatures(Math.min(i,predParents(i))+1)(Math.max(i,predParents(i))+1)(if (predParents(i) < i) 0 else 1)
      require(predFeats.size != 0, "Should never have zero pred feats")
      for (feat <- predFeats) {
        gradient.incrementCount(offsetWeight(feat, predLabels(i)), -1)
      }
    }
    predScore - goldScore
  }
  
  def computeObjective(ex: DiscourseDepEx, weights: AdagradWeightVector): Double = accumulateGradientAndComputeObjective(ex, weights, new IntCounter())

  // N.B. Does not cache features so should not be used in normal decoding
  def decode(ex: DiscourseDepExNoGold, weights: AdagradWeightVector): (Array[Int], Array[String]) = {
    val feats = featurizer.extractFeaturesMST(ex, false)
    val (parents, labelIndices, _) = decode(ex, feats, weights, 0, None)
    parents -> labelIndices.map(labelIdx.getObject(_))
  }
  
  private def decode(ex: DiscourseDepExNoGold, feats: Array[Array[Array[Array[Int]]]], weights: AdagradWeightVector, lossWeight: Double, goldParents: Option[Seq[Int]]): (Array[Int], Array[Int], Double) = {
    val depths = if (goldParents.isDefined) Some(DiscourseTree.computeDepths(goldParents.get, Array.fill(goldParents.get.size)(""), false)) else None
    val time = System.nanoTime
    val labelScoresCache = Array.fill(labelIdx.size)(0.0)
    val staticLabels = Array.fill(ex.size + 1, ex.size + 1, 2)(0)
    val scores = Array.fill(ex.size + 1, ex.size + 1, 2)(0.0)
    for (i <- 0 until ex.size + 1; j <- i until ex.size + 1; dir <- 0 to 1) {
      // We only need scores where i < j 
      if (i < j && ((dir == 0 && j >= 1) || (dir == 1 && i >= 1))) {
        // Score each potential edge
        // Note that this works because roots are -1 (our notation) / 0 (MST parser notation)
        // Try all possible
        var scoreAugment = 0.0
        if (goldParents.isDefined) {
          val parents = goldParents.get
          val isGold = if (dir == 0) parents(j-1) == i-1 else parents(i-1) == j-1 // dir = 0 => left is the head
          // Loss augmentation 
          scoreAugment = (if (isGold) 0 else getLoss(lossWeight, if (dir == 0) depths.get(j-1) else depths.get(i-1)))
        }
        // We can independently max over the label here because labels don't interact with anything else
        // in the DP.
        for (label <- 0 until labelIdx.size) {
          labelScoresCache(label) = scoreFeats(feats(i)(j)(dir), label, weights)
        }
        val bestLabel = ClassifyUtils.argMaxIdx(labelScoresCache)
        staticLabels(i)(j)(dir) = bestLabel
        scores(i)(j)(dir) = labelScoresCache(bestLabel) + scoreAugment
      } else {
        // Make sure we don't use these
        staticLabels(i)(j)(dir) = -1
        scores(i)(j)(dir) = Double.NegativeInfinity
      }
    }
    val time2 = System.nanoTime
    featMillis += (time2 - time)/1000000
    val nullFvs = Array.tabulate(ex.size + 1, ex.size + 1, 2)((i, j, dir) => new FeatureVector)
    // Call out to MSTParser to do the decode.
    val pair = KsummDependencyDecoder.decodeProjective(ex.size + 1, nullFvs, scores, 1)
    if (pair.getFirst().size != ex.size) {
      Logger.logss("ERROR: Projective decoding didn't return a result!")
    }
    val predParents = pair.getFirst
    val predScore = pair.getSecond
    val predLabels = Array.tabulate(predParents.size)(i => staticLabels(Math.min(predParents(i)+1,i+1))(Math.max(predParents(i)+1,i+1))(if (predParents(i) < i) 0 else 1))
    dpMillis += (System.nanoTime - time2)/1000000
    (predParents, predLabels, predScore)
  }
  
  // Loss augmentation weights things at lower depth more heavily, but this doesn't matter much...
  private def getLoss(lossWeight: Double, depth: Int) = {
    if (withLoss) lossWeight / depth * lossScalingFactor else lossWeight * lossScalingFactor
  }
  
  private def offsetWeight(weightIdx: Int, labelIdx: Int) = {
    require(featurizer.featIdx.size() == numSurfaceFeats, "Number of surface features should not change; conjunctions of these " +
            "features and labels are computed by offsetting this manually and this process will break")
    weightIdx + labelIdx * numSurfaceFeats
  }
  
  private def scoreFeats(surfaceFeats: Array[Int], labelIdx: Int, weights: AdagradWeightVector) = {
    weights.scoreWithPosnOffset(surfaceFeats, labelIdx * numSurfaceFeats)
  }
  
  private def scoreParse(ex: DiscourseDepEx, weights: AdagradWeightVector, parents: Array[Int], labelIndices: Array[Int], lossWeight: Double) = {
    var score = 0.0
    val goldParents = ex.getParents(useMultinuclearLinks)
    val depths = DiscourseTree.computeDepths(goldParents)
    val goldLabels = if (labelIdx.size == 1) Array.fill(goldParents.size)(0) else ex.getLabels(useMultinuclearLinks).map(labelIdx.indexOf(_))
    for (i <- 0 until parents.size) {
      val goldFeats = ex.cachedFeatures(Math.min(parents(i)+1,i+1))(Math.max(parents(i)+1,i+1))(if (parents(i) < i) 0 else 1)
      score += scoreFeats(goldFeats, labelIndices(i), weights)
      // TODO: Change this to be correct!
      val isGold = parents(i) == goldParents(i) && labelIndices(i) == goldLabels(i)
      score += (if (isGold) 0.0 else getLoss(lossWeight, depths(i)))
    }
    score
  }
}

@SerialVersionUID(1L)
class DiscourseDependencyFeaturizer(val featIdx: Indexer[String]) extends Serializable {
  
  private def maybeAdd(feats: ArrayBuffer[Int], addToIndexer: Boolean, feat: String) {
    if (addToIndexer) {
      feats += featIdx.getIndex(feat)
    } else {
      val idx = featIdx.indexOf(feat)
      if (idx != -1) {
        feats += idx
      }
    }
  }
  
  def extractFeaturesMSTCached(ex: DiscourseDepEx, addToIndexer: Boolean) = {
    if (ex.cachedFeatures == null) {
      ex.cachedFeatures = extractFeaturesMST(ex.stripGold, addToIndexer)
    }
    ex.cachedFeatures
  }
  
  def extractFeaturesMST(ex: DiscourseDepExNoGold, addToIndexer: Boolean): Array[Array[Array[Array[Int]]]] = {
    Array.tabulate(ex.size + 1, ex.size + 1, 2)((i, j, dir) => {
      if (i < j) {
        extractFeaturesMST(ex, i, j, dir, addToIndexer)
      } else {
        Array[Int]()
      }
    })
  }
  
  // By convention, 0 is the root
  private def extractFeaturesMST(ex: DiscourseDepExNoGold, first: Int, second: Int, dir: Int, addToIndexer: Boolean): Array[Int] = {
    require(first < second)
    val parent = if (dir == 0) first else second
    val child = if (dir == 0) second else first
    // Can never have the child be the root symbol
    if (child == 0) {
      Array[Int]()
    } else {
      extractFeatures(ex, parent - 1, child - 1, addToIndexer)
    }
  }
  
  // Main feature extraction method.
  private def extractFeatures(ex: DiscourseDepExNoGold, parent: Int, child: Int, addToIndexer: Boolean): Array[Int] = {
    val feats = new ArrayBuffer[Int]()
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    val childWords = ex.getLeafWords(child)
    val childPoss = ex.getLeafPoss(child)
    // TODO: Paragraphs
    // TODO: Other conjunctions?
    // TODO: Neural features of some sort
    val childSentIdx = ex.eduAlignments(child)._1._1
    val isChildOneSent = ex.eduAlignments(child)._1._1 == ex.eduAlignments(child)._2._1
    if (parent == -1) {
      addLexicalFeatures(feats, addToIndexer, childWords, "ROOTCh")
      addLexicalFeatures(feats, addToIndexer, childPoss, "ROOTChPos")
      add("ROOT-Dist=" + bucket(child))
      add("ROOT-Len=" + bucket(childWords.size))
      add("ROOT-SentIdx=" + bucket(childSentIdx))
      add("ROOT-SentPosn=" + sentencePosnDescriptor(ex, child))
    } else {
      val parentWords = ex.getLeafWords(parent)
      val parentPoss = ex.getLeafPoss(parent)
      val parentSentIdx = ex.eduAlignments(parent)._1._1
      val isParentOneSent = ex.eduAlignments(parent)._1._1 == ex.eduAlignments(parent)._2._1
      addLexicalFeatures(feats, addToIndexer, childWords, "Ch")
      addLexicalFeatures(feats, addToIndexer, childPoss, "ChPos")
      addLexicalFeatures(feats, addToIndexer, parentWords, "Pa")
      addLexicalFeatures(feats, addToIndexer, parentPoss, "PaPos")
      add("DirDist=" + bucket(parent - child) + "-" + dir(parent, child))
      // These didn't seem to help...
//      add("DirDistPCPosns=" + bucket(parent - child) + "-" + dir(parent, child) + "-" + bucket(parent) + "-" + bucket(child))
//      add("DirDistPCEndPosns=" + bucket(parent - child) + "-" + dir(parent, child) + "-" + bucket(ex.eduAlignments.size - parent) + "-" + bucket(ex.eduAlignments.size - child))
//      add("DirDistPCMixedPosns=" + bucket(parent - child) + "-" + dir(parent, child) + "-" + bucket(parent) + "-" + bucket(ex.eduAlignments.size - child))
      add("LensDir=" + bucket(childWords.size) + "-" + bucket(parentWords.size) + "-" + dir(parent, child))
      add("LensDist=" + bucket(childWords.size) + "-" + bucket(parentWords.size) + "-" + bucket(parent - child))
      add("SentDistDir=" + bucket(Math.max(parentSentIdx, childSentIdx) - Math.min(parentSentIdx, childSentIdx)) + "-" + dir(parent, child))
      add("BegDistDir=" + bucket(parentSentIdx) + "-" + bucket(childSentIdx) + "-" + dir(parent, child))
      add("BegSentDistDir=" + bucket(parentSentIdx) + "-" + bucket(childSentIdx) + "-" + dir(parent, child))
      add("EndDistDir=" + bucket(ex.size - parent) + "-" + bucket(ex.size - child) + "-" + dir(parent, child))
      add("EndSentDistDir=" + bucket(ex.rawDoc.doc.size - parentSentIdx) + "-" + bucket(ex.rawDoc.doc.size - childSentIdx) + "-" + dir(parent, child))
      add("SentDescsDir=" + sentencePosnDescriptor(ex, parent) + "-" + sentencePosnDescriptor(ex, child) + "-" + dir(parent, child))
      // Dominance set features from Soricut and Marcu 2003 "Sentence Level Discourse Parsing using Syntactic and Lexical Information"
      if (childSentIdx == parentSentIdx && isChildOneSent && isParentOneSent) {
        val tree = ex.rawDoc.corefDoc.rawDoc.trees(childSentIdx)
        val childHeads = CorefUtils.getSpanHeads(tree, ex.eduAlignments(child)._1._2, ex.eduAlignments(child)._2._2)
        val parentHeads = CorefUtils.getSpanHeads(tree, ex.eduAlignments(child)._1._2, ex.eduAlignments(child)._2._2)
        for (childHead <- childHeads) {
          // Identify if some "head" of the child is a direct descendent of something in the parent
          val childHeadParent = tree.childParentDepMap(childHead)
          if (ex.eduAlignments(parent)._1._2 <= childHeadParent && childHeadParent < ex.eduAlignments(parent)._2._2) {
            addDominanceSetFeatures(feats, addToIndexer, tree, childHead, "PC")
          }
        }
        for (parentHead <- parentHeads) {
          // Identify if some "head" of the parent is a direct descendent of something in the child
          val parentHeadParent = tree.childParentDepMap(parentHead)
          if (ex.eduAlignments(child)._1._2 <= parentHeadParent && parentHeadParent < ex.eduAlignments(child)._2._2) {
            addDominanceSetFeatures(feats, addToIndexer, tree, parentHead, "CP")
          }
        }
      }
    }
    if (parent != -1) {
      val numFeats = feats.size
      val isIntrasentential = childSentIdx == ex.eduAlignments(parent)._1._1
      for (i <- 0 until numFeats) {
        val rawFeat = featIdx.getObject(feats(i))
        // Conjoin with whether it's root or not
        add(rawFeat + "-IS=" + isIntrasentential)
      }
    }
    feats.toArray
  }
  
  private def addDominanceSetFeatures(feats: ArrayBuffer[Int], addToIndexer: Boolean, tree: DepConstTree, childIdx: Int, prefix: String) {
    val parentIdx = tree.childParentDepMap(childIdx)
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    add("TagPairDir" + prefix + "=" + tree.pos(parentIdx) + "->" + tree.pos(childIdx) + "-" + dir(parentIdx, childIdx))
    add("WordPairDir" + prefix + "=" + tree.words(parentIdx) + "->" + tree.words(childIdx) + "-" + dir(parentIdx, childIdx))
  }
  
  private def addLexicalFeatures(feats: ArrayBuffer[Int], addToIndexer: Boolean, lexemes: Seq[String], strIdentifier: String) {
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    add(strIdentifier + "1=" + access(lexemes, 0))
    add(strIdentifier + "2=" + access(lexemes, 1))
    add(strIdentifier + "3=" + access(lexemes, 2))
    add(strIdentifier + "12=" + access(lexemes, 0, 2))
    add(strIdentifier + "123=" + access(lexemes, 0, 3))
    add(strIdentifier + "-1=" + access(lexemes, -1))
    add(strIdentifier + "-2=" + access(lexemes, -2))
    add(strIdentifier + "-3=" + access(lexemes, -3))
    add(strIdentifier + "-12=" + access(lexemes, -2, 0))
    add(strIdentifier + "-123=" + access(lexemes, -3, 0))
  }
  
  private def access(words: Seq[String], idx: Int): String = {
    if (idx < 0) access(words, words.size + idx) else if (idx >= words.size) "<EOS>" else words(idx)
  }
  
  private def access(words: Seq[String], startIdx: Int, endIdx: Int): String = {
    (startIdx until endIdx).map(access(words, _)).reduce(_ + " " + _)
  }
  
  private def dir(parent: Int, child: Int) = {
    if (parent < child) "R" else "L"
  }
  
  private def bucket(n: Int) = {
    if (n <= 4) n else if (n <= 8) "5-8" else if (n <= 16) "9-16" else if (n <= 32) "17-32" else "33+"  
  }
  
  private def sentencePosnDescriptor(ex: DiscourseDepExNoGold, idx: Int) = {
    val alignments = ex.eduAlignments
    val startsSentence = idx == 0 || alignments(idx - 1)._2._1 < alignments(idx)._1._1
    val endsSentence = idx == ex.size - 1 || alignments(idx)._2._1 < alignments(idx + 1)._1._1
    (if (startsSentence) "S" else "I") + (if (endsSentence) "E" else "I")
  }
}

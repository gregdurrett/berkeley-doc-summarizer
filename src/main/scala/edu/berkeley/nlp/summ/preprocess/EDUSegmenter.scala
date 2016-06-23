package edu.berkeley.nlp.summ.preprocess

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import edu.berkeley.nlp.futile.LightRunner
import edu.berkeley.nlp.futile.classify.ClassifyUtils
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.entity.ConllDoc
import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.entity.coref.NumberGenderComputer
import edu.berkeley.nlp.summ.data.DiscourseTreeReader
import edu.berkeley.nlp.summ.data.DiscourseDepEx
import edu.berkeley.nlp.summ.LikelihoodAndGradientComputer
import edu.berkeley.nlp.summ.GeneralTrainer

/**
 * Training harness for the EDU segmenter.
 */
object EDUSegmenterMain {
  
  val parallel = true
  val numItrs = 5
  val eta = 0.1
  val reg = 1e-8
  val batchSize = 1
  
  val recallErrorLossWeight = 1.5
  
  val modelPath = "edusegmenter.ser.gz"
  
  // Higher than 1 actually starts to hurt things on test...
  val wordCountThreshold = 1
  
  val useSemiMarkov = true
  
  // Evaluate the segmenter, versus training one that respects the documents with summaries, etc.
  val runEDUSegmenterEval = false
  
  val testSetNames = Seq("wsj_0602.out", "wsj_0607.out", "wsj_0616.out", "wsj_0623.out", "wsj_0627.out", "wsj_0632.out", "wsj_0644.out", "wsj_0654.out", "wsj_0655.out",
                         "wsj_0667.out", "wsj_0684.out", "wsj_0689.out", "wsj_1113.out", "wsj_1126.out", "wsj_1129.out", "wsj_1142.out", "wsj_1146.out", "wsj_1148.out",
                         "wsj_1169.out", "wsj_1183.out", "wsj_1189.out", "wsj_1197.out", "wsj_1306.out", "wsj_1307.out", "wsj_1325.out", "wsj_1331.out", "wsj_1346.out",
                         "wsj_1354.out", "wsj_1365.out", "wsj_1376.out", "wsj_1380.out", "wsj_1387.out", "wsj_2336.out", "wsj_2354.out", "wsj_2373.out", "wsj_2375.out",
                         "wsj_2385.out", "wsj_2386.out")
  
  val summDocNames = Seq("file1.out.dis", "wsj_1105.out.dis", "wsj_1111.out.dis", "wsj_1128.out.dis", "wsj_1142.out.dis", "wsj_1154.out.dis", "wsj_1162.out.dis", "wsj_1322.out.dis", "wsj_2309.out.dis", "wsj_2317.out.dis",
                         "wsj_1102.out.dis", "wsj_1107.out.dis", "wsj_1121.out.dis", "wsj_1131.out.dis", "wsj_1143.out.dis", "wsj_1157.out.dis", "wsj_1302.out.dis", "wsj_1331.out.dis", "wsj_2313.out.dis", "wsj_2322.out.dis",
                         "wsj_1103.out.dis", "wsj_1109.out.dis", "wsj_1122.out.dis", "wsj_1140.out.dis", "wsj_1146.out.dis", "wsj_1161.out.dis", "wsj_1317.out.dis", "wsj_1394.out.dis", "wsj_2315.out.dis", "wsj_2326.out.dis")
                         
  val docsPath = "../ksumm/data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES-PROC2/"
  val discourseTreesPath = "../ksumm/data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES/"
                         
  def extractSegments(predictions: Array[Array[Boolean]]): Seq[((Int,Int),(Int,Int))] = {
    val spans = new ArrayBuffer[((Int,Int),(Int,Int))]
    for (sentIdx <- 0 until predictions.size) {
      var lastFencepost = (sentIdx, 0)
      for (boundIdx <- 0 until predictions(sentIdx).size) {
        if (predictions(sentIdx)(boundIdx)) {
          spans += (lastFencepost -> (sentIdx, boundIdx + 1))
          lastFencepost = (sentIdx, boundIdx + 1)
        }
      }
      // Finish the sentence
      spans += lastFencepost -> (sentIdx, predictions(sentIdx).size + 1)
    }
    spans
  }
  
  def eval(segmenter: EDUSegmenter, testExs: Seq[DiscourseDepEx]) {
    var numCorrect = 0
    var numPredicted = 0
    var numGold = 0
    var total = 0
    for (testEx <- testExs) {
      val pred = segmenter.decode(testEx)
      val gold = testEx.goldSegmentBoundaries
      for (sentIdx <- 0 until pred.size) {
        for (boundIdx <- 0 until pred(sentIdx).size) {
          val isGold = gold(sentIdx)(boundIdx)
          val isGoldFlex = isGold || (boundIdx > 0 && gold(sentIdx)(boundIdx-1)) || (boundIdx < gold(sentIdx).size - 1) && gold(sentIdx)(boundIdx + 1)
          if (pred(sentIdx)(boundIdx) && isGold) numCorrect += 1
//          if (pred(sentIdx)(boundIdx) && isGoldFlex) numCorrect += 1
          if (pred(sentIdx)(boundIdx)) numPredicted += 1
          if (gold(sentIdx)(boundIdx)) numGold += 1
          total += 1
          if (!pred(sentIdx)(boundIdx) && gold(sentIdx)(boundIdx)) {
//            Logger.logss(boundIdx + " " + testEx.conllDoc.words(sentIdx)(boundIdx) + "\n" + PennTreeRenderer.render(testEx.conllDoc.trees(sentIdx).constTree))
            Logger.logss("RECALL ERROR: " + (boundIdx + 1) + " " + testEx.conllDoc.words(sentIdx).slice(0, boundIdx+1) + " ||| " + testEx.conllDoc.words(sentIdx).slice(boundIdx+1, testEx.conllDoc.words(sentIdx).size))
            Logger.logss("  golds: " + testEx.goldEduSpans(sentIdx))
            Logger.logss("  preds (starts): " + (0 until pred(sentIdx).size).map(i => if (pred(sentIdx)(i)) i+1 else -1).filter(_ != -1))
          }
          if (pred(sentIdx)(boundIdx) && !gold(sentIdx)(boundIdx)) {
            Logger.logss("PRECISION ERROR: " + (boundIdx + 1) + " " + testEx.conllDoc.words(sentIdx).slice(0, boundIdx+1) + " ||| " + testEx.conllDoc.words(sentIdx).slice(boundIdx+1, testEx.conllDoc.words(sentIdx).size))
            Logger.logss("  golds: " + testEx.goldEduSpans(sentIdx))
            Logger.logss("  pred (starts): " + (0 until pred(sentIdx).size).map(i => if (pred(sentIdx)(i)) i+1 else -1).filter(_ != -1))
          }
        }
      }
    }
    Logger.logss("P/R/F1: " + ClassifyUtils.renderPRF1(numCorrect, numPredicted, numGold) + "; total: " + total)
    val numTrivialBoundaries = testExs.map(_.rawDoc.doc.size).foldLeft(0)(_ + _) - 1
    Logger.logss("P/R/F1 with trivial: " + ClassifyUtils.renderPRF1(numCorrect + numTrivialBoundaries, numPredicted + numTrivialBoundaries, numGold + numTrivialBoundaries))
  }
                         
  def main(args: Array[String]) {
    LightRunner.initializeOutput(EDUSegmenterMain.getClass())
    LightRunner.populateScala(EDUSegmenterMain.getClass(), args)
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData("data/gender.data");
    val mpc = new MentionPropertyComputer(Some(numberGenderComputer))
    val exs = DiscourseTreeReader.readAllAlignAndFilter(docsPath, discourseTreesPath, mpc)
    
    // Make sure we don't train on anything in the summ test set
    val summExs = exs.filter(ex => summDocNames.contains(ex.discourseTreeFileName))
    val (trainExs, testExs) = if (runEDUSegmenterEval) {
      exs.filter(ex => !testSetNames.contains(ex.conllDoc.docID)) -> exs.filter(ex => testSetNames.contains(ex.conllDoc.docID))
    } else {
      val trainExs = new ArrayBuffer[DiscourseDepEx] ++ exs.slice(0, (exs.size * 0.8).toInt)
      Logger.logss(trainExs.size + " train exs before removing test exes")
      trainExs --= summExs
      Logger.logss(trainExs.size + " train exs after removing")
      val testExs = exs.slice((exs.size * 0.8).toInt, exs.size)
      trainExs -> testExs
    }
    Logger.logss(trainExs.size + " final train exes, " + testExs.size + " final test exes")
    
    Logger.logss("Extracting training set features")
    val featIdx = new Indexer[String]
    
    val trainWordCounts = new Counter[String]
    trainExs.foreach(_.conllDoc.words.foreach(_.foreach(trainWordCounts.incrementCount(_, 1.0))))
    
    val basicFeaturizer = new EDUSegmenterFeaturizer(featIdx, trainWordCounts, wordCountThreshold)
    val model = if (useSemiMarkov) {
      val featurizer = new EDUSegmenterSemiMarkovFeaturizer(featIdx, basicFeaturizer)
      trainExs.foreach(ex => featurizer.extractFeaturesCached(ex, true))
      val computer = new EDUSegmenterSemiMarkovComputer(featurizer)
      val initialWeights = computer.getInitialWeights(0)
      Logger.logss(initialWeights.size + " total features")
      val weights = new GeneralTrainer(true).trainAdagrad(trainExs, computer, eta, reg, batchSize, numItrs, initialWeights, verbose = true);
      new EDUSegmenterSemiMarkov(computer, weights)
    } else {
      trainExs.foreach(ex => basicFeaturizer.extractFeaturesCached(ex, true))
      val computer = new EDUSegmenterComputer(basicFeaturizer, recallErrorLossWeight)
      val initialWeights = computer.getInitialWeights(0)
      Logger.logss(initialWeights.size + " total features")
      val weights = new GeneralTrainer(true).trainAdagrad(trainExs, computer, eta, reg, batchSize, numItrs, initialWeights, verbose = true);
      new EDUSegmenterBinary(computer, weights)
    }
    Logger.logss("TRAIN PERFORMANCE")
    eval(model, trainExs)
    Logger.logss("TEST PERFORMANCE")
    eval(model, testExs)
    if (modelPath != "") {
      IOUtils.writeObjFileHard(modelPath, model)
    }
    LightRunner.finalizeOutput()
  }
}

trait EDUSegmenter extends Serializable {
  def decode(ex: DiscourseDepEx): Array[Array[Boolean]]
  def decode(doc: ConllDoc): Array[Array[Boolean]]
}

@SerialVersionUID(1L)
class EDUSegmenterBinary(val computer: EDUSegmenterComputer,
                         val weights: Array[Double]) extends EDUSegmenter {
  
  def decode(ex: DiscourseDepEx) = computer.decode(ex, weights)
  
  def decode(doc: ConllDoc) = {
    val feats = computer.featurizer.extractFeatures(doc, false)
    feats.map(_.map(vec => ClassifyUtils.scoreIndexedFeats(vec, weights) > 0))
  }
}


@SerialVersionUID(1L)
class EDUSegmenterComputer(val featurizer: EDUSegmenterFeaturizer,
                           val recallErrorLossWeight: Double) extends LikelihoodAndGradientComputer[DiscourseDepEx] with Serializable {
  
  def getInitialWeights(initialWeightsScale: Double): Array[Double] = Array.tabulate(featurizer.featIdx.size)(i => 0.0)
  
  def accumulateGradientAndComputeObjective(ex: DiscourseDepEx, weights: Array[Double], gradient: Array[Double]): Double = {
    val (predSegs, predScore) = decode(ex, weights, 1.0);
//    val recomputedPredScore = scoreParse(ex, weights, predParents, 1.0)
    val goldSegs = ex.goldSegmentBoundaries;
    val goldScore = scoreSegmentation(ex, weights, ex.goldSegmentBoundaries, 1.0)
//    Logger.logss("Pred score: " + predScore + ", recomputed pred score: " + recomputedPredScore + ", gold score: " + goldScore)
    for (sentIdx <- 0 until ex.conllDoc.numSents) {
      for (boundIdx <- 0 until ex.conllDoc.words(sentIdx).size - 1) {
        val feats = ex.cachedEduFeatures(sentIdx)(boundIdx)
        if (predSegs(sentIdx)(boundIdx) != goldSegs(sentIdx)(boundIdx)) {
          if (goldSegs(sentIdx)(boundIdx)) {
            // Update positive
            for (feat <- feats) {
              gradient(feat) += 1
            }
          } else {
            for (feat <- feats) {
              gradient(feat) -= 1
            }
          }
        }
      }
    }
    predScore - goldScore
  }
  
  def computeObjective(ex: DiscourseDepEx, weights: Array[Double]): Double = accumulateGradientAndComputeObjective(ex, weights, Array.tabulate(weights.size)(i => 0.0))
  
  def decode(ex: DiscourseDepEx, weights: Array[Double]): Array[Array[Boolean]] = {
    decode(ex, weights, 0)._1
  }
  
  private def decode(ex: DiscourseDepEx, weights: Array[Double], lossWeight: Double): (Array[Array[Boolean]], Double) = {
    val feats = featurizer.extractFeaturesCached(ex, false)
    var cumScore = 0.0
    val allPreds = Array.tabulate(ex.conllDoc.numSents)(sentIdx => {
      Array.tabulate(ex.conllDoc.words(sentIdx).size - 1)(boundIdx => {
        val isGold = ex.goldSegmentBoundaries(sentIdx)(boundIdx)
        val score = ClassifyUtils.scoreIndexedFeats(feats(sentIdx)(boundIdx), weights) + (if (isGold) (-lossWeight * recallErrorLossWeight) else lossWeight)
        if (score > 0) {
          cumScore += score
          true
        } else {
          cumScore -= score
          false
        }
      })
    })
    (allPreds, cumScore)
  }
  
  private def scoreSegmentation(ex: DiscourseDepEx, weights: Array[Double], segmentation: Array[Array[Boolean]], lossWeight: Double) = {
    var score = 0.0
    val feats = featurizer.extractFeaturesCached(ex, false)
    for (sentIdx <- 0 until ex.conllDoc.numSents) {
      for (boundIdx <- 0 until ex.conllDoc.words(sentIdx).size - 1) {
        val isGold = ex.goldSegmentBoundaries(sentIdx)(boundIdx)
        val scoreIncr = ClassifyUtils.scoreIndexedFeats(feats(sentIdx)(boundIdx), weights) + (if (isGold) -lossWeight else lossWeight)
        if (segmentation(sentIdx)(boundIdx)) {
          score += scoreIncr
        } else {
          score -= scoreIncr
        }
      }
    }
    score
  }
}

@SerialVersionUID(1L)
class EDUSegmenterFeaturizer(val featIdx: Indexer[String],
                             val trainWordCounts: Counter[String] = new Counter[String],
                             val wordCountThreshold: Int = 0) extends Serializable {
  
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
  
  def extractFeaturesCached(ex: DiscourseDepEx, addToIndexer: Boolean): Array[Array[Array[Int]]] = {
    // Only featurize sentence-internal boundaries (sentence boundaries are trivially EDU segments)
    if (ex.cachedEduFeatures == null) {
      ex.cachedEduFeatures = extractFeatures(ex.conllDoc, addToIndexer)
    }
    ex.cachedEduFeatures
  }
  
  def extractFeatures(doc: ConllDoc, addToIndexer: Boolean): Array[Array[Array[Int]]] = {
    // Only featurize sentence-internal boundaries (sentence boundaries are trivially EDU segments)
    Array.tabulate(doc.numSents)(sentIdx => {
      Array.tabulate(doc.words(sentIdx).size - 1)(boundIdx => {
        extractFeatures(doc, sentIdx, boundIdx, addToIndexer)
      })
    })
  }
  
  def extractFeatures(doc: ConllDoc, sentIdx: Int, boundIdx: Int, addToIndexer: Boolean): Array[Int] = {
    val beforeIdx = boundIdx
    val afterIdx = boundIdx + 1
    def words(idx: Int) = {
      val word = doc.words(sentIdx)(idx)
      if (trainWordCounts.getCount(word) >= wordCountThreshold) {
        word
      } else {
        "UNK"
      }
    }
    val pos = doc.pos(sentIdx)
    val feats = new ArrayBuffer[Int]()
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    add("WordBefore=" + words(beforeIdx))
    add("PosBefore=" + pos(beforeIdx))
    // TODO: N-grams on path to root
    // TODO: Pairs of deps for this and next
    if (beforeIdx > 0) {
      add("WordTwoBefore=" + words(beforeIdx - 1))
      add("PosTwoBefore=" + pos(beforeIdx - 1))
    } else {
      add("WordTwoBefore=<S>")
      add("PosTwoBefore=<S>")
    }
    add("WordAfter=" + words(afterIdx))
    add("PosAfter=" + pos(afterIdx))
    if (afterIdx + 1 < pos.size) {
      add("WordTwoAfter=" + words(afterIdx + 1))
      add("PosTwoAfter=" + words(afterIdx + 1))
    } else {
      add("WordTwoAfter=</S>")
      add("PosTwoAfter=</S>")
    }
    // Bigrams
    add("PosBigram=" + pos(beforeIdx) + "-" + pos(afterIdx))
    add("MixedBigram=" + pos(beforeIdx) + "-" + words(afterIdx))
    add("MixedBigram=" + words(beforeIdx) + "-" + pos(afterIdx))
    add("WordBigram=" + words(beforeIdx) + "-" + words(afterIdx))
    // Syntactic features
    getConstituencySyntacticFeats(doc, sentIdx, boundIdx, feats, addToIndexer)
    getDependencySyntacticFeats(doc, sentIdx, boundIdx, feats, addToIndexer)
    // Surface features
    // Word after last punc / beginning of sentence
    var lastPuncPosn = boundIdx - 1
    while (lastPuncPosn >= 0 && Character.isLetterOrDigit(words(lastPuncPosn).charAt(0))) {
      lastPuncPosn -= 1
    }
    val lastPunc = if (lastPuncPosn == -1) "<s>" else "" + words(lastPuncPosn)
    add("LastPunc=" + lastPunc)
    add("DistToLastPunc=" + bucket(boundIdx - lastPuncPosn))
//    if (lastPuncPosn != -1) {
      add("LastPuncWordAfter=" + words(lastPuncPosn + 1))
      add("LastPuncTagAfter=" + pos(lastPuncPosn + 1))
//    }
    // Word before next punc / end of sentence
    var nextPuncPosn = boundIdx + 2
    while (nextPuncPosn < pos.size && Character.isLetterOrDigit(words(nextPuncPosn).charAt(0))) {
      nextPuncPosn += 1
    }
    val nextPunc = if (nextPuncPosn == pos.size) "</s>" else words(nextPuncPosn)
    add("NextPunc=" + nextPunc)
    add("DistToNext=" + bucket(nextPuncPosn - boundIdx + 1))
//    if (nextPuncPosn != pos.size) {
      add("NextPuncWordBefore=" + words(nextPuncPosn - 1))
      add("NextPuncTagBefore=" + pos(nextPuncPosn - 1))
//    }
    feats.toArray
  }
  
  def bucket(n: Int) = {
    if (n <= 4) n else if (n <= 8) "5-8" else if (n <= 16) "9-16" else if (n <= 32) "17-32" else "33+"  
  }
  
  private def getConstituencySyntacticFeats(doc: ConllDoc, sentIdx: Int, boundIdx: Int, feats: ArrayBuffer[Int], addToIndexer: Boolean) = {
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    var featsAdded = false
    val tree = doc.trees(sentIdx)
    val constTree = tree.constTree
    val spanMap = constTree.getSpanMap()
    val constituents = constTree.getConstituents()
    // MAX PROJECTION
    // Get maximal projection that also had right child
    // More negative = largest will appear first
    for (span <- spanMap.keySet().asScala.toSeq.sortBy(span => span.getFirst - span.getSecond)) {
      // Check if we haven't added feats yet and if this is a maximal projection
      if (!featsAdded && span.getFirst <= boundIdx && boundIdx < span.getSecond && tree.getSpanHead(span.getFirst, span.getSecond) == boundIdx) {
        // This is the maximal projection, but does it have a right child?
        val maxTrees = spanMap.get(span).asScala.filter(_.getChildren.size >= 2)
        if (maxTrees.size >= 1) {
          val parent = maxTrees(0)
          val children = parent.getChildren
          var childIdx = -1
          for (i <- 0 until children.size - 1) {
            val child = children.get(i)
            val childSpan = constituents.get(child).getStart() -> (constituents.get(child).getEnd() + 1)
            val childHead = tree.getSpanHead(childSpan._1, childSpan._2)
            if (childHead == boundIdx) {
              childIdx = i
            }
          }
          // We've now identified which child is the one we want
          if (childIdx != -1) {
            featsAdded = true
            val rightChildSpan = constituents.get(children.get(childIdx+1)).getStart -> (constituents.get(children.get(childIdx+1)).getEnd + 1)
            val rightChildHead = tree.getSpanHead(rightChildSpan._1, rightChildSpan._2)
            if (children.size == 2) {
              add("CompleteRule=" + parent.getLabel() + " -> " + children.get(childIdx).getLabel + " " + children.get(childIdx+1).getLabel)
            } else {
              add("PartialRule=" + parent.getLabel() + " -> " + (if (childIdx > 0) "... " else "") + children.get(childIdx).getLabel +
                  " " + children.get(childIdx+1).getLabel + (if (childIdx + 1 < children.size - 1) " ..." else ""))
            }
            add("RightHead=" + doc.words(sentIdx)(rightChildHead))
            add("RightTag=" + doc.pos(sentIdx)(rightChildHead))
          }
        }
      }  
    }
    // SPAN BOUNDARY
    // TODO: These actually work a bit better!
//    val spansStartingHere = spanMap.keySet().asScala.toSeq.filter(_.getFirst == boundIdx + 1)
//    add("SpansStartingHere=" + bucket(spansStartingHere.size))
//    for (span <- spansStartingHere) {
//      val label = spanMap.get(span).get(0).getLabel()
//      add("SpanStartingHereLen=" + label + "-" + bucket(span.getSecond().intValue - span.getFirst().intValue))
//    }
//    val spansEndingHere = spanMap.keySet().asScala.toSeq.filter(_.getSecond == boundIdx + 1)
//    add("SpansEndingHere=" + bucket(spansEndingHere.size))
//    for (span <- spansEndingHere) {
//      val label = spanMap.get(span).get(0).getLabel()
//      add("SpanEndingHereLen=" + label + "-" + bucket(span.getSecond().intValue - span.getFirst().intValue))
//    }
    val spansEndingHere = spanMap.keySet().asScala.toSeq.filter(_.getSecond == boundIdx + 1)
    add("SpansEndingHere=" + spansEndingHere.size)
    val labelsEndingHere = spansEndingHere.map(spanMap.get(_).get(0).getLabel()).toSet
    for (label <- labelsEndingHere) {
      add("SpanEndingHere=" + label)
    }
    // LOWEST PARENT
    var lowestParentFeatsAdded = false
    for (span <- spanMap.keySet().asScala.toSeq.sortBy(span => span.getSecond - span.getFirst)) {
      if (!lowestParentFeatsAdded && span.getFirst <= boundIdx && boundIdx + 1 < span.getSecond) {
        val parents = spanMap.get(span).asScala.filter(_.getChildren.size >= 2)
        if (parents.size >= 1) {
          val parent = parents(0)
          val children = parent.getChildren().asScala
          val childrenSpans = children.map(child => constituents.get(child).getStart() -> (constituents.get(child).getEnd() + 1))
          add("LowestRule=" + parent.getLabel + " -> " + children.map(_.getLabel).reduce(_ + " " + _))
          if (childrenSpans.filter(_._2 == boundIdx + 1).size >= 1) {
            val leftSpanIdx = childrenSpans.zipWithIndex.filter(_._1._2 == boundIdx + 1).map(_._2).head
            add("LowestRuleSplit=" + parent.getLabel + " -> " + children.map(_.getLabel) + " " + leftSpanIdx)
            add("LowestRuleSplitTags=" + parent.getLabel + " -> " + children(leftSpanIdx).getLabel + "-" + doc.pos(sentIdx)(boundIdx) + " | " +
                                                                children(leftSpanIdx + 1).getLabel + "-" + doc.pos(sentIdx)(boundIdx+1))
          }
          lowestParentFeatsAdded = true
        }
      }
    }
  }
  
  private def getDependencySyntacticFeats(doc: ConllDoc, sentIdx: Int, boundIdx: Int, feats: ArrayBuffer[Int], addToIndexer: Boolean) = {
    val tree = doc.trees(sentIdx)
    val words = doc.words(sentIdx)
    val pos = doc.pos(sentIdx)
    val depMap = tree.childParentDepMap
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    def accessPos(idx: Int) = if (idx == -1) "ROOT" else pos(idx)
    add("ParentPoses=" + accessPos(depMap(boundIdx)) + "-" + accessPos(depMap(boundIdx + 1)))
    val beforeChildren = (0 until tree.size).filter(i => depMap(i) == boundIdx)
    val afterChildren = (0 until tree.size).filter(i => depMap(i) == boundIdx + 1)
    // POS unigrams and bigrams of children of the word before the gap
    for (i <- 0 until beforeChildren.size) {
      add("BeforeChild=" + accessPos(beforeChildren(i)))
      if (i < beforeChildren.size - 1) {
        add("BeforeChildrenPair=" + accessPos(beforeChildren(i)) + "-" + accessPos(beforeChildren(i+1)))
      }
    }
    // POS unigrams and bigrams of children of the word after the gap
    for (i <- 0 until afterChildren.size) {
      add("AfterChild=" + accessPos(afterChildren(i)))
      if (i < afterChildren.size - 1) {
        add("AfterChildrenPair=" + accessPos(afterChildren(i)) + "-" + accessPos(afterChildren(i+1)))
      }
    }
    // Pairs of POS tag children across the gap
    for (beforeChild <- beforeChildren) {
      for (afterChild <- afterChildren) {
        add("ChildPair=" + accessPos(beforeChild) + "-" + accessPos(afterChild))
      }
    }
    // Paths to root from predecessor and successor
    val beforeRootPath = getPathToRoot(depMap, boundIdx)
    for (i <- 0 until beforeRootPath.size - 1) {
      add("BeforePathPair=" + accessPos(beforeRootPath(i)) + "-" + accessPos(beforeRootPath(i+1)))
    }
    val afterRootPath = getPathToRoot(depMap, boundIdx + 1)
    for (i <- 0 until afterRootPath.size - 1) {
      add("AfterPathPair=" + accessPos(afterRootPath(i)) + "-" + accessPos(afterRootPath(i+1)))
    }
  }
  
  private def getPathToRoot(parents: HashMap[Int,Int], posn: Int) = {
    val pathToRoot = new ArrayBuffer[Int]
    var currIdx = posn
    while (currIdx != -1) {
      pathToRoot += currIdx
      currIdx = parents(currIdx)
    }
    pathToRoot += -1
    pathToRoot
  }
}

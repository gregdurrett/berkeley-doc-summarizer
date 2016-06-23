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
import edu.berkeley.nlp.entity.ConllDoc
import edu.berkeley.nlp.summ.data.DiscourseDepEx
import edu.berkeley.nlp.summ.LikelihoodAndGradientComputer

@SerialVersionUID(1L)
class EDUSegmenterSemiMarkovFeaturizer(val featIdx: Indexer[String],
                                       val wrappedFeaturizer: EDUSegmenterFeaturizer) extends Serializable {
  
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
  
  def extractFeaturesCached(ex: DiscourseDepEx, addToIndexer: Boolean): Array[Array[Array[Array[Int]]]] = {
    // Only featurize sentence-internal boundaries (sentence boundaries are trivially EDU segments)
    if (ex.cachedEduSemiMarkovFeatures == null) {
      ex.cachedEduSemiMarkovFeatures = extractFeatures(ex.conllDoc, addToIndexer)
    }
    ex.cachedEduSemiMarkovFeatures
  }
  
  def extractFeatures(doc: ConllDoc, addToIndexer: Boolean): Array[Array[Array[Array[Int]]]] = {
    // Features on boundaries from the binary version
    val wrappedFeats = wrappedFeaturizer.extractFeatures(doc, addToIndexer)
    Array.tabulate(doc.numSents)(sentIdx => {
      Array.tabulate(doc.words(sentIdx).size)(startIdx => {
        Array.tabulate(doc.words(sentIdx).size + 1)(endIdx => {
          if (endIdx > startIdx) {
            extractFeatures(doc, sentIdx, startIdx, endIdx, wrappedFeats, addToIndexer)
          } else {
            Array[Int]()
          }
        })
      })
    })
  }
  
  private def extractFeatures(doc: ConllDoc, sentIdx: Int, startIdx: Int, endIdx: Int, wrappedFeats: Array[Array[Array[Int]]], addToIndexer: Boolean): Array[Int] = {
    val feats = new ArrayBuffer[Int]
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    val bucketedLen = wrappedFeaturizer.bucket(endIdx - startIdx)
    if (startIdx > 0) {
      // Don't add these features because they'll be the end of some other span by definition
//      feats ++= wrappedFeaturizer.extractFeatures(doc, sentIdx, endIdx - 1, addToIndexer)
    } else {
      add("StartSent,Len=" + bucketedLen)
    }
    if (endIdx < doc.words(sentIdx).size - 1) {
      feats ++= wrappedFeats(sentIdx)(endIdx - 1)
    } else {
      add("EndSent,Len=" + bucketedLen)
      if (startIdx == 0) {
        add("WholeSent,Len=" + bucketedLen)
      }
    }
    if (endIdx - startIdx == 1) {
      add("SingleWord=" + doc.words(sentIdx)(startIdx))
    } else if (endIdx - startIdx == 2) {
      add("TwoWords,First=" + doc.words(sentIdx)(startIdx))
      add("TwoWords,Second=" + doc.words(sentIdx)(startIdx+1))
    }
    // Look at first and last, also the context words
    val beforePos = if (startIdx == 0) "<S>" else doc.pos(sentIdx)(startIdx-1)
    val firstPos = doc.pos(sentIdx)(startIdx)
    val lastPos = doc.pos(sentIdx)(endIdx - 1)
    val afterPos = if (endIdx == doc.pos(sentIdx).size) "</S>" else doc.pos(sentIdx)(endIdx)
    add("FirstLastPOS=" + firstPos + "-" + afterPos)
    add("BeforeAfterPOS=" + beforePos + "-" + afterPos)
//    add("BFLAPOS=" + beforePos + "-" + firstPos + "-" + lastPos + "-" + afterPos)
    var dominatingConstituents = doc.trees(sentIdx).getAllConstituentTypes(startIdx, endIdx)
    if (dominatingConstituents.isEmpty) {
      dominatingConstituents = Seq("None")
    } else {
      // None of these dependency features seem to help
//      // We have a valid span, fire features on dependencies
//      val headIdx = doc.trees(sentIdx).getSpanHead(startIdx, endIdx)
////      add("HeadWord=" + doc.words(sentIdx)(headIdx))
//      add("HeadPos=" + doc.pos(sentIdx)(headIdx))
//      val parentIdx = doc.trees(sentIdx).childParentDepMap(headIdx)
//      if (parentIdx == -1) {
//        add("Parent=ROOT")
//      } else {
////        add("ParentWord=" + doc.words(sentIdx)(parentIdx))
//        add("ParentPos=" + doc.pos(sentIdx)(parentIdx))
//        add("ParentDist=" + Math.signum(parentIdx - headIdx) + ":" + wrappedFeaturizer.bucket(parentIdx - headIdx))
//      }
    }
    // Fire features on constituent labels (or None if it isn't a constituent)
    for (constituent <- dominatingConstituents) {
      add("DominatingConstituent=" + constituent)
      add("DominatingConstituentLength=" + constituent + "-" + bucketedLen)
      add("DominatingConstituentBefore=" + constituent + "-" + beforePos)
      add("DominatingConstituentAfter=" + constituent + "-" + afterPos)
      // This makes it way slower and doesn't help
//      val maybeParent = doc.trees(sentIdx).getParent(startIdx, endIdx)
//      if (!maybeParent.isDefined) {
//        add("DominatingParent=None")
//      } else {
//        val (parent, childIdx) = maybeParent.get
//        val childrenStr = (0 until parent.getChildren().size).map(i => (if (childIdx == i) ">" else "") + parent.getChildren().get(i).getLabel()).foldLeft("")(_ + " " + _)
////        Logger.logss(parent.getLabel() + " ->" + childrenStr)
////        add("DominatingRule=" + parent.getLabel() + " ->" + childrenStr)
//        add("DominatingParent=" + parent.getLabel() + " -> " + constituent)
//      }
    }
    feats.toArray
  }
}

@SerialVersionUID(1L)
class EDUSegmenterSemiMarkovComputer(val featurizer: EDUSegmenterSemiMarkovFeaturizer,
                                     val wholeSpanLossScale: Double = 4.0) extends LikelihoodAndGradientComputer[DiscourseDepEx] with Serializable {
  
  def getInitialWeights(initialWeightsScale: Double): Array[Double] = Array.tabulate(featurizer.featIdx.size)(i => 0.0)
  
  def accumulateGradientAndComputeObjective(ex: DiscourseDepEx, weights: Array[Double], gradient: Array[Double]): Double = {
    val (predSegs, predScore) = decode(ex, weights, 1.0);
//    val recomputedPredScore = scoreParse(ex, weights, predParents, 1.0)
    val goldSegs = ex.goldEduSpans
    val goldScore = scoreSegmentation(ex, weights, goldSegs, 1.0)
//    Logger.logss("Pred score: " + predScore + ", recomputed pred score: " + recomputedPredScore + ", gold score: " + goldScore)
    for (sentIdx <- 0 until ex.conllDoc.numSents) {
      for (startIdx <- 0 until ex.conllDoc.words(sentIdx).size) {
        for (endIdx <- startIdx + 1 to ex.conllDoc.words(sentIdx).size) {
          val seg = startIdx -> endIdx
          val increment = (if (goldSegs(sentIdx).contains(seg)) 1 else 0) + (if (predSegs(sentIdx).contains(seg)) -1 else 0)
          if (increment != 0) {
            val feats = ex.cachedEduSemiMarkovFeatures(sentIdx)(startIdx)(endIdx)
            for (feat <- feats) {
              gradient(feat) += increment
            }
          }
        }
      }
    }
    predScore - goldScore
  }
  
  def computeObjective(ex: DiscourseDepEx, weights: Array[Double]): Double = accumulateGradientAndComputeObjective(ex, weights, Array.tabulate(weights.size)(i => 0.0))
  
  def decode(ex: DiscourseDepEx, weights: Array[Double]): Array[Array[Boolean]] = {
    EDUSegmenterSemiMarkov.convertSegsToBooleanArray(decode(ex, weights, 0)._1)
  }
  
  def decode(ex: DiscourseDepEx, weights: Array[Double], lossWeight: Double): (Array[Seq[(Int,Int)]], Double) = {
    val feats = featurizer.extractFeaturesCached(ex, false)
    var cumScore = 0.0
    val allPreds = Array.tabulate(ex.conllDoc.numSents)(sentIdx => {
      val result = decodeSentence(feats(sentIdx), ex.conllDoc.words(sentIdx).size, weights, lossWeight, Some(ex.goldEduSpans(sentIdx)))
      cumScore += result._2
      result._1
    })
    (allPreds, cumScore)
  }
  
  def decodeSentence(feats: Array[Array[Array[Int]]], sentLen: Int, weights: Array[Double], lossWeight: Double, goldSpans: Option[Seq[(Int,Int)]]): (Seq[(Int,Int)], Double) = {
    val chart = Array.tabulate(sentLen + 1)(i => if (i == 0) 0.0 else Double.NegativeInfinity)
    val backptrs = Array.tabulate(sentLen + 1)(i => -1)
    for (endIdx <- 1 to sentLen) {
      for (startIdx <- 0 until endIdx) {
        val isGold = if (goldSpans.isDefined) goldSpans.get.contains(startIdx -> endIdx) else false
        val lossScore = if (!isGold) {
          if (startIdx == 0 && endIdx == sentLen) {
//            lossWeight
            lossWeight * wholeSpanLossScale
          } else {
            lossWeight
          }
        } else {
          0.0
        }
        val score = ClassifyUtils.scoreIndexedFeats(feats(startIdx)(endIdx), weights) + lossScore
        if (chart(startIdx) + score > chart(endIdx)) {
          backptrs(endIdx) = startIdx
          chart(endIdx) = chart(startIdx) + score
        }
      }
    }
    // Recover the gold derivation
    val pairs = new ArrayBuffer[(Int,Int)]
    var ptr = sentLen
    while (ptr > 0) {
      pairs.prepend(backptrs(ptr) -> ptr)
      ptr = backptrs(ptr)
    }
    (pairs.toSeq, chart(sentLen))
  }
  
  private def scoreSegmentation(ex: DiscourseDepEx, weights: Array[Double], segmentation: Seq[Seq[(Int,Int)]], lossWeight: Double) = {
    var score = 0.0
    val feats = featurizer.extractFeaturesCached(ex, false)
    for (sentIdx <- 0 until ex.conllDoc.numSents) {
      for (segment <- segmentation(sentIdx)) {
        val isGold = ex.goldEduSpans(sentIdx).contains(segment)
        score += ClassifyUtils.scoreIndexedFeats(feats(sentIdx)(segment._1)(segment._2), weights) + (if (!isGold) lossWeight else 0.0)
      }
    }
    score
  }
}

@SerialVersionUID(1L)
class EDUSegmenterSemiMarkov(val computer: EDUSegmenterSemiMarkovComputer,
                             val weights: Array[Double]) extends EDUSegmenter {
  
  def decode(ex: DiscourseDepEx) = computer.decode(ex, weights)
  
  def decode(doc: ConllDoc) = {
    val feats = computer.featurizer.extractFeatures(doc, false)
    val result = Array.tabulate(feats.size)(i => {
      computer.decodeSentence(feats(i), doc.words(i).size, weights, 0.0, None)._1
    })
    EDUSegmenterSemiMarkov.convertSegsToBooleanArray(result)
  }
}

object EDUSegmenterSemiMarkov {
  
  def convertSegsToBooleanArray(segments: Seq[Seq[(Int,Int)]]): Array[Array[Boolean]] = {
    Array.tabulate(segments.size)(i => {
      val seq = segments(i)
      val starts = seq.map(_._1)
      Array.tabulate(seq.last._2 - 1)(i => starts.contains(i+1))
    })
  }
}

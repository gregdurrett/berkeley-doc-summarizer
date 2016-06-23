package edu.berkeley.nlp.summ

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer

import edu.berkeley.nlp.entity.GUtil
import edu.berkeley.nlp.entity.coref.PairwiseScorer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.IntCounter
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.summ.data.DepParseDoc
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed
import edu.berkeley.nlp.summ.data.DiscourseTree
import edu.berkeley.nlp.summ.data.FragilePronoun
import edu.berkeley.nlp.summ.data.PronounReplacement
import edu.berkeley.nlp.summ.data.StopwordDict
import edu.berkeley.nlp.summ.data.SummDoc

/**
 * Top-level interface to the main summarizer. This is the object that gets serialized/deserialized
 * and is distributed with the system.
 * 
 * @author gdurrett
 */
@SerialVersionUID(1L)
class CompressiveAnaphoraSummarizer(val computer: CompressiveAnaphoraSummarizerComputer,
                                    val weights: AdagradWeightVector) extends DiscourseDepExSummarizer {
  
  var numBadPronounsAt5 = 0
  var numBadPronounsAt8 = 0
  var numSummaries = 0
  var numCutsEqViolated = 0
  var numRestarts = 0
  
  def summarize(ex: DiscourseDepExProcessed, budget: Int, oneSentPerLine: Boolean): Seq[String] = {
    val (edus, prons) = computer.decode(ex, weights, budget)
    val allPronReplacements = if (computer.doPronounReplacement) computer.identifyPronounReplacements(ex) else Seq[PronounReplacement]()
    val fragileProns = if (computer.useFragilePronouns) ex.identifyFragilePronouns(computer.corefPredictor) else Seq[FragilePronoun]()
    val usedPronReps = prons.map(allPronReplacements(_))
    for (fragilePron <- fragileProns) {
      val isFragilePresent = edus.contains(fragilePron.eduIdx) 
      val areAntecedentsPresent = fragilePron.antecedentEdus.map(edus.contains(_)).foldLeft(true)(_ && _)
      val isReplacementDone = usedPronReps.filter(_.mentIdx == fragilePron.mentIdx).size > 0
      if (isFragilePresent && (!areAntecedentsPresent && !isReplacementDone)) {
        numBadPronounsAt5 += 1
      }
    }
    val parents = ex.getParents(computer.discourseType)
    for (i <- 0 until ex.eduAlignments.size) {
      if (parents(i) != -1 && !edus.contains(i) && edus.contains(parents(i)) && ex.parentLabels(i).startsWith("=")) {
        numCutsEqViolated += 1
      }
      if (parents(i) != -1 && edus.contains(i) && !edus.contains(parents(i))) {
        numRestarts += 1
      }
    }
    numSummaries += 1
    // END PRONOUN ANALYSIS CODE
    ex.getSummaryTextWithPronounsReplaced(edus, prons.map(allPronReplacements(_)), oneSentPerLine)
  }
  
  def summarizeOracle(ex: DiscourseDepExProcessed, budget: Int): Seq[String] = {
    val (edus, prons, _, _, _, _) = computer.decodeOracleBigramRecall(ex, budget)
    val allPronReplacements = computer.identifyPronounReplacements(ex)
    ex.getSummaryTextWithPronounsReplaced(edus, prons.map(allPronReplacements(_)), false)
  }
  
  def display(ex: DiscourseDepExProcessed, budget: Int) {
    val (edus, prons) = computer.decode(ex, weights, budget)
    val allPronReplacements = if (computer.doPronounReplacement) computer.identifyPronounReplacements(ex) else Seq[PronounReplacement]()
    val usedPronReplacements = prons.map(allPronReplacements(_))
    val summary = ex.getSummaryTextWithPronounsReplaced(edus, usedPronReplacements, false)
    val edusWithPronsReplaced = usedPronReplacements.map(_.eduIdx).toSet
    
    // Display coref posteriors
    val indicesOfInterest = usedPronReplacements.map(_.mentIdx)
    val corefPosteriors = if (computer.corefPredictor.isDefined) {
      CorefUtils.computePosteriors(ex.rawDoc.corefDoc, computer.corefPredictor.get, indicesOfInterest)
    } else {
      Array.tabulate(indicesOfInterest.size)(i => Array.fill(i+1)(0.0))
    }
    val parents = ex.getParents(computer.discourseType)
    for (i <- 0 until edus.size) {
      val pronStr = if (edusWithPronsReplaced.contains(edus(i))) {
        val usedPronIndicesThisEdu = usedPronReplacements.zipWithIndex.filter(_._1.eduIdx == edus(i)).map(_._2)
        var str = " ("
        for (pronIdx <- usedPronIndicesThisEdu) {
          val pronRep = usedPronReplacements(pronIdx)
          val antecedents = ex.rawDoc.corefDoc.goldClustering.getCluster(pronRep.mentIdx).filter(_ < pronRep.mentIdx)
//          val cumPr = antecedents.map(corefPosteriors(pronIdx)(_)).foldLeft(Double.NegativeInfinity)(SloppyMath.logAdd(_, _))
          val cumLogPr = if (computer.corefPredictor.isDefined) pronRep.computeCorefClusterLogPosterior(ex.rawDoc.corefDoc, computer.corefPredictor.get) else 0.0
          str += pronRep.render(ex.rawDoc.corefDoc) + " -- Pr=" + GUtil.fmtProb(Math.exp(corefPosteriors(pronIdx)(pronRep.antIdx))) + ", CumPr=" + GUtil.fmtProb(Math.exp(cumLogPr))
          if (pronIdx != usedPronIndicesThisEdu.last) {
            str += ", "
          }
        }
        str += "): "
        str
      } else {
        ": "
      }
      val fragileProns = if (ex.cachedFragilePronouns == null) Seq[FragilePronoun]() else ex.cachedFragilePronouns.filter(_.eduIdx == edus(i))
      var fragilePronStr = ""
      for (fragilePron <- fragileProns) {
        val pronText = CorefUtils.getMentionText(ex.rawDoc.corefDoc.goldMentions(fragilePron.mentIdx)).foldLeft("")(_ + " " + _).trim
        val antText = if (fragilePron.antecedentMentIndices.isDefined) {
          fragilePron.antecedentMentIndices.get.map(antMentIdx => CorefUtils.getMentionText(ex.rawDoc.corefDoc.goldMentions(antMentIdx)).foldLeft("")(_ + " " + _).trim)
        } else {
          ""
        }
        fragilePronStr = "  fragile pronoun: " + pronText + ", depends on EDUs: " + fragilePron.antecedentEdus + " from depending on mentions: " + antText
      }
      Logger.logss(edus(i) + pronStr + summary(i))
      if (!fragilePronStr.isEmpty) Logger.logss(fragilePronStr)
      if (parents(edus(i)) != -1 && !edus.contains(parents(edus(i)))) Logger.logss("  (restart)")
      val cutEqChildren = (0 until parents.size).filter(j => parents(j) == edus(i) && ex.parentLabels(j).startsWith("=") && !edus.contains(j))
      if (cutEqChildren.size > 0) Logger.logss("  (cut eq link: " + cutEqChildren + " " + cutEqChildren.map(ex.parentLabels(_)) + ")")
    }
    Logger.logss("Tokens (approximated with whitespace): " + summary.map(_.split("\\s+").size).foldLeft(0)(_ + _) + " / budget of " + budget)
  }
  
  def printStatistics() {
    Logger.logss("Num bad pronouns at thresholds of 0.5 / 0.8: " + numBadPronounsAt5 + " / " + numBadPronounsAt8 + " out of " + numSummaries + " summaries")
    Logger.logss("Num cuts EQ violated: " + numCutsEqViolated + " out of " + numSummaries + " summaries")
    Logger.logss("Num restarts: " + numRestarts + " out of " + numSummaries + " summaries")
  }
}

/**
 * Handles decoding and computing gradients for the main summarizer. Options here are
 * passed into the ILP for both normal and loss-augmented decodes.
 */
@SerialVersionUID(1L)
class CompressiveAnaphoraSummarizerComputer(val featurizer: CompressiveAnaphoraFeaturizer,
                                            val fixedBudget: Int,
                                            val budgetScale: Double,
                                            val discourseType: String,
                                            val numEqualsConstraints: Int,
                                            val numParentConstraints: Int,
                                            val doPronounReplacement: Boolean,
                                            val doPronounConstraints: Boolean,
                                            val useFragilePronouns: Boolean,
                                            val replaceWithNEOnly: Boolean,
                                            val corefPredictor: Option[PairwiseScorer],
                                            val corefConfidenceThreshold: Double,
                                            val useUnigramRouge: Boolean) extends LikelihoodAndGradientComputerSparse[DiscourseDepExProcessed] with Serializable {
  
  def getInitialWeights(initialWeightsScale: Double): Array[Double] = Array.tabulate(featurizer.featIdx.size)(i => 0.0)
  
  def accumulateGradientAndComputeObjective(ex: DiscourseDepExProcessed, weights: AdagradWeightVector, gradient: IntCounter): Double = {
    val budget = if (fixedBudget > 0) fixedBudget else (ex.summary.map(_.getWords.size).foldLeft(0)(_ + _) * budgetScale).toInt
    // Oracle decode using bigram recall: the ILP maximizes score, so by plugging in bigram recall we get the oracle
    // If trainType 
    val (goldEdus, goldProns, goldBigrams, goldCuts, goldRestarts, goldLossScore) = decodeOracleBigramRecall(ex, budget)
    // Negate bigram recall scores for loss-augmented scoring
    // This is an approximation for proj rouge but should be basically right...
    val realGoldLossScore = (goldLossScore + 0.5).toInt
    val goldScore = scoreGoldSummary(ex, goldEdus, goldProns, goldBigrams, goldCuts, goldRestarts, realGoldLossScore, weights)
    // Compute prediction; changes depending on the loss type we're using
    val summBigrams = ex.getSummBigrams(useUnigramRouge)
    val (predEdus, predProns, predBigrams, predCuts, predRestarts, predScore) = decode(ex, weights, budget, 1.0, Some(summBigrams))
    // Apply gradient
    val eduFeats = featurizer.extractFeaturesCached(ex, false);
    for (i <- 0 until ex.eduAlignments.size) {
      val factor = (if (goldEdus.contains(i)) 1 else 0) + (if (predEdus.contains(i)) -1 else 0)
      if (factor != 0) {
        var featIdx = 0
        while (featIdx < eduFeats(i).size) {
          gradient.incrementCount(eduFeats(i)(featIdx), factor)
          featIdx += 1
        }
      }
    }
    val pronFeats = featurizer.extractPronounFeaturesCached(ex, identifyPronounReplacements(ex), false)
    for (i <- 0 until pronFeats.size) {
      val factor = (if (goldProns.contains(i)) 1 else 0) + (if (predProns.contains(i)) -1 else 0)
      if (factor != 0) {
        var featIdx = 0
        while (featIdx < pronFeats(i).size) {
          gradient.incrementCount(pronFeats(i)(featIdx), factor)
          featIdx += 1
        }
      }
    }
    if (featurizer.featSpec.contains("ngramtype")) {
      val docBigrams = ex.getDocBigramsSeq(useUnigramRouge)
      val bigramFeats = featurizer.extractBigramFeaturesCached(ex, docBigrams, false)
      for (i <- 0 until docBigrams.size) {
        val factor = (if (goldBigrams.contains(i)) 1 else 0) + (if (predBigrams.contains(i)) -1 else 0)
        if (factor != 0) {
          var featIdx = 0
          while (featIdx < bigramFeats(i).size) {
            gradient.incrementCount(bigramFeats(i)(featIdx), factor)
            featIdx += 1
          }
        }
      }
    }
    predScore - goldScore
  }
  
  def identifyPronounReplacements(ex: DiscourseDepExProcessed) = {
    ex.identifyPronounReplacements(replaceWithNEOnly, corefPredictor, corefConfidenceThreshold)
  }
  
  def computeObjective(ex: DiscourseDepExProcessed, weights: AdagradWeightVector): Double = accumulateGradientAndComputeObjective(ex, weights, new IntCounter)
  
  def decode(ex: DiscourseDepExProcessed, weights: AdagradWeightVector, budget: Int): (Seq[Int], Seq[Int]) = {
    val results = decode(ex, weights, budget, 0, None)
    results._1 -> results._2
  }
  
  // 5-tuple: EDUs, pron reps, bigrams, cuts, restarts
  private def decode(ex: DiscourseDepExProcessed, weights: AdagradWeightVector, budget: Int, lossWeight: Double,
                     summBigrams: Option[Set[(String,String)]]): (Seq[Int], Seq[Int], Seq[Int], Seq[Int], Seq[Int], Double) = {
    val feats = featurizer.extractFeaturesCached(ex, false)
    // Negative bigram recall => lower recall is better, which is correct for loss augmented decoding
    val allPronReplacements = if (doPronounReplacement) identifyPronounReplacements(ex) else Seq[PronounReplacement]()
    val pronReplacementFeats = featurizer.extractPronounFeaturesCached(ex, allPronReplacements, false)
    val leafScoresRaw = (0 until ex.eduAlignments.size).map(leafIdx => weights.score(feats(leafIdx)))
    val leafScores = DiscourseDepExSummarizer.biasTowardsEarlier(leafScoresRaw)
    val pronReplacementScores = (0 until allPronReplacements.size).map(pronIdx => weights.score(pronReplacementFeats(pronIdx)))
    val bigramSeq = ex.getDocBigramsSeq(useUnigramRouge)
    val bigramFeats = featurizer.extractBigramFeaturesCached(ex, bigramSeq, false)
    val bigramScores = if (summBigrams.isDefined) {
      (0 until bigramSeq.size).map(bigramIdx => weights.score(bigramFeats(bigramIdx)) + (if (summBigrams.get.contains(bigramSeq(bigramIdx))) -lossWeight else 0.0))
    } else {
      (0 until bigramSeq.size).map(bigramIdx => weights.score(bigramFeats(bigramIdx)))
    }
    val fragilePronouns = if (useFragilePronouns) ex.identifyFragilePronouns(corefPredictor) else Seq[FragilePronoun]()
    val (maybeCutFeatures, maybeRestartFeatures) = (None, None)
    val results = CompressiveAnaphoraSummarizerILP.summarizeILPWithGLPK(ex, ex.getParents(discourseType), leafScores, bigramScores, allPronReplacements, pronReplacementScores, budget,
                                                                        numEqualsConstraints, numParentConstraints, doPronounConstraints, fragilePronouns, useUnigramRouge)
    results
  }
  
  def decodeOracleBigramRecall(ex: DiscourseDepExProcessed, budget: Int): (Seq[Int], Seq[Int], Seq[Int], Seq[Int], Seq[Int], Double) = {
    val allPronReplacements = if (doPronounReplacement) identifyPronounReplacements(ex) else Seq[PronounReplacement]()
    val fragilePronouns = if (useFragilePronouns) ex.identifyFragilePronouns(corefPredictor) else Seq[FragilePronoun]()
    val leafScores = DiscourseDepExSummarizer.biasTowardsEarlier(Array.fill(ex.eduAlignments.size)(0.0))
    val pronReplacementScores = Array.fill(allPronReplacements.size)(0.0)
    val bigramSeq = ex.getDocBigramsSeq(useUnigramRouge)
    val summBigrams = ex.getSummBigrams(useUnigramRouge)
    val bigramScores = bigramSeq.map(bigram => if (summBigrams.contains(bigram)) 1.0 else 0.0)
    val results = CompressiveAnaphoraSummarizerILP.summarizeILPWithGLPK(ex, ex.getParents(discourseType), leafScores, bigramScores, allPronReplacements, pronReplacementScores, budget,
                                                                        numEqualsConstraints, numParentConstraints, doPronounConstraints, fragilePronouns, useUnigramRouge)
    results
  }
  
  
  private def scoreGoldSummary(ex: DiscourseDepExProcessed,
                               summary: Seq[Int],
                               pronReplacements: Seq[Int],
                               bigrams: Seq[Int],
                               cuts: Seq[Int],
                               restarts: Seq[Int],
                               goldLossScore: Double,
                               weights: AdagradWeightVector) = {
    val eduFeats = featurizer.extractFeaturesCached(ex, false);
    val allPronReplacements = identifyPronounReplacements(ex)
    val pronFeats = featurizer.extractPronounFeaturesCached(ex, allPronReplacements, false);
    val docBigrams = ex.getDocBigramsSeq(useUnigramRouge)
    val summBigrams = ex.getSummBigrams(useUnigramRouge)
    val bigramFeats = featurizer.extractBigramFeaturesCached(ex, docBigrams, false);
    var score = 0.0
    for (eduIdx <- summary) {
      score += weights.score(eduFeats(eduIdx))
    }
    for (pronReplacement <- pronReplacements) {
      score += weights.score(pronFeats(pronReplacement))
      score -= ex.getBigramRecallDelta(allPronReplacements(pronReplacement), useUnigramRouge)
    }
    for (bigramIdx <- bigrams) {
      score += weights.score(bigramFeats(bigramIdx))
    }
    score - goldLossScore
  }
}

/**
 * Feature extraction for all of the decisions that end up being featurized in the 
 */
@SerialVersionUID(1L)
class CompressiveAnaphoraFeaturizer(val featIdx: Indexer[String],
                                    val featSpec: Set[String],
                                    val discourseType: String,
                                    val wordCounts: Counter[String],
                                    val lexicalCountCutoff: Int = 1) extends Serializable {
  
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
  val bucketBoundaries = Array(0, 1, 2, 3, 4, 5, 8, 16, 32, 64)
  
  def bucket(num: Int) = {
    var i = 0;
    while (i < bucketBoundaries.size && bucketBoundaries(i) < num) {
      i += 1
    }
    i
  }
  
  def bucket(num: Double) = {
    var i = 0;
    while (i < bucketBoundaries.size && bucketBoundaries(i) < num) {
      i += 1
    }
    i
  }
  
  def extractFeaturesCached(ex: DiscourseDepExProcessed, addToIndexer: Boolean): Array[Array[Int]] = {
    if (ex.cachedSummFeats == null) {
      ex.cachedSummFeats = extractFeatures(ex, addToIndexer)
    }
    ex.cachedSummFeats
  }
  
  def extractFeatures(ex: DiscourseDepExProcessed, addToIndexer: Boolean) = {
    Array.tabulate(ex.eduAlignments.size)(idx => {
      val span = ex.eduAlignments(idx)
      val docSentWords = ex.rawDoc.doc(span._1._1).getWords.slice(span._1._2, span._2._2)
      val docSentPoss = ex.rawDoc.doc(span._1._1).getPoss.slice(span._1._2, span._2._2)
      val feats = extractBasicFeatures(featSpec, wordCounts, lexicalCountCutoff, ex.cachedWordCounts, span._1._1, docSentWords, docSentPoss, addToIndexer)
      feats ++= extractFancyFeatures(featSpec, wordCounts, lexicalCountCutoff, ex.rawDoc, span._1._1, span._1._2, span._2._2, addToIndexer)
      extractDiscourseFeatures(ex, idx, feats, wordCounts, addToIndexer)
      feats.toArray
    })
  }
  
  def extractDiscourseFeatures(ex: DiscourseDepExProcessed, idx: Int, feats: ArrayBuffer[Int], wordCounts: Counter[String], addToIndexer: Boolean, prefix: String = "") = {
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    val sentIdx = ex.eduAlignments(idx)._1._1
    val wordIdxStart = ex.eduAlignments(idx)._1._2
    val wordIdxEnd = ex.eduAlignments(idx)._2._2
    if (featSpec.contains("edushape")) {
      val otherEdusInSameSentence = ex.eduAlignments.filter(alignment => alignment._1._1 == sentIdx)
      val myIdx = otherEdusInSameSentence.indexOf(ex.eduAlignments(idx))
      add(prefix + "StatusPosn=" + myIdx + "/" + otherEdusInSameSentence.size + "-" + bucket(sentIdx))
    }
    if (featSpec.contains("edulex")) {
      add(prefix + "PrecedingWord=" + (if (wordIdxStart > 0) wordOrUnk(ex.doc(sentIdx).getWord(wordIdxStart-1), wordCounts, lexicalCountCutoff) else "<S>"))
      add(prefix + "FirstWord=" + wordOrUnk(ex.doc(sentIdx).getWord(wordIdxStart), wordCounts, lexicalCountCutoff))
      add(prefix + "LastWord=" + wordOrUnk(ex.doc(sentIdx).getWord(wordIdxEnd - 1), wordCounts, lexicalCountCutoff))
      add(prefix + "FollowingWord=" + (if (wordIdxEnd < ex.doc(sentIdx).size) wordOrUnk(ex.doc(sentIdx).getWord(wordIdxEnd), wordCounts, lexicalCountCutoff) else "</S>"))
    }
    if (featSpec.contains("discourse")) {
      val parents = ex.getParents(discourseType)
      val depth = DiscourseTree.computeDepth(parents, ex.parentLabels, true, idx)
      val numDominated = DiscourseTree.computeNumDominated(parents, idx)
      add(prefix + "DepthPosn=" + bucket(depth) + "-" + bucket(sentIdx))
      add(prefix + "DominatedPosn=" + bucket(numDominated) + "-" + bucket(sentIdx))
      val signedDistToParent = parents(idx) - idx
      if (parents(idx) != -1) {
        add(prefix + "ParentDirDistPosn=" + bucket(signedDistToParent) + "-" + Math.signum(signedDistToParent) + "-" + bucket(sentIdx))
      } else {
        add(prefix + "Parent=ROOT")
      } 
    }
    if (featSpec.contains("hirao")) {
      val parents = ex.getParents(discourseType)
      val depth = DiscourseTree.computeDepth(parents, ex.parentLabels, true, idx)
      var count = 0.0
      val docSentWords = ex.rawDoc.doc(sentIdx).getWords.slice(wordIdxStart, wordIdxEnd)
      val docSentPoss = ex.rawDoc.doc(sentIdx).getPoss.slice(wordIdxStart, wordIdxEnd)
      for (i <- 0 until docSentWords.size) {
        if (isContentToken(docSentPoss(i))) {
          val word = docSentWords(i)
          val pos = docSentPoss(i)
          count += Math.log(1 + wordCounts.getCount(word))/Math.log(2)
        }
      }
      add(prefix + "RawEduScore=" + bucket(count))
      add(prefix + "TKPEduScore=" + bucket(count/depth))
      add(prefix + "EduScoreDepth=" + bucket(count) + "-" + bucket(depth))
    }
    if (featSpec.contains("types")) {
      val parents = ex.getParents(discourseType)
      val labels = ex.parentLabels
      val depth = DiscourseTree.computeDepth(parents, ex.parentLabels, true, idx)
      // TODO: Explore other conjunctions?
      if (parents(idx) != -1) {
        val label = ex.parentLabels(idx)
        val signedDistToParent = parents(idx) - idx
        add(prefix + "ParentDirDistType=" + bucket(signedDistToParent) + "-" + Math.signum(signedDistToParent) + "-" + label)
        add(prefix + "ParentDirDistTypePosn=" + bucket(signedDistToParent) + "-" + Math.signum(signedDistToParent) + "-" + label + "-" + bucket(sentIdx))
      }
    }
  }
  
  def extractPronounFeaturesCached(ex: DiscourseDepExProcessed, pronReplacements: Seq[PronounReplacement], addToIndexer: Boolean): Array[Array[Int]] = {
    if (ex.cachedPronFeats == null) {
//      val pronReplacements = ex.identifyPronounReplacements
      ex.cachedPronFeats = Array.tabulate(pronReplacements.size)(i => extractPronounFeatures(ex, pronReplacements(i), addToIndexer))
    }
    ex.cachedPronFeats
  }
  
  private def extractPronounFeatures(ex: DiscourseDepExProcessed, pronReplacement: PronounReplacement, addToIndexer: Boolean): Array[Int] = {
    val feats = new ArrayBuffer[Int]
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    // Features: types of the two things
    add("PRIndicator")
    if (featSpec.contains("replacement")) {
      val currMent = ex.rawDoc.corefDoc.goldMentions(pronReplacement.mentIdx)
      val antMent = ex.rawDoc.corefDoc.goldMentions(pronReplacement.antIdx)
      add("PRCurrPron=" + CorefUtils.getMentionText(currMent).map(_.toLowerCase))
      add("PRAntType=" + antMent.mentionType)
      add("PRAntSentDist=" + bucket(antMent.sentIdx - currMent.sentIdx))
      add("PRLen=" + bucket(antMent.endIdx - antMent.startIdx))
    }
    feats.toArray
  }
  
  def extractBigramFeaturesCached(ex: DiscourseDepExProcessed, bigrams: Seq[(String,String)], addToIndexer: Boolean): Array[Array[Int]] = {
    if (ex.cachedBigramFeats == null) {
      ex.cachedBigramFeats = Array.tabulate(bigrams.size)(i => extractBigramFeatures(ex, bigrams(i), addToIndexer))
    }
    ex.cachedBigramFeats
  }
  
  def extractBigramFeatures(ex: DiscourseDepExProcessed, bigram: (String,String), addToIndexer: Boolean): Array[Int] = {
    val feats = new ArrayBuffer[Int]
    def add(feat: String) = maybeAdd(feats, addToIndexer, feat)
    if (featSpec.contains("ngramtype")) {
      require(bigram._2 == "", "Type-level n-gram features aren't implemented for bigrams right now!")
      val unigram = bigram._1
      val occurrenceIndices = (0 until ex.rawDoc.doc.size).filter(sentIdx => ex.rawDoc.doc(sentIdx).getWords.contains(unigram))
      require(!occurrenceIndices.isEmpty)
      val firstSentIdx = occurrenceIndices.head
      val tokIdx = ex.rawDoc.doc(firstSentIdx).getWords.indexOf(unigram)
      add("BIWord=" + wordOrUnk(unigram, wordCounts, lexicalCountCutoff))
      add("BITag=" + ex.rawDoc.doc(firstSentIdx).getPos(tokIdx))
      val bucketedCount = occurrenceIndices.map(sentIdx => ex.rawDoc.doc(sentIdx).getWords.filter(_ == unigram).size).foldLeft(0)(_ + _)
      add("BIPosn=" + bucket(firstSentIdx))
      add("BICount=" + bucket(bucketedCount))
    }
    feats.toArray
  }
  
  def extractBasicFeatures[D<:DepParseDoc](featSpec: Set[String],
                                           wordCounts: Counter[String],
                                           lexicalCountCutoff: Int,
                                           docWordCounts: Counter[String],
                                           sentIdx: Int,
                                           docSentWords: Seq[String],
                                           docSentPoss: Seq[String],
                                           addToIndexer: Boolean): ArrayBuffer[Int] = {
    val feats = new ArrayBuffer[Int];
//    val sent = ex.rawDoc.doc(sentIdx)
//    val anchoredCaseframesThisSent = ex.docCaseframes(sentIdx)
//    val caseframesThisSent = anchoredCaseframesThisSent.map(_.caseframe)
    // Positional features
//    maybeAdd(feats, addToIndexer, "SentPosn=" + sentIdx)
//    maybeAdd(feats, addToIndexer, "OverallPosn=" + ex.overallCaseframeIdx(sentIdx, cfIdx))
    if (featSpec.contains("position")) {
      maybeAdd(feats, addToIndexer, "SentPosn=" + bucket(sentIdx))
//      maybeAdd(feats, addToIndexer, "OverallPosn=" + bucket(ex.overallCaseframeIdx(sentIdx, cfIdx)))
    }
    if (featSpec.contains("compression")) {
      // TODO: ADD COMPRESSION FEATURES
    }
    if (featSpec.contains("basiclex")) {
      for (i <- 0 until docSentWords.size) {
        if (isContentToken(docSentPoss(i))) {
          val word = docSentWords(i)
          val pos = docSentPoss(i)
          maybeAdd(feats, addToIndexer, "Word=" + wordOrUnk(word, wordCounts, lexicalCountCutoff))
          maybeAdd(feats, addToIndexer, "Pos=" + pos)
        }
      }
    }
    if (featSpec.contains("posnbasiclex")) {
      for (i <- 0 until docSentWords.size) {
        if (isContentToken(docSentPoss(i))) {
          val word = docSentWords(i)
          val pos = docSentPoss(i)
          maybeAdd(feats, addToIndexer, "SentPosnWord=" + bucket(sentIdx) + "-" + wordOrUnk(word, wordCounts, lexicalCountCutoff))
          maybeAdd(feats, addToIndexer, "SentPosnPos=" + bucket(sentIdx) + "-" + pos)
        }
      }
    }
    if (featSpec.contains("centrality")) {
      // Look at duplicates of N*, V*
      for (i <- 0 until docSentWords.size) {
        if (isContentToken(docSentPoss(i))) {
          val word = docSentWords(i)
          val pos = docSentPoss(i)
          maybeAdd(feats, addToIndexer, "WordFreqPos=" + bucket(docWordCounts.getCount(word).toInt) + "-" + pos)
        }
      }
    }
    if (featSpec.contains("centralitylex")) {
      // Look at duplicates of N*, V*
      for (i <- 0 until docSentWords.size) {
        if (isContentToken(docSentPoss(i))) {
          val word = docSentWords(i)
          val pos = docSentPoss(i)
          maybeAdd(feats, addToIndexer, "SentPosnWordFreq=" + bucket(sentIdx) + "-" + bucket(docWordCounts.getCount(word).toInt))
          maybeAdd(feats, addToIndexer, "WordFreqWordIden=" + bucket(docWordCounts.getCount(word).toInt) + "-" + wordOrUnk(word, wordCounts, lexicalCountCutoff))
        }
      }
    }
    if (featSpec.contains("firstword")) {
      maybeAdd(feats, addToIndexer, "FirstWord=" + wordOrUnk(docSentWords(0), wordCounts, lexicalCountCutoff))
    }
    if (featSpec.contains("config")) {
      maybeAdd(feats, addToIndexer, "SentLenPosn=" + bucket(docSentWords.size) + "-" + bucket(sentIdx))
    }
//    if (featSpec.contains("lexical")) {
//      for (cf <- caseframesThisSent) {
//        maybeAdd(feats, addToIndexer, "Pred=" + wordOrUnk(cf.predWord, wordCounts, lexicalCountCutoff))
//        maybeAdd(feats, addToIndexer, "PredSentPosn=" + wordOrUnk(cf.predWord, wordCounts, lexicalCountCutoff) + "-" + bucket(sentIdx))
//        maybeAdd(feats, addToIndexer, "PredPOS=" + cf.predPos)
//        maybeAdd(feats, addToIndexer, "Arg=" + wordOrUnk(cf.argWord, wordCounts, lexicalCountCutoff))
//        maybeAdd(feats, addToIndexer, "ArgPOS=" + cf.argPos)
//        maybeAdd(feats, addToIndexer, "Label=" + cf.label)
//        maybeAdd(feats, addToIndexer, "Triple=" + cf.predPos + "-" + cf.label + "-" + cf.argPos)
////        maybeAdd(feats, addToIndexer, "PredArg=" + wordOrUnk(cf.predWord) + "-" + wordOrUnk(cf.argWord))
//      }
//    }
//    if (featSpec.contains("posnlexical")) {
//      for (cf <- caseframesThisSent) {
//        maybeAdd(feats, addToIndexer, "SentPosnPred=" + wordOrUnk(cf.predWord, wordCounts, lexicalCountCutoff) + "-" + bucket(sentIdx))
//        maybeAdd(feats, addToIndexer, "SentPosnPredPOS=" + cf.predPos + "-" + bucket(sentIdx))
//        maybeAdd(feats, addToIndexer, "SentPosnArg=" + wordOrUnk(cf.argWord, wordCounts, lexicalCountCutoff) + "-" + bucket(sentIdx))
//        maybeAdd(feats, addToIndexer, "SentPosnArgPOS=" + cf.argPos + "-" + bucket(sentIdx))
//        maybeAdd(feats, addToIndexer, "SentPosnLabel=" + cf.label + "-" + bucket(sentIdx))
//        maybeAdd(feats, addToIndexer, "SentPosnTriple=" + cf.predPos + "-" + cf.label + "-" + cf.argPos + "-" + bucket(sentIdx))
////        maybeAdd(feats, addToIndexer, "PredArg=" + wordOrUnk(cf.predWord) + "-" + wordOrUnk(cf.argWord))
//      }
//    }
//    if (featSpec.contains("context")) {
//      // Seem to have no effect on small data
//      for (anchoredCf <- anchoredCaseframesThisSent) {
//        maybeAdd(feats, addToIndexer, "PredPW=" + wordOrUnk(fetchWordOrNull(docSentWords, anchoredCf.predIdx - 1), wordCounts, lexicalCountCutoff))
//        maybeAdd(feats, addToIndexer, "PredNW=" + wordOrUnk(fetchWordOrNull(docSentWords, anchoredCf.predIdx + 1), wordCounts, lexicalCountCutoff))
//        maybeAdd(feats, addToIndexer, "PredPP=" + wordOrUnk(fetchWordOrNull(docSentPoss, anchoredCf.predIdx - 1), wordCounts, lexicalCountCutoff))
//        maybeAdd(feats, addToIndexer, "PredNP=" + wordOrUnk(fetchWordOrNull(docSentPoss, anchoredCf.predIdx + 1), wordCounts, lexicalCountCutoff))
//      }
//    }
//    if (featSpec.contains("basicxmlcats")) {
//      val cats = ex.rawDoc.getCategories
//      for (cat <- cats) {
//        maybeAdd(feats, featIdx, addToIndexer, "CatSentPosn=" + cat + "-" + bucket(sentIdx))
//        for (i <- 0 until docSentWords.size) {
//          val word = docSentWords(i)
//          val pos = docSentWords(i)
//          maybeAdd(feats, addToIndexer, "CatWord=" + cat + "-" + wordOrUnk(word, wordCounts, lexicalCountCutoff))
//          maybeAdd(feats, addToIndexer, "CatPos=" + cat + "-" + pos)
//        }
//      }
//    }
//    if (featSpec.contains("xmlcats")) {
//      val cats = ex.rawDoc.categories
//      for (cat <- cats) {
//        for (cf <- caseframesThisSent) {
//          maybeAdd(feats, addToIndexer, "CatPredArg=" + cat + "-" + wordOrUnk(cf.predWord) + "-" + wordOrUnk(cf.argWord))
//          maybeAdd(feats, addToIndexer, "CatPred=" + cat + "-" + wordOrUnk(cf.predWord))
//          maybeAdd(feats, addToIndexer, "CatArg=" + cat + "-" + wordOrUnk(cf.argWord))
//        }
//      }
//    }
    feats
  }
  
  // Make additional use of NER, coref, parser, etc.
  def extractFancyFeatures(featSpec: Set[String], wordCounts: Counter[String], lexicalCountCutoff: Int, rawDoc: SummDoc, sentIdx: Int, startIdx: Int, endIdx: Int, addToIndexer: Boolean): Array[Int] = {
    val feats = new ArrayBuffer[Int]
    if (featSpec.contains("parse")) {
      val tree = rawDoc.corefDoc.rawDoc.trees(sentIdx)
      for (subtree <- tree.constTree.getPostOrderTraversal().asScala) {
        if (!subtree.isPreTerminal() && !subtree.isLeaf) {
          val rule = subtree.getLabel() + "->" + subtree.getChildren().asScala.map(_.getLabel).reduce(_ + "-" + _)
          maybeAdd(feats, addToIndexer, "RuleApp=" + rule)
        }
      }
    }
    if (featSpec.contains("coref")) {
      val entitiesRepresented = rawDoc.entitiesPerSent(sentIdx)
      val entitiesBySize = rawDoc.entitiesBySize
      for (entityIdx <- entitiesRepresented) {
        val sizeIdx = entitiesBySize.indexOf(entityIdx)
        val entityType = rawDoc.entitySemanticTypes(entityIdx)
        if (sizeIdx < 10) {
//          maybeAdd(feats, addToIndexer, "EntitySize=" + bucket(sizeIdx))
          maybeAdd(feats, addToIndexer, "EntitySizePosn=" + bucket(sizeIdx) + "-" + bucket(sentIdx))
          maybeAdd(feats, addToIndexer, "EntitySizeTypePosn=" + bucket(sizeIdx) + "-" + entityType + "-" + bucket(sentIdx))
        }
      }
      for (ment <- rawDoc.getMentionsInSpan(sentIdx, startIdx, endIdx)) {
        // Shape features on the mention
        maybeAdd(feats, addToIndexer, "MentTypeLenPosn=" + ment.mentionType + "-" + bucket(ment.endIdx - ment.startIdx) + "-" + bucket(sentIdx))
      }
      maybeAdd(feats, addToIndexer, "NumEntitiesPosn=" + bucket(entitiesRepresented.size) + "-" + bucket(sentIdx))
    }
    if (featSpec.contains("ner")) {
      val nerTypesRepresented = rawDoc.getSentMents(sentIdx).map(_.nerString).filter(_ != "O")
//      ex.rawDoc.getSentMents(sentIdx).foreach(ment => {
//        Logger.logss(ment.nerString + ": " + ex.rawDoc.corefDoc.rawDoc.words(sentIdx).slice(ment.startIdx, ment.endIdx).reduce(_ + " " + _)) 
//      })
//      Logger.logss(nerTypesRepresented + ": " + ex.rawDoc.doc(sentIdx).getWords.reduce(_ + " " + _))
      for (nerType <- nerTypesRepresented) {
        maybeAdd(feats, addToIndexer, "NerTypePosn=" + nerType + "-" + bucket(sentIdx))
      }
    }
    feats.toArray
  }
  
  def getOverlappingEntities(sortedSeq1: Seq[Int], sortedSeq2: Seq[Int]) = {
    val results = new ArrayBuffer[Int]
    var idx1 = 0
    while (idx1 < sortedSeq1.size) {
      var idx2 = 0
      while (idx2 < sortedSeq2.size && sortedSeq1(idx1) >= sortedSeq2(idx2)) {
        if (sortedSeq1(idx1) == sortedSeq2(idx2)) {
          results += sortedSeq1(idx1)
        }
        idx2 += 1
      }
      idx1 += 1
    }
    results
  }
  
  def isContentToken[D<:DepParseDoc](pos: String) = {
    !StopwordDict.stopwordTags.contains(pos)
  }
  
  def fetchWordOrNull(words: Seq[String], idx: Int) = {
    if (idx < 0) "<s>" else if (idx >= words.size) "</s>" else words(idx)
  }
  
  def wordOrUnk(word: String, wordCounts: Counter[String], lexicalCountCutoff: Int) = {
    val featStr = if (wordCounts.getCount(word) >= lexicalCountCutoff || word == "<s>" || word == "</s>") word else "UNK"
    featStr
  }
}

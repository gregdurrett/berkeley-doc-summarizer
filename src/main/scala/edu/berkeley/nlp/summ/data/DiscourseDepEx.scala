package edu.berkeley.nlp.summ.data

import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.entity.coref.CorefDoc
import edu.berkeley.nlp.entity.coref.MentionType
import edu.berkeley.nlp.entity.coref.PairwiseScorer
import edu.berkeley.nlp.futile.classify.ClassifyUtils
import edu.berkeley.nlp.futile.math.SloppyMath
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.summ.RougeComputer
import edu.berkeley.nlp.summ.CorefUtils
import edu.berkeley.nlp.summ.compression.SyntacticCompressor
import edu.berkeley.nlp.summ.preprocess.EDUSegmenterMain
import edu.berkeley.nlp.summ.preprocess.DiscourseDependencyParser
import edu.berkeley.nlp.summ.preprocess.EDUSegmenter

class DiscourseDepEx(val rawDoc: SummDoc,
                     private val discourseTree: DiscourseTree,
                     val eduAlignments: Seq[((Int,Int),(Int,Int))]) {
  
  val conllDoc = rawDoc.corefDoc.rawDoc
  
//  private val parentsMultiRoots = discourseTree.parents
//  private val rootIndices = parentsMultiRoots.zipWithIndex.filter(_._1 == -1).map(_._2)
//  // Link up all subsequent roots to the first root
//  private val parents = Array.tabulate(parentsMultiRoots.size)(i => if (rootIndices.contains(i) && rootIndices(0) != i) rootIndices(0) else parentsMultiRoots(i))
  
  val nontrivialSegmentBoundaries = eduAlignments.map(_._1).filter(_._2 != 0)
  val goldSegmentBoundaries = Array.tabulate(conllDoc.numSents)(sentIdx =>
    Array.tabulate(conllDoc.words(sentIdx).size - 1)(boundIdx => nontrivialSegmentBoundaries.contains(sentIdx -> (boundIdx + 1))))
  // For each sentence, what are the gold spans?
  val goldEduSpans: Seq[Seq[(Int,Int)]] = (0 until conllDoc.numSents).map(sentIdx => eduAlignments.filter(_._1._1 == sentIdx).map(edu => edu._1._2 -> edu._2._2))
  
  var cachedFeatures: Array[Array[Array[Array[Int]]]] = null;
  var cachedEduFeatures: Array[Array[Array[Int]]] = null;
  var cachedEduSemiMarkovFeatures: Array[Array[Array[Array[Int]]]] = null;
  
  def size = eduAlignments.size
  
  def discourseTreeFileName = discourseTree.name
  
  def getLeaves = discourseTree.leaves
  
  def getParents(useMultinuclearLinks: Boolean, parentType: String = "normal") = {
    val parents = discourseTree.getParents(useMultinuclearLinks)
    if (parentType == "intra") {
      // Only keep parent-child structures within a sentence, otherwise let there be many roots
      Array.tabulate(parents.size)(i => if (parents(i) == -1 || eduAlignments(parents(i))._1._1 == eduAlignments(i)._1._1) parents(i) else -1)
    } else if (parentType == "flat") {
      Array.tabulate(parents.size)(i => -1)
    } else {
      parents
    }
  }
  
  def getLabels(useMultinuclearLinks: Boolean) = {
    discourseTree.getLabels(useMultinuclearLinks)
  }
  
  def getParentsMultiRoots = discourseTree.parentsMultiRoot
  
  def getLeafWords(idx: Int) = DiscourseDepEx.getLeafWords(rawDoc, eduAlignments, idx)
  def getLeafPoss(idx: Int) = DiscourseDepEx.getLeafPoss(rawDoc, eduAlignments, idx)
  
  def stripGold = new DiscourseDepExNoGold(rawDoc, eduAlignments)
}

object DiscourseDepEx {
  
  def getLeafWords(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], idx: Int): Seq[String] = {
    val eduSpan = eduAlignments(idx)
    if (eduSpan._1._1 == eduSpan._2._1) {
      rawDoc.doc(eduSpan._1._1).getWords.slice(eduSpan._1._2, eduSpan._2._2)
    } else {
      val allWords = new ArrayBuffer[String]
      for (sentIdx <- eduSpan._1._1 to eduSpan._2._1) {
        val sentWords = rawDoc.doc(sentIdx).getWords
        allWords ++= sentWords.slice(if (sentIdx == eduSpan._1._1) eduSpan._1._2 else 0, if (sentIdx == eduSpan._2._1) eduSpan._2._2 else sentWords.size) 
      }
      allWords
    }
  }
  
  def getLeafPoss(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], idx: Int): Seq[String] = {
    val eduSpan = eduAlignments(idx)
    if (eduSpan._1._1 == eduSpan._2._1) {
      rawDoc.doc(eduSpan._1._1).getPoss.slice(eduSpan._1._2, eduSpan._2._2)
    } else {
      val allPoss = new ArrayBuffer[String]
      for (sentIdx <- eduSpan._1._1 to eduSpan._2._1) {
        val sentWords = rawDoc.doc(sentIdx).getPoss
        allPoss ++= sentWords.slice(if (sentIdx == eduSpan._1._1) eduSpan._1._2 else 0, if (sentIdx == eduSpan._2._1) eduSpan._2._2 else sentWords.size) 
      }
      allPoss
    }
  }
}


class DiscourseDepExNoGold(val rawDoc: SummDoc,
                           val eduAlignments: Seq[((Int,Int),(Int,Int))]) extends Serializable {
  
  def size = eduAlignments.size
  def getLeafWords(idx: Int) = DiscourseDepEx.getLeafWords(rawDoc, eduAlignments, idx)
  def getLeafPoss(idx: Int) = DiscourseDepEx.getLeafPoss(rawDoc, eduAlignments, idx)
}


@SerialVersionUID(1L)
case class PronounReplacement(val mentIdx: Int,
                              val antIdx: Int,
                              val eduIdx: Int,
                              val replacementWords: Seq[String],
                              val replacementPoss: Seq[String],
                              val addedGenitive: Boolean,
                              val removedGenitive: Boolean,
                              val prevEDUsContainingEntity: Seq[Int]) extends Serializable {
  
  def render(corefDoc: CorefDoc): String = {
    "[" + CorefUtils.getMentionText(corefDoc.goldMentions(mentIdx)).reduce(_ + " " + _) + " -->" + replacementWords.foldLeft("")(_ + " " + _) + (if (addedGenitive || removedGenitive) " *GENITIVE*" else "") + "]"
  }
  
  def computeCorefClusterLogPosterior(corefDoc: CorefDoc, corefPredictor: PairwiseScorer) = {
    val posteriors = CorefUtils.computePosteriors(corefDoc, corefPredictor, Seq(mentIdx)).head
    val antecedents = corefDoc.goldClustering.getCluster(mentIdx).filter(_ < mentIdx)
    antecedents.map(posteriors(_)).foldLeft(Double.NegativeInfinity)(SloppyMath.logAdd(_, _))
  }
}

@SerialVersionUID(1L)
case class FragilePronoun(val mentIdx: Int,
                          val eduIdx: Int,
                          val antecedentEdus: Seq[Int],
                          val antecedentMentIndices: Option[Seq[Int]]) extends Serializable {
  
}

object FragilePronoun {
  def computeCorefClusterLogPosteriorAllMents(corefDoc: CorefDoc, corefPredictor: PairwiseScorer, mentIdx: Int) = {
    CorefUtils.computePosteriors(corefDoc, corefPredictor, Seq(mentIdx)).head
  }
}

@SerialVersionUID(-4341752039191008531L)
class DiscourseDepExProcessed(val rawDoc: SummDoc,
                              val eduAlignments: Seq[((Int,Int),(Int,Int))],
                              val parents: Seq[Int],
                              val parentLabels: Seq[String]) extends DepParseDoc with Serializable {
  
  
  val cachedWordCounts = new Counter[String]
  rawDoc.doc.map(_.getWords).flatten.foreach(word => cachedWordCounts.incrementCount(word, 1.0))
  
  val leafSizes = eduAlignments.map(indices => indices._2._2 - indices._1._2)
  private val summBigrams = rawDoc.summary.flatMap(sent => RougeComputer.getBigramsNoStopwords(sent.getWords, sent.getPoss)).toSet
//  private val bigramRecallScores = eduAlignments.map(edu => {
//    val words = rawDoc.doc(edu._1._1).getWords.slice(edu._1._2, edu._2._2)
//    val poss = rawDoc.doc(edu._1._1).getPoss.slice(edu._1._2, edu._2._2)
//    (RougeComputer.getBigramsNoStopwords(words, poss).toSet & summBigrams).size.toDouble
//  })
  var cachedSummFeats: Array[Array[Int]] = null;
  var cachedPronFeats: Array[Array[Int]] = null;
  var cachedPronounReplacements: Seq[PronounReplacement] = null
  var cachedFragilePronouns: Seq[FragilePronoun] = null
  var cachedBigramFeats: Array[Array[Int]] = null;
  
  def name: String = rawDoc.name
  def doc: Seq[DepParse] = rawDoc.doc
  def summary: Seq[DepParse] = rawDoc.summary
  
  def getSummBigrams(useUnigrams: Boolean = false): Set[(String,String)] = {
    if (useUnigrams) {
      rawDoc.summary.flatMap(sent => RougeComputer.getUnigramsNoStopwords(sent.getWords, sent.getPoss)).toSet
    } else {
      summBigrams
    }
  }
  
  def getBigramRecallScores(useUnigrams: Boolean = false) = (0 until eduAlignments.size).map(i => getBigramRecallScore(i, useUnigrams))
  
  def getBigramRecallScore(eduIdx: Int, useUnigrams: Boolean = false) = {
    val edu = eduAlignments(eduIdx)
    val summBigrams = getSummBigrams(useUnigrams)
    val words = rawDoc.doc(edu._1._1).getWords.slice(edu._1._2, edu._2._2)
    val poss = rawDoc.doc(edu._1._1).getPoss.slice(edu._1._2, edu._2._2)
    if (useUnigrams) {
      (RougeComputer.getUnigramsNoStopwords(words, poss).toSet & summBigrams).size.toDouble
    } else {
      (RougeComputer.getBigramsNoStopwords(words, poss).toSet & summBigrams).size.toDouble
    }
  }
  
  def getLen(edus: Seq[Int], prons: Seq[Int]) = {
    edus.map(getEduWords(_).size).foldLeft(0)(_ + _) - prons.map(cachedPronounReplacements(_).replacementWords.size - 1).foldLeft(0)(_ + _) 
  }
  
  def getEduWords(eduIdx: Int) = {
    val edu = eduAlignments(eduIdx)
    rawDoc.doc(edu._1._1).getWords.slice(edu._1._2, edu._2._2)
  }
  
  def getEduPoss(eduIdx: Int) = {
    val edu = eduAlignments(eduIdx)
    rawDoc.doc(edu._1._1).getPoss.slice(edu._1._2, edu._2._2)
  }
  
  
  private def getDocUnigrams = (0 until eduAlignments.size).map(i => RougeComputer.getUnigramsNoStopwords(getEduWords(i), getEduPoss(i)).toSet).reduce(_ ++ _)
  private def getDocBigrams = (0 until eduAlignments.size).map(i => RougeComputer.getBigramsNoStopwords(getEduWords(i), getEduPoss(i)).toSet).reduce(_ ++ _)
  
  def getDocBigramCounts(useUnigrams: Boolean) = {
    val counter = new Counter[(String,String)]
    for (i <- 0 until eduAlignments.size) {
      (if (useUnigrams) RougeComputer.getUnigramsNoStopwords(getEduWords(i), getEduPoss(i)) else RougeComputer.getBigramsNoStopwords(getEduWords(i), getEduPoss(i))).foreach(counter.incrementCount(_, 1.0))
    }
    counter
  }
  def getDocBigramsSeq(useUnigrams: Boolean) = if (useUnigrams) getDocUnigrams.toSeq.sorted else getDocBigrams.toSeq.sorted
  
  def getParents(parentType: String = "normal"): Seq[Int] = {
    if (parentType == "intra") {
      // Only keep parent-child structures within a sentence, otherwise let there be many roots
      Array.tabulate(parents.size)(i => if (parents(i) == -1 || eduAlignments(parents(i))._1._1 == eduAlignments(i)._1._1) parents(i) else -1)
    } else if (parentType == "flat") {
      Array.tabulate(parents.size)(i => -1)
    } else if (parentType == "firstk") {
      Array.tabulate(parents.size)(i => (i-1))
    } else {
      parents
    }
  }
  
  def getFirstKWords(k: Int, oneSentPerLine: Boolean = true) = {
    val lines = new ArrayBuffer[String]
    var numWordsUsed = 0
    var currEduIdx = 0
    while (numWordsUsed < k && currEduIdx < eduAlignments.size) {
      val sentIdx = eduAlignments(currEduIdx)._1._1
      val currEduWords = getEduWords(currEduIdx)
      val eduWordsToUse = currEduWords.slice(0, Math.min(currEduWords.size, k - numWordsUsed))
      if (currEduIdx == 0 || sentIdx != eduAlignments(currEduIdx - 1)._1._1) {
        lines += eduWordsToUse.reduce(_ + " " + _)
      } else {
        lines(lines.size - 1) += " " + eduWordsToUse.reduce(_ + " " + _)
      }
      numWordsUsed += eduWordsToUse.size
      currEduIdx += 1
    }
    lines(lines.size - 1) = punctuateSentence(lines(lines.size - 1))
    lines
  }
  
  def getSummaryText(edus: Seq[Int], oneSentPerLine: Boolean = true) = {
    getSummaryTextWithPronounsReplaced(edus, Seq[PronounReplacement](), oneSentPerLine)
  }
  
  def getSummaryTextWithPronounsReplaced(edus: Seq[Int], pronReplacements: Seq[PronounReplacement], oneSentPerLine: Boolean): Seq[String] = {
//    val clustersSeen = new HashSet[Int]
    val text = new ArrayBuffer[String]
    
    var lastEduAlignment = -1
    for (eduIdx <- edus) {
      val sentIdx = eduAlignments(eduIdx)._1._1
      val eduStartIdx = eduAlignments(eduIdx)._1._2
      val eduEndIdx = eduAlignments(eduIdx)._2._2
      val sent = rawDoc.doc(sentIdx)
      val eduWords = new ArrayBuffer[String] ++ sent.getWords.slice(eduStartIdx, eduEndIdx)
      val replacementsThisSent = pronReplacements.filter(_.eduIdx == eduIdx)
      for (replacement <- replacementsThisSent) {
        val ment = rawDoc.corefDoc.goldMentions(replacement.mentIdx)
        eduWords(ment.startIdx - eduStartIdx) = replacement.replacementWords.foldLeft("")(_ + " " + _).trim
      }
      // Aggregate results within the same sentence then add . at the end if necessary
      if (oneSentPerLine) {
        if (sentIdx != lastEduAlignment) {
          if (!text.isEmpty) {
            text(text.size - 1) = capitalizeSentenceFixThat(punctuateSentence(text(text.size - 1)))
          }
          lastEduAlignment = sentIdx
          text += ""
        }
        text(text.size - 1) = text(text.size - 1) + (if (text.last.isEmpty) "" else " ") + eduWords.reduce(_ + " " + _)
      } else {
        text += eduWords.reduce(_ + " " + _)
      }
    }
    if (oneSentPerLine && !text.isEmpty) {
      text(text.size - 1) = capitalizeSentenceFixThat(punctuateSentence(text(text.size - 1)))
    }
    text
  }
  
  def getSummaryTextWithPronounsReplacedDemo(edus: Seq[Int], pronReplacements: Seq[PronounReplacement]): (Seq[String], String, Seq[(Int,Int)], Seq[(Int,Int)]) = {
    val text = getSummaryTextWithPronounsReplaced(edus, pronReplacements, oneSentPerLine = true)
    var fullText = ""
    val compressionSpans = new ArrayBuffer[(Int,Int)]
    val pronRepSpans = new ArrayBuffer[(Int,Int)]
    var lastEduAlignment = -1
    val sentIndices = edus.map(eduAlignments(_)._1._1).distinct
    for (sentIdx <- sentIndices) {
      val sentEduStartIdx = eduAlignments.filter(_._1._1 < sentIdx).size
      val sentEduEndIdx = eduAlignments.filter(_._1._1 <= sentIdx).size
      for (eduIdx <- sentEduStartIdx until sentEduEndIdx) {
        val eduStartIdx = eduAlignments(eduIdx)._1._2
        val eduEndIdx = eduAlignments(eduIdx)._2._2
        val sent = rawDoc.doc(sentIdx)
        val eduWords = new ArrayBuffer[String] ++ sent.getWords.slice(eduStartIdx, eduEndIdx)
        if (edus.contains(eduIdx)) {
          val replacementsThisSent = pronReplacements.filter(_.eduIdx == eduIdx)
          for (replacement <- replacementsThisSent) {
            val ment = rawDoc.corefDoc.goldMentions(replacement.mentIdx)
//            val repStr = replacement.replacementWords.foldLeft("")(_ + " " + _).trim
//            eduWords(ment.startIdx - eduStartIdx) = repStr
            val pronRepStart = fullText.size + eduWords.slice(0, ment.startIdx - eduStartIdx).foldLeft("")( _ + " " + _).trim.size
            pronRepSpans += pronRepStart -> (pronRepStart + eduWords(ment.startIdx - eduStartIdx).size)
          }
        } else {
          compressionSpans += fullText.size -> (fullText.size + (eduWords.foldLeft("")(_ + " " + _).trim).size)
        }
        fullText += eduWords.foldLeft("")(_ + " " + _).trim + " "
      }
    }
    (text, fullText, compressionSpans, pronRepSpans)
  }
  
  private def capitalizeSentenceFixThat(text: String) = {
    // Essentially always indicates a relative clause and should be deleted
    val fixedText = if (text.startsWith("that")) {
      text.drop(4).trim
    } else {
      text
    }
    if (fixedText.size > 0 && Character.isLetter(fixedText.charAt(0)) && !Character.isUpperCase(fixedText.charAt(0))) {
      Character.toUpperCase(fixedText.charAt(0)) + fixedText.substring(1)
    } else {
      fixedText
    }
  }
  
  private def punctuateSentence(origText: String) = {
    // Replace duplicated commas or commas before periods
    var text = origText.replace(", ,", ",").replace(", .", ".")
    if (text.trim.startsWith(",")) {
      text = text.drop(1).trim
    }
    if (text.trim.endsWith(",") || text.trim.endsWith(";") || text.trim.endsWith(":")) {
      text.dropRight(1) + "."
    } else if (!(text.endsWith(".") || text.endsWith("!") || text.endsWith("?") || text.endsWith("''"))) {
      text + " ."
    } else {
      text
    }
  }
  
  def identifyPronounReplacements(replaceWithNEOnly: Boolean, corefPredictor: Option[PairwiseScorer], corefConfidenceThreshold: Double): Seq[PronounReplacement] = {
    if (cachedPronounReplacements == null) {
      this.cachedPronounReplacements = identifyPronounReplacementsNoCache(replaceWithNEOnly, corefPredictor, corefConfidenceThreshold)
    }
    this.cachedPronounReplacements
  }
  
  // Maps from a mention index to the words that the mention would be replaced with
  def identifyPronounReplacementsNoCache(replaceWithNEOnly: Boolean, corefPredictor: Option[PairwiseScorer], corefConfidenceThreshold: Double): Seq[PronounReplacement] = {
    val replacements = new ArrayBuffer[PronounReplacement]
    for (eduIdx <- 0 until eduAlignments.size) {
      val sentIdx = eduAlignments(eduIdx)._1._1
      val mentsThisSentence = rawDoc.getMentionsInSpan(sentIdx, eduAlignments(eduIdx)._1._2, eduAlignments(eduIdx)._2._2)
      // Identify mentions which represent clusters which haven't been seen before
      for (ment <- mentsThisSentence) {
        if (ment.mentionType.isClosedClass()) {
          val clusterIdx = rawDoc.corefDoc.goldClustering.getClusterIdx(ment.mentIdx)
          val cluster = rawDoc.corefDoc.goldClustering.getCluster(ment.mentIdx).sorted
          require(cluster.contains(ment.mentIdx))
          val prevMents = cluster.slice(0, cluster.indexOf(ment.mentIdx)).map(mentIdx => rawDoc.corefDoc.goldMentions(mentIdx))
          val prevEdusContainingEntity = prevMents.map(prevMent => {
            val edus = eduAlignments.filter(alignment => alignment._1._1 == prevMent.sentIdx && alignment._1._2 <= prevMent.headIdx && prevMent.headIdx < alignment._2._2)
            require(edus.size == 1)
            eduAlignments.indexOf(edus.head)
          }).toSet
          if (prevEdusContainingEntity.contains(eduIdx)) {
            // Do nothing because there's a previous mention of this entity in the same sentence
          } else {
            // TODO: Maybe choose something else (e.g. a PROPER if the first mention is NOMINAL, etc.)
            val pron = rawDoc.corefDoc.rawDoc.words(ment.sentIdx)(ment.headIdx)
            val pronTag = rawDoc.corefDoc.rawDoc.pos(ment.sentIdx)(ment.headIdx)
            val isPronGenitive = pronTag == "PRP$"
            val antIdx = cluster.head
            val firstMention = rawDoc.corefDoc.goldMentions(antIdx)
            if (firstMention.mentionType == MentionType.PROPER || firstMention.mentionType == MentionType.NOMINAL) {
              val (startIdx, endIdx) = if (replaceWithNEOnly) {
                val nerSpan = CorefUtils.getMentionNerSpan(firstMention)
                if (nerSpan.isDefined) nerSpan.get else firstMention.startIdx -> firstMention.endIdx
              } else {
                firstMention.startIdx -> firstMention.endIdx
              }
              val words = new ArrayBuffer[String] ++ rawDoc.corefDoc.rawDoc.words(firstMention.sentIdx).slice(startIdx, endIdx)
              val poss = new ArrayBuffer[String] ++ rawDoc.corefDoc.rawDoc.pos(firstMention.sentIdx).slice(startIdx, endIdx)
              // Munge things around if the replacement text should be genitive and isn't or vice versa
              val isMentGenitive = words.last.endsWith("'s")
              var addedGenitive = false
              var removedGenitive = false
              if (isPronGenitive && !isMentGenitive) {
                words(words.size - 1) += "'s"
                addedGenitive = true
              }
              if (!isPronGenitive && isMentGenitive) {
                if (words(words.size - 1) == "'s") {
                  words.remove(words.size - 1)
                  poss.remove(poss.size - 1)
                } else {
                  words(words.size - 1).replace("'s", "")
                }
                removedGenitive = true
              }
              val pronRep = new PronounReplacement(ment.mentIdx, antIdx, eduIdx, words, poss, addedGenitive, removedGenitive, prevEdusContainingEntity.toSeq.sorted)
              val isCorefConfident = !corefPredictor.isDefined || Math.exp(pronRep.computeCorefClusterLogPosterior(rawDoc.corefDoc, corefPredictor.get)) > corefConfidenceThreshold
//              if (!isPronTypeRight) {
//                Logger.logss("Prohibiting: " + pron + " - " + words.toSeq + ", type = " + firstMention.nerString)
//              } else {
//                Logger.logss("Allowing: " + pron + " - " + words.toSeq + ", type = " + firstMention.nerString)
//              }
              if (isCorefConfident) { 
                replacements += pronRep
              }
            }
          }
        }
      }
    }
    replacements
  }
  
  def identifyFragilePronouns(corefPredictor: Option[PairwiseScorer]): Seq[FragilePronoun] = {
    if (cachedFragilePronouns == null) {
      this.cachedFragilePronouns = identifyFragilePronounsNoCache(corefPredictor)
    }
    this.cachedFragilePronouns
  }
  
  def identifyFragilePronounsNoCache(corefPredictor: Option[PairwiseScorer]): Seq[FragilePronoun] = {
    val fragiles = new ArrayBuffer[FragilePronoun]
    for (eduIdx <- 0 until eduAlignments.size) {
      val sentIdx = eduAlignments(eduIdx)._1._1
      val mentsThisSentence = rawDoc.getMentionsInSpan(sentIdx, eduAlignments(eduIdx)._1._2, eduAlignments(eduIdx)._2._2)
      // Identify mentions that are determined to have antecedents by the algorithm (things that are probably singletons
      // aren't interesting to us)
      for (ment <- mentsThisSentence) {
        if (ment.mentionType.isClosedClass()) {
          val clusterIdx = rawDoc.corefDoc.goldClustering.getClusterIdx(ment.mentIdx)
          val cluster = rawDoc.corefDoc.goldClustering.getCluster(ment.mentIdx).sorted
          require(cluster.contains(ment.mentIdx))
          if (cluster.head != ment.mentIdx) {
            val posterior = FragilePronoun.computeCorefClusterLogPosteriorAllMents(rawDoc.corefDoc, corefPredictor.get, ment.mentIdx)
            // Identify the two most likely antecedents
            val bestAntIdx = ClassifyUtils.argMaxIdx(posterior)
            // If the best option isn't very confident, include a second one
            var secondBestAntIdx = -1
            for (i <- 0 until posterior.size) {
              if (i != bestAntIdx && (secondBestAntIdx == -1 || posterior(i) > posterior(secondBestAntIdx))) {
                secondBestAntIdx = i
              }
            }
            val allAnts = if (Math.exp(posterior(bestAntIdx)) < 0.6) Seq(bestAntIdx, secondBestAntIdx).sorted else Seq(bestAntIdx)
            val antEdusRaw = allAnts.map(antMentIdx => {
              val antMent = rawDoc.corefDoc.goldMentions(antMentIdx)
              val edus = eduAlignments.filter(alignment => alignment._1._1 == antMent.sentIdx &&
                                              alignment._1._2 <= antMent.headIdx && antMent.headIdx < alignment._2._2)
              require(edus.size == 1)
              eduAlignments.indexOf(edus.head)
            }).sorted.distinct
            // Can't put meaningful constraints on the same EDU
            val antEdus = antEdusRaw.filter(_ != eduIdx)
            if (!antEdus.isEmpty) {
              fragiles += FragilePronoun(ment.mentIdx, eduIdx, antEdus, Some(allAnts))
            }
          }
        }
      }
    }
    fragiles
  }
  
  def getBigramDelta(pronReplacement: PronounReplacement, useUnigrams: Boolean): (Set[(String,String)],Set[(String,String)]) = {
    // Find the bigram implied by the edu before
    val mention = rawDoc.corefDoc.goldMentions(pronReplacement.mentIdx)
    // If the mention neither starts nor ends the EDU,
    val edu = eduAlignments(pronReplacement.eduIdx)
    val sent = rawDoc.doc(edu._1._1)
    val oldWords = sent.getWords.slice(edu._1._2, edu._2._2)
    val oldPoss = sent.getPoss.slice(edu._1._2, edu._2._2)
    val newWords = sent.getWords.slice(edu._1._2, mention.startIdx) ++ pronReplacement.replacementWords ++ sent.getWords.slice(mention.endIdx, edu._2._2)
    val newPoss = sent.getPoss.slice(edu._1._2, mention.startIdx) ++ pronReplacement.replacementPoss ++ sent.getPoss.slice(mention.endIdx, edu._2._2)
    val newBigrams = if (useUnigrams) RougeComputer.getUnigramsNoStopwords(newWords, newPoss).toSet else RougeComputer.getBigramsNoStopwords(newWords, newPoss).toSet
    val oldBigrams = if (useUnigrams) RougeComputer.getUnigramsNoStopwords(oldWords, oldPoss).toSet else RougeComputer.getBigramsNoStopwords(oldWords, oldPoss).toSet
    (newBigrams -- oldBigrams, oldBigrams -- newBigrams)
  }
//  
  def getBigramRecallDelta(pronReplacement: PronounReplacement, useUnigrams: Boolean): Double = {
    // Find the bigram implied by the edu before
    val mention = rawDoc.corefDoc.goldMentions(pronReplacement.mentIdx)
    // If the mention neither starts nor ends the EDU,
    val edu = eduAlignments(pronReplacement.eduIdx)
    val sent = rawDoc.doc(edu._1._1)
    val newWords = sent.getWords.slice(edu._1._2, mention.startIdx) ++ pronReplacement.replacementWords ++ sent.getWords.slice(mention.endIdx, edu._2._2)
    val newPoss = sent.getPoss.slice(edu._1._2, mention.startIdx) ++ pronReplacement.replacementPoss ++ sent.getPoss.slice(mention.endIdx, edu._2._2)
    val newBigrams = if (useUnigrams) RougeComputer.getUnigramsNoStopwords(newWords, newPoss).toSet else RougeComputer.getBigramsNoStopwords(newWords, newPoss).toSet
    val summBigrams = getSummBigrams(useUnigrams)
    require(newBigrams != null)
    require(summBigrams != null)
    (newBigrams & summBigrams).size - getBigramRecallScore(pronReplacement.eduIdx, useUnigrams)
  }
  
  def renderWithinSentencesForOutput = {
    val (synEdus, synParents, synLabels) = SyntacticCompressor.compress(rawDoc)
    val lines = new ArrayBuffer[String]
    for (sentIdx <- 0 until rawDoc.corefDoc.rawDoc.numSents) {
      val sentEdus = eduAlignments.filter(_._1._1 == sentIdx)
      val sentWords = rawDoc.corefDoc.rawDoc.words(sentIdx)
      var lineStr = ""
      for (edu <- sentEdus) {
        lineStr += "[" + sentWords.slice(edu._1._2, edu._2._2).foldLeft("")(_ + " " + _).trim + "] "
      }
      val sentEdusStart = eduAlignments.filter(_._1._1 < sentIdx).size
      val sentEdusEnd = eduAlignments.filter(_._1._1 <= sentIdx).size
      lines += lineStr.trim + "; PARENTS = " + parents.slice(sentEdusStart, sentEdusEnd).toSeq.toString + "; LABELS = " + parentLabels.slice(sentEdusStart, sentEdusEnd).toSeq.toString 
    }
    lines
  }
}

object DiscourseDepExProcessed {
  
  def makeTrivial(rawDoc: SummDoc): DiscourseDepExProcessed = {
    val eduAlignments = (0 until rawDoc.doc.size).map(i => (i, 0) -> (i, rawDoc.doc(i).size))
    val parents = (0 until eduAlignments.size).map(i => -1)
    val labels = (0 until eduAlignments.size).map(i => "-")
    apply(rawDoc, eduAlignments, parents, labels)
  }
  
  def makeWithSyntacticCompressions(rawDoc: SummDoc): DiscourseDepExProcessed = {
    val (eduAlignments, parents, labels) = SyntacticCompressor.compress(rawDoc)
    apply(rawDoc, eduAlignments, parents, labels)
  }
  
  def makeWithEduAndSyntactic(rawDoc: SummDoc, segmenter: EDUSegmenter, parser: DiscourseDependencyParser): DiscourseDepExProcessed = {
    val predictions = segmenter.decode(rawDoc.corefDoc.rawDoc)
    val eduAlignments = EDUSegmenterMain.extractSegments(predictions)
    makeWithEduAndSyntactic(rawDoc, eduAlignments, parser)
  }
  
  def makeWithEduAndSyntactic(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], parser: DiscourseDependencyParser): DiscourseDepExProcessed = {
    val (parents, labels) = parser.decode(new DiscourseDepExNoGold(rawDoc, eduAlignments))
    require(parents.size == labels.size && parents.size == eduAlignments.size, eduAlignments.size + " " + parents.size + " " + labels.size)
    makeWithEduAndSyntactic(rawDoc, eduAlignments, parents, labels)
  }
  
  def makeWithEduAndSyntactic(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], parents: Seq[Int], labels: Seq[String]): DiscourseDepExProcessed = {
    require(parents.size == labels.size && parents.size == eduAlignments.size, eduAlignments.size + " " + parents.size + " " + labels.size)
    val (newEduAlignments, newParents, newLabels) = SyntacticCompressor.refineEdus(rawDoc, eduAlignments, parents, labels)
    apply(rawDoc, newEduAlignments, newParents, newLabels)
  }
  
  def apply(rawDoc: SummDoc, segmenter: EDUSegmenter, parser: DiscourseDependencyParser): DiscourseDepExProcessed = {
    val predictions = segmenter.decode(rawDoc.corefDoc.rawDoc)
    val eduAlignments = EDUSegmenterMain.extractSegments(predictions)
    apply(rawDoc, eduAlignments, parser)
  }
  
  def apply(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], parser: DiscourseDependencyParser): DiscourseDepExProcessed = {
    val (parents, labels) = parser.decode(new DiscourseDepExNoGold(rawDoc, eduAlignments))
    apply(rawDoc, eduAlignments, parents, labels)
  }
  
  def apply(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], parents: Seq[Int], labels: Seq[String]): DiscourseDepExProcessed = {
    new DiscourseDepExProcessed(rawDoc, eduAlignments, parents, labels)
  }
}

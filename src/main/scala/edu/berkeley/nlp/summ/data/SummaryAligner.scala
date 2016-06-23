package edu.berkeley.nlp.summ.data

import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.LightRunner
import edu.berkeley.nlp.futile.EditDistance.EditOp
import edu.berkeley.nlp.summ.RougeComputer

object SummaryAligner {

  def alignDocAndSummary(depParseDoc: DepParseDoc, verbose: Boolean): Array[Int] = {
    alignDocAndSummary(depParseDoc.doc.map(_.getWords.toSeq), depParseDoc.summary.map(_.getWords.toSeq), depParseDoc.name, verbose)
  }
  
  def getEditDistanceWithDeletion(docSent: Seq[String], summSent: Seq[String]) = {
    edu.berkeley.nlp.futile.EditDistance.editDistance(docSent.map(_.toLowerCase).asJava, summSent.map(_.toLowerCase).asJava, 1.0, 0.0, 1.0, true)
  }
  
  def getEditDistanceOpsWithDeletion(docSent: Seq[String], summSent: Seq[String]): Array[EditOp] = {
    edu.berkeley.nlp.futile.EditDistance.getEditDistanceOperations(docSent.map(_.toLowerCase).asJava, summSent.map(_.toLowerCase).asJava, 1.0, 0.0, 1.0, true)
  }
  
  /**
   * Produces a one-to-many alignment between the doc and the summary (i.e. each summary
   * sentence is aligned to at most one document sentence). Length is the length of the
   * summary (so summary is the target).
   */
  def alignDocAndSummary(docSentences: Seq[Seq[String]], summary: Seq[Seq[String]], docName: String = "", verbose: Boolean = false): Array[Int] = {
    val alignments = Array.tabulate(summary.size)(i => -1)
    var numSentsAligned = 0
    for (summSentIdx <- 0 until summary.size) {
      var someAlignment = false;
      var bestAlignmentEd = Int.MaxValue
      var bestAlignmentChoice = -1
      for (docSentIdx <- 0 until docSentences.size) {
        val ed = edu.berkeley.nlp.futile.EditDistance.editDistance(docSentences(docSentIdx).asJava, summary(summSentIdx).asJava, 1.0, 0.0, 1.0, true)
        if (ed < bestAlignmentEd) {
          bestAlignmentEd = ed.toInt
          bestAlignmentChoice = docSentIdx
        }
      }
      if (verbose) {
        Logger.logss(summSentIdx + ": best alignment choice = " + bestAlignmentChoice + ", ed = " + bestAlignmentEd)
      }
      if (bestAlignmentEd < summary(summSentIdx).size * 0.5) {
        someAlignment = true
        alignments(summSentIdx) = bestAlignmentChoice
        if (verbose) {
          Logger.logss("ALIGNED: " + summSentIdx + " -> " + bestAlignmentChoice)
          Logger.logss("S1: " + docSentences(bestAlignmentChoice).reduce(_ + " " + _))
          Logger.logss("S2: " + summary(summSentIdx).reduce(_ + " " + _))
          Logger.logss("ED: " + bestAlignmentEd)
        }
      }
      if (!someAlignment) {
//          Logger.logss("UNALIGNED: " + summSentIdx + "  " + summary(summSentIdx).reduce(_ + " " + _));
      } else {
        numSentsAligned += 1
      } 
    }
    if (verbose && numSentsAligned > 0) {
      Logger.logss(">1 alignment for " + docName)
    }
    alignments
  }

  def alignDocAndSummaryOracleRouge(depParseDoc: DepParseDoc, summSizeCutoff: Int): Array[Int] = {
    alignDocAndSummaryOracleRouge(depParseDoc.doc.map(_.getWords.toSeq), depParseDoc.summary.map(_.getWords.toSeq), summSizeCutoff)
  }
  
  def alignDocAndSummaryOracleRouge(docSentences: Seq[Seq[String]], summary: Seq[Seq[String]], summSizeCutoff: Int): Array[Int] = {
    val choices = Array.tabulate(summary.size)(i => {
      if (summary(i).size >= summSizeCutoff) {
        val summSent = summary(i)
        var bestRougeSourceIdx = -1
        var bestRougeScore = 0
        for (j <- 0 until docSentences.size) {
          var score = RougeComputer.computeRouge2SuffStats(Seq(docSentences(j)), Seq(summSent))._1
          if (score > bestRougeScore) {
            bestRougeSourceIdx = j
            bestRougeScore = score
          }
        }
        bestRougeSourceIdx
      } else {
        -1
      }
    })
    choices
  }
  
  def identifySpuriousSummary(firstSentence: Seq[String]) = {
    val firstWords = firstSentence.slice(0, Math.min(10, firstSentence.size)).map(_.toLowerCase)
    val firstWordsNoPlurals = firstWords.map(word => if (word.endsWith("s")) word.dropRight(1) else word)
    firstWordsNoPlurals.contains("letter") || firstWordsNoPlurals.contains("article") || firstWordsNoPlurals.contains("column") ||
      firstWordsNoPlurals.contains("review") || firstWordsNoPlurals.contains("interview") || firstWordsNoPlurals.contains("profile")
  }
  
  def identifySpuriousSentence(sentence: Seq[String]) = {
    val sentenceLcNoPlurals = sentence.map(_.toLowerCase).map(word => if (word.endsWith("s")) word.dropRight(1) else word)
//    sentenceLcNoPlurals.contains("photo")
    sentenceLcNoPlurals.contains("photo") || sentenceLcNoPlurals.contains("photo.")
  }
}
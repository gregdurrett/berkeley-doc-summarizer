package edu.berkeley.nlp.summ

import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed
import edu.berkeley.nlp.summ.data.FragilePronoun
import edu.berkeley.nlp.summ.data.PronounReplacement

/**
 * Baseline summarizer based on Gillick and Favre (2009) that extracts a set of sentences
 * aiming to maximize coverage of high-scoring bigram types in the source document. Specifically,
 * we optimize for
 * 
 * sum_{bigram in summary bigrams} count(bigram)
 * 
 * @author gdurrett
 */
@SerialVersionUID(1L)
class BigramCountSummarizer(useUnigrams: Boolean = false) extends DiscourseDepExSummarizer {
  
  def decodeBigramCounts(ex: DiscourseDepExProcessed, budget: Int): (Seq[Int], Seq[Int], Seq[Int], Seq[Int], Seq[Int], Double) = {
    val allPronReplacements = Seq[PronounReplacement]()
    val leafScores = DiscourseDepExSummarizer.biasTowardsEarlier(Array.fill(ex.eduAlignments.size)(0.0))
//      val pronReplacementScores = Array.tabulate(allPronReplacements.size)(i => ex.getBigramRecallDelta(allPronReplacements(i), useUnigramRouge))
    val pronReplacementScores = Array.fill(allPronReplacements.size)(0.0)
    val bigramSeq = ex.getDocBigramsSeq(useUnigrams)
    val bigramCounts = ex.getDocBigramCounts(useUnigrams)
    val bigramScores = bigramSeq.map(bigram => bigramCounts.getCount(bigram))
    val results = CompressiveAnaphoraSummarizerILP.summarizeILPWithGLPK(ex, ex.getParents("flat"), leafScores, bigramScores, allPronReplacements, pronReplacementScores, budget,
                                                                        0, 0, false, Seq[FragilePronoun](), useUnigrams)
    results
  }
  
  def summarize(ex: DiscourseDepExProcessed, budget: Int, cleanUpForHumans: Boolean = true): Seq[String] = {
    val edus = decodeBigramCounts(ex, budget)._1
    ex.getSummaryTextWithPronounsReplaced(edus, Seq(), cleanUpForHumans)
  }
  
  def summarizeOracle(ex: DiscourseDepExProcessed, budget: Int): Seq[String] = {
    throw new RuntimeException("Unimplemented")
  }
  
  def display(ex: DiscourseDepExProcessed, budget: Int) {
    throw new RuntimeException("Unimplemented")
  }
  
  def printStatistics() {
    
  }
}

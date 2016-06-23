package edu.berkeley.nlp.summ

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import edu.berkeley.nlp.entity.DepConstTree
import edu.berkeley.nlp.entity.coref.Mention
import edu.berkeley.nlp.entity.coref.PronounDictionary
import edu.berkeley.nlp.entity.coref.MentionType
import edu.berkeley.nlp.entity.coref.CorefDoc
import edu.berkeley.nlp.entity.GUtil
import edu.berkeley.nlp.futile.math.SloppyMath

object CorefUtils {
  
  def getAntecedent(corefDoc: CorefDoc, predictor: edu.berkeley.nlp.entity.coref.PairwiseScorer, index: Int) = {
    val posteriors = computePosteriors(corefDoc, predictor, Seq(index))
    GUtil.argMaxIdx(posteriors(0))
  }
  
  def computePosteriors(corefDoc: CorefDoc, predictor: edu.berkeley.nlp.entity.coref.PairwiseScorer, indicesOfInterest: Seq[Int]): Array[Array[Double]] = {
    val docGraph = new edu.berkeley.nlp.entity.coref.DocumentGraph(corefDoc, false)
    Array.tabulate(indicesOfInterest.size)(idxIdxOfInterest => {
      val idx = indicesOfInterest(idxIdxOfInterest)
      val scores = Array.tabulate(idx+1)(antIdx => predictor.score(docGraph, idx, antIdx, false).toDouble)
      val logNormalizer = scores.foldLeft(Double.NegativeInfinity)(SloppyMath.logAdd(_, _))
      for (antIdx <- 0 until scores.size) {
        scores(antIdx) = scores(antIdx) - logNormalizer
      }
      scores
    })
  }
  
  /**
   * This exists to make results consistent with what was there before
   */
  def remapMentionType(ment: Mention) = {
    val newMentionType = if (ment.endIdx - ment.startIdx == 1 && PronounDictionary.isDemonstrative(ment.rawDoc.words(ment.sentIdx)(ment.headIdx))) {
      MentionType.DEMONSTRATIVE;
    } else if (ment.endIdx - ment.startIdx == 1 && PronounDictionary.isPronLc(ment.rawDoc.words(ment.sentIdx)(ment.headIdx))) {
      MentionType.PRONOMINAL;
    } else if (ment.rawDoc.pos(ment.sentIdx)(ment.headIdx) == "NNS" || ment.rawDoc.pos(ment.sentIdx)(ment.headIdx) == "NNPS") {
      MentionType.PROPER;
    } else {
      MentionType.NOMINAL;
    }
    new Mention(ment.rawDoc,
                ment.mentIdx,
                ment.sentIdx,
                ment.startIdx,
                ment.endIdx,
                ment.headIdx,
                ment.allHeadIndices,
                ment.isCoordinated,
                newMentionType,
                ment.nerString,
                ment.number,
                ment.gender)
                
  }
  
  def getMentionText(ment: Mention) = ment.rawDoc.words(ment.sentIdx).slice(ment.startIdx, ment.endIdx)
  
  def getMentionNerSpan(ment: Mention): Option[(Int,Int)] = {
    // Smallest NER chunk that contains the head
    val conllDoc = ment.rawDoc
    val matchingChunks = conllDoc.nerChunks(ment.sentIdx).filter(chunk => chunk.start <= ment.headIdx && ment.headIdx < chunk.end);
    if (!matchingChunks.isEmpty) {
      val smallestChunk = matchingChunks.sortBy(chunk => chunk.end - chunk.start).head;
      Some(smallestChunk.start -> smallestChunk.end)
    } else {
      None
    }
  }

  def getSpanHeads(tree: DepConstTree, startIdx: Int, endIdx: Int): Seq[Int] = getSpanHeads(tree.childParentDepMap, startIdx, endIdx);
  
  def getSpanHeads(childParentDepMap: HashMap[Int,Int], startIdx: Int, endIdx: Int): Seq[Int] = {
    // If it's a constituent, only one should have a head outside
    val outsidePointing = new ArrayBuffer[Int];
    for (i <- startIdx until endIdx) {
      val ptr = childParentDepMap(i);
      if (ptr < startIdx || ptr >= endIdx) {
        outsidePointing += i;
      }
    }
    outsidePointing
  }
  
  def isDefinitelyPerson(str: String): Boolean = {
    val canonicalization = PronounDictionary.canonicalize(str)
    // N.B. Don't check "we" or "they" because those might be used in inanimate cases
    canonicalization == "i" || canonicalization == "you" || canonicalization == "he" || canonicalization == "she" 
  }
}
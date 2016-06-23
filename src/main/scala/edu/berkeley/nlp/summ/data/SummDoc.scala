package edu.berkeley.nlp.summ.data

import java.io.File

import scala.collection.mutable.ArrayBuffer

import edu.berkeley.nlp.entity.ConllDocReader
import edu.berkeley.nlp.entity.coref.CorefDoc
import edu.berkeley.nlp.entity.coref.CorefDocAssembler
import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.entity.lang.EnglishCorefLanguagePack
import edu.berkeley.nlp.entity.lang.Language
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.summ.CorefUtils

@SerialVersionUID(2350732155930072470L)
case class SummDoc(val name: String,
                   val corefDoc: CorefDoc,
                   val doc: Seq[DepParse],
                   val summSents: Seq[Seq[String]],
                   val summary: Seq[DepParse]) extends DepParseDoc {
  
  def getSentMents(sentIdx: Int) = corefDoc.goldMentions.filter(_.sentIdx == sentIdx)
    
  val sentMentStartIndices = (0 to corefDoc.rawDoc.numSents).map(sentIdx => {
    val mentsFollowing = corefDoc.goldMentions.filter(_.sentIdx >= sentIdx)
    if (mentsFollowing.isEmpty) corefDoc.goldMentions.size else mentsFollowing.head.mentIdx;
  })
  val entitiesPerSent = (0 until corefDoc.rawDoc.numSents).map(sentIdx => getEntitiesInSpan(sentIdx, 0, doc(sentIdx).size))
  
  def getMentionsInSpan(sentIdx: Int, startIdx: Int, endIdx: Int) = {
    val mentsInSent = corefDoc.goldMentions.slice(sentMentStartIndices(sentIdx), sentMentStartIndices(sentIdx+1))
    mentsInSent.filter(ment => startIdx <= ment.startIdx && ment.endIdx <= endIdx)
  }
  
  def getEntitiesInSpan(sentIdx: Int, startIdx: Int, endIdx: Int) = {
    getMentionsInSpan(sentIdx, startIdx, endIdx).map(ment => corefDoc.goldClustering.getClusterIdx(ment.mentIdx)).distinct.sorted
  }
//  
  val entitiesBySize = corefDoc.goldClustering.clusters.zipWithIndex.map(clusterAndIdx => clusterAndIdx._1.size -> clusterAndIdx._2).sortBy(- _._1).map(_._2)
  val entitySemanticTypes = (0 until corefDoc.goldClustering.clusters.size).map(clusterIdx => {
    val cluster = corefDoc.goldClustering.clusters(clusterIdx)
    val types = new Counter[String]
    cluster.foreach(mentIdx => types.incrementCount(corefDoc.goldMentions(mentIdx).nerString, 1.0))
    types.removeKey("O")
    types.argMax()
  })
}

object SummDoc {
  
  def makeSummDoc(name: String, corefDoc: CorefDoc, summSents: Seq[Seq[String]]): SummDoc = {
    val doc = (0 until corefDoc.rawDoc.numSents).map(i => {
//      DepParse.fromDepConstTree(corefDoc.rawDoc.trees(i))
      new DepParseConllWrapped(corefDoc.rawDoc, i)
    })
    val summary = (0 until summSents.size).map(i => {
      new DepParseRaw(summSents(i).toArray, Array.tabulate(summSents(i).size)(i => "-"))
    })
    new SummDoc(name, corefDoc, doc, summSents, summary)
  }
  
  def readSummDocNoAbstract(docPath: String,
                            mentionPropertyComputer: MentionPropertyComputer,
                            filterSpuriousDocs: Boolean = false,
                            filterSpuriousSummSents: Boolean = false) = {
    val doc = new ConllDocReader(Language.ENGLISH).readConllDocs(docPath)(0)
    val assembler = new CorefDocAssembler(new EnglishCorefLanguagePack, true)
    val corefDoc = assembler.createCorefDoc(doc, mentionPropertyComputer)
    makeSummDoc(new File(docPath).getName, corefDoc, Seq[Seq[String]]())
  }
  
  def readSummDocs(docsPath: String,
                   abstractsPath: String,
                   abstractsAreConll: Boolean,
                   maxFiles: Int = -1,
                   mentionPropertyComputer: MentionPropertyComputer,
                   filterSpuriousDocs: Boolean = false,
                   filterSpuriousSummSents: Boolean = false,
                   docFilter: (SummDoc => Boolean)) =  {
    val docFiles = new File(docsPath).listFiles.sorted
    val processedDocs = new ArrayBuffer[SummDoc]
    var docIdx = 0
    var numSummSentsFiltered = 0
    var iter = docFiles.iterator
    val assembler = new CorefDocAssembler(new EnglishCorefLanguagePack, true)
    var filteredByDocFilter = 0
    var filteredBySpurious = 0
    // XXX
//    val otherMpc = new singledoc.coref.MentionPropertyComputer(Some(HorribleCorefMunger.reverseMungeNumberGenderComputer(mentionPropertyComputer.maybeNumGendComputer.get)))
//    val maybeCorefModel = Some(GUtil.load("models/coref-onto.ser.gz").asInstanceOf[PairwiseScorer])
    // XXX
    while ((maxFiles == -1 || docIdx < maxFiles) && iter.hasNext) {
      val docFile = iter.next
      if (docIdx % 500 == 0) {
        Logger.logss("  Processing document " + docIdx + "; kept " + processedDocs.size + " so far")
      }
      docIdx += 1
      val fileName = docFile.getName()
      val abstractFile = new File(abstractsPath + "/" + fileName)
      require(abstractFile.exists(), "Couldn't find abstract file at " + abstractsPath + "/" + fileName)
//      val doc = DepParse.readFromFile(docFile.getAbsolutePath())
      val doc = new ConllDocReader(Language.ENGLISH).readConllDocs(docFile.getAbsolutePath)(0)
      val summRaw = if (abstractsAreConll) {
        DepParse.readFromConllFile(abstractFile.getAbsolutePath())
      } else {
        DepParse.readFromConstFile(abstractFile.getAbsolutePath())
      }
      val summ = if (filterSpuriousSummSents) {
        summRaw.filter(sentParse => !SummaryAligner.identifySpuriousSentence(sentParse.getWords))
      } else {
        summRaw
      }
      numSummSentsFiltered += (summRaw.size - summ.size)
      if (summ.size == 0 || (filterSpuriousDocs && SummaryAligner.identifySpuriousSummary(summRaw(0).getWords))) {
//        val corefDoc = assembler.createCorefDoc(doc, mentionPropertyComputer)
//        val summDoc = makeSummDoc(fileName, corefDoc, summ.map(_.getWords.toSeq), docCompressor)
//        if (docFilter(summDoc)) {
//          filteredBySpurious += 1
//        } else {
//          filteredByDocFilter += 1
//        }
        // Do nothing
      } else {
        val rawCorefDoc = assembler.createCorefDoc(doc, mentionPropertyComputer)
        val rawGoldMents = rawCorefDoc.goldMentions.map(CorefUtils.remapMentionType(_))
        val corefDoc = new CorefDoc(rawCorefDoc.rawDoc, rawGoldMents, rawCorefDoc.goldClustering, rawGoldMents)
        val summDoc = makeSummDoc(fileName, corefDoc, summ.map(_.getWords.toSeq))
        if (docFilter(summDoc)) {
          processedDocs += summDoc 
        } else {
          filteredByDocFilter += 1
        }
      }
    }
    Logger.logss("Read docs from " + docsPath + " and abstracts from " + abstractsPath + "; filtered " +
                 (docFiles.size - processedDocs.size) + "/" + docFiles.size + " docs (" + filteredByDocFilter +
                 " from doc filter (len, etc.) and " + filteredBySpurious + " from spurious detection (article, etc.)) and " + numSummSentsFiltered + " sentences")
    processedDocs
  }
}
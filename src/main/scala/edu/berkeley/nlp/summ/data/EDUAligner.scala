package edu.berkeley.nlp.summ.data

import java.io.File

import scala.collection.mutable.ArrayBuffer

import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.entity.coref.NumberGenderComputer
import edu.berkeley.nlp.futile.util.Logger

object EDUAligner {
  
  def align(leafWords: Seq[Seq[String]], docSents: Seq[DepParse]) = {
    var currSentIdx = 0
    var currWordIdx = 0
    val leafSpans = new ArrayBuffer[((Int,Int),(Int,Int))]
    for (i <- 0 until leafWords.size) {
      val start = (currSentIdx, currWordIdx)
      val currLen = docSents(currSentIdx).size
      require(currWordIdx + leafWords(i).size <= currLen,
              currWordIdx + " " + leafWords(i).size + " " + currLen + "\nsent = " + docSents(currSentIdx).getWords.toSeq + ", leaf words = " + leafWords(i).toSeq)
      var leafWordIdx = 0
      while (leafWordIdx < leafWords(i).size) {
        val docWord = docSents(currSentIdx).getWord(currWordIdx)
        val leafWord = leafWords(i)(leafWordIdx)
        val currWordsEqual = docWord == leafWord
        val currWordsEffectivelyEqual = docWord.contains("'") || docWord.contains("`") // Ignore some punc symbols because they're weird
        // Spurious period but last thing ended in period, so it was probably added by the tokenizer (like "Ltd. .")
        if (!currWordsEqual && docWord == "." && currWordIdx > 0 && docSents(currSentIdx).getWord(currWordIdx - 1).endsWith(".")) {
          currWordIdx += 1
          if (currWordIdx == docSents(currSentIdx).size) {
            currSentIdx += 1
            currWordIdx = 0
          }
          // N.B. don't advance leafWordIdx
        } else {
          require(currWordsEqual || currWordsEffectivelyEqual, docWord + " :: " + leafWord + "\nsent = " + docSents(currSentIdx).getWords.toSeq + ", leaf words = " + leafWords(i).toSeq)
          currWordIdx += 1
          if (currWordIdx == docSents(currSentIdx).size) {
            currSentIdx += 1
            currWordIdx = 0
          }
          leafWordIdx += 1
        }
      }
      val end = if (currWordIdx == 0) {
        (currSentIdx - 1, docSents(currSentIdx - 1).size)
      } else {
        (currSentIdx, currWordIdx)
      }
      leafSpans += start -> end
//        if (currWordIdx == docSents(currSentIdx).size) {
//          currSentIdx += 1
//          currWordIdx = 0
//        }
    }
    leafSpans
//    }
  }
  
  def main(args: Array[String]) {
    val allTreeFiles = new File("data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES/").listFiles.sortBy(_.getName).filter(_.getName.endsWith(".out.dis"))
    val allTrees = allTreeFiles.map(file => DiscourseTreeReader.readDisFile(file.getAbsolutePath))
//    val allSummDocs = new File("data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES-PREPROC/").listFiles.sortBy(_.getName))
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData("data/gender.data");
    val mpc = new MentionPropertyComputer(Some(numberGenderComputer))
    val allSummDocFiles = new File("data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES-PROC2/").listFiles.sortBy(_.getName)
    val allSummDocs = allSummDocFiles.map(file => SummDoc.readSummDocNoAbstract(file.getAbsolutePath, mpc, filterSpuriousDocs = false, filterSpuriousSummSents = false))
    val summNames = new File("data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/SUMM-SUBSET-PROC/").listFiles.map(_.getName)
    require(allTrees.size == allSummDocs.size)
    val badFiles = new ArrayBuffer[String]
    for (i <- 0 until allTrees.size) {
      require(allTreeFiles(i).getName.dropRight(4) == allSummDocFiles(i).getName, allTreeFiles(i).getName.dropRight(4) + " " + allSummDocFiles(i).getName)
      Logger.logss(allSummDocFiles(i).getName)
      try {
        align(allTrees(i).leafWords, allSummDocs(i).doc)
      } catch {
        case e: Exception => {
          Logger.logss(e)
          badFiles += allSummDocFiles(i).getName
        }
      }
    }
    Logger.logss(badFiles.size + " bad files: " + badFiles)
    val badSummDocs = (badFiles.toSet & summNames.toSet)
    Logger.logss(badSummDocs.size + " bad summarized files: " + badSummDocs.toSeq.sorted) 
  }
}
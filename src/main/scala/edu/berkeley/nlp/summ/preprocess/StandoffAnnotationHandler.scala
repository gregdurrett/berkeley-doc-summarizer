package edu.berkeley.nlp.summ.preprocess

import edu.berkeley.nlp.entity.ConllDocReader
import edu.berkeley.nlp.entity.lang.Language
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.util.Counter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import edu.berkeley.nlp.entity.ConllDoc
import java.io.File
import edu.berkeley.nlp.entity.ConllDocWriter
import edu.berkeley.nlp.futile.LightRunner

/**
 * Handles combining standoff annotations with data files from the New York Times corpus.
 */
object StandoffAnnotationHandler {
  
  val readAnnotations = true
  val inputDir = "../summ-data/nyt08/trainsm_corefner_standoff/"
//  val outputDir = "../summ-data/nyt08/trainsm_corefner_standoff/"
  val outputDir = "../summ-data/nyt08/trainsm_corefner_reconstituted/"
  val rawXMLDir = "../summ-data/nyt08/train/"
  
  val maxNumFiles = Int.MaxValue
  
  val tagName = "block class=\"full_text\""
//  val tagName = "abstract"

  def main(args: Array[String]) {
    LightRunner.initializeOutput(StandoffAnnotationHandler.getClass())
    LightRunner.populateScala(StandoffAnnotationHandler.getClass(), args)
    
//    val corefNerFile = "../summ-data/nyt08/trainsm_corefner/1570865"
//    val rawXMLFile = "../summ-data/nyt08/train/1570865.xml"
//    align(corefNerFile, rawXMLFile)
//    System.exit(0)
    
    if (readAnnotations) {
      val corefNerFilesToAlign = new File(inputDir).listFiles()
      for (corefNerFile <- corefNerFilesToAlign) {
        val rawXMLPath = rawXMLDir + "/" + corefNerFile.getName() + ".xml"
        Logger.logss("Dealing with " + corefNerFile.getAbsolutePath)
        val reconstitutedConllDoc = reverseStandoffAnnotations(corefNerFile.getAbsolutePath, rawXMLPath)
        if (reconstitutedConllDoc.docID == "") {
          val path = outputDir + "/" + corefNerFile.getName
          val outWriter = IOUtils.openOutHard(path)
          Logger.logss("Wrote to " + path)
          outWriter.close
        } else {
          val path = outputDir + "/" + reconstitutedConllDoc.docID
          val outWriter = IOUtils.openOutHard(path)
          ConllDocWriter.writeDoc(outWriter, reconstitutedConllDoc)
          Logger.logss("Wrote to " + path)
          outWriter.close
        }
      }
    } else {
      val rawCorefNerFilesToAlign = new File(inputDir).listFiles()
      val corefNerFilesToAlign = rawCorefNerFilesToAlign.slice(0, Math.min(maxNumFiles, rawCorefNerFilesToAlign.size))
      for (corefNerFile <- corefNerFilesToAlign) {
        val rawXMLPath = rawXMLDir + "/" + corefNerFile.getName() + ".xml"
        Logger.logss("=========================")
        Logger.logss("Aligning " + corefNerFile.getAbsolutePath + " " + rawXMLPath)
        val standoffConllDoc = makeStandoffAnnotations(corefNerFile.getAbsolutePath, rawXMLPath, tagName)
        // This happens if we have a blank file
        if (standoffConllDoc.docID.isEmpty) {
          // Write nothing
          val path = outputDir + "/" + corefNerFile.getName
          val outWriter = IOUtils.openOutHard(path)
          outWriter.close
        } else {
          val path = outputDir + "/" + standoffConllDoc.docID
          val outWriter = IOUtils.openOutHard(path)
          ConllDocWriter.writeDoc(outWriter, standoffConllDoc)
          Logger.logss("Wrote to " + path)
          outWriter.close
        }
      }
    }
    LightRunner.finalizeOutput()
  }
  
  val tokenizationMapping = new HashMap[String,String] ++ Seq("(" -> "-LRB-", ")" -> "-RRB-",
                                                              "[" -> "-LSB-", "]" -> "-RSB-",
                                                              "{" -> "-LCB-", "}" -> "-RCB-",
                                                              "&amp;" -> "&")
  val reverseTokenizationMapping = new HashMap[String,String]
  for (entry <- tokenizationMapping) {
    reverseTokenizationMapping += entry._2 -> entry._1
  }
  
//  def extractWords(alignments: ArrayBuffer[ArrayBuffer[(Int,Int,Int)]], docLines: Seq[String]) = {
//    alignments.map(_.map(alignment => docLines(alignment._1).substring(alignment._2, alignment._3)))
//  }
  
  // Use :: as delimiter
  val delimiter = "::"
  val alignmentRe = ("[0-9]+" + delimiter + "[0-9]+" + delimiter + "[0-9]+").r
  
  def extractWord(alignment: String, docLines: Seq[String]) = {
    if (alignmentRe.findFirstIn(alignment) != None) {
      val alignmentSplit = alignment.split(delimiter).map(_.toInt)
      val word = docLines(alignmentSplit(0)).substring(alignmentSplit(1), alignmentSplit(2))
      if (tokenizationMapping.contains(word)) {
        tokenizationMapping(word)
      } else {
        word
      }
    } else {
      alignment
    }
  }
  
  def extractWords(alignments: ArrayBuffer[ArrayBuffer[String]], docLines: Seq[String]) = {
    alignments.map(_.map(alignment => extractWord(alignment, docLines)))
  }
  
  def fetchCurrWord(corefNerDoc: ConllDoc, sentIdx: Int, wordIdx: Int) = {
    val word = corefNerDoc.words(sentIdx)(wordIdx)
    if (reverseTokenizationMapping.contains(word)) {
      reverseTokenizationMapping(word)
    } else {
      word
    }
  }
  
  def doMatch(lineChar: Char, docChar: Char) = {
    lineChar == docChar || Character.toLowerCase(lineChar) == Character.toLowerCase(docChar) ||
      ((lineChar == ';' || lineChar == '.' || lineChar == '?' || lineChar == '!') && (docChar == ';' || docChar == '.' || docChar == '?' || docChar == '!')) ||
      ((lineChar == ''' || lineChar == '`') && (docChar == ''' || docChar == '`'))
      
  }
  
  def makeStandoffAnnotations(corefNerFile: String, rawXMLFile: String, tagName: String) = {
    val corefNerDoc = new ConllDocReader(Language.ENGLISH).readConllDocs(corefNerFile).head
    val rawXMLLines = IOUtils.readLinesHard(rawXMLFile).asScala
    // Only do the alignment if there are a nonzero number of words
    val newConllDoc = if (corefNerDoc.words.size > 0) {
      val alignments = align(corefNerDoc, rawXMLLines, tagName: String)
      new ConllDoc(corefNerDoc.docID,
                   corefNerDoc.docPartNo,
                   alignments,
                   corefNerDoc.pos,
                   corefNerDoc.trees,
                   corefNerDoc.nerChunks,
                   corefNerDoc.corefChunks,
                   corefNerDoc.speakers)
    } else {
      corefNerDoc
    }
    newConllDoc
  }
  
  def align(corefNerDoc: ConllDoc, rawXMLLines: Seq[String], tagName: String) = {
    val closeTagName = tagName.substring(0, if (tagName.contains(" ")) tagName.indexOf(" ") else tagName.size)
    var sentPtr = 0
    var wordPtr = 0
    var charPtr = 0
    var currWord = fetchCurrWord(corefNerDoc, sentPtr, wordPtr)
    val alignments = new ArrayBuffer[ArrayBuffer[String]]
    alignments += new ArrayBuffer[String]
    var inText = false
    var badCharactersSkipped = 0
    // Iterate through the XML file and advance through the CoNLL document simultaneously
    for (lineIdx <- 0 until rawXMLLines.size) {
      val line = rawXMLLines(lineIdx)
      if (line.contains("</" + closeTagName + ">")) {
        inText = false
      }
      if (inText) {
//        Logger.logss(sentPtr)
        if (!line.contains("<p>") || !line.contains("</p>")) {
          Logger.logss("ANOMALOUS LINE")
          Logger.logss(line)
          inText = false
        }
        val lineStart = line.indexOf("<p>") + 3
        val relevantLine = line.substring(lineStart, line.indexOf("</p>"))
//        Logger.logss("CURRENT LINE: " + relevantLine)
        var linePtr = 0
        while (inText && linePtr < relevantLine.size) {
//          Logger.logss("Checking line index " + linePtr + "; looking for " + corefNerDoc.words(sentPtr)(wordPtr)(charPtr) + " and was " + relevantLine.substring(linePtr, linePtr+1))
          if (doMatch(relevantLine.charAt(linePtr), currWord(charPtr))) {
//            Logger.logss("Matching " + relevantLine.substring(linePtr, linePtr+1))
            charPtr += 1
          } else if (!Character.isWhitespace(relevantLine.charAt(linePtr))) {
            badCharactersSkipped += 1
//            Logger.logss("Bad character skipped! " + relevantLine.charAt(linePtr) + " " + currWord(charPtr))
          }
          if (charPtr == currWord.size) {
            // Store the alignment
            val alignment = lineIdx + delimiter + (lineStart + linePtr - currWord.size + 1) + delimiter  + (lineStart + linePtr + 1)
            if (linePtr - currWord.size + 1 >= 0 && relevantLine.substring(linePtr - currWord.size + 1, linePtr + 1) == currWord) {
              alignments.last += alignment
            } else {
//              Logger.logss("Mismatch: :" + currWord + ": :" + relevantLine.substring(linePtr - currWord.size + 1, linePtr + 1) + ":")
              alignments.last += currWord
            }
//            Logger.logss("Storing alignment: " + currWord + " " + alignment)
            wordPtr += 1
            charPtr = 0
          }
          if (wordPtr == corefNerDoc.words(sentPtr).size) {
            alignments += new ArrayBuffer[String]
            sentPtr += 1
//            Logger.logss("NEW SENTENCE: " + corefNerDoc.words(sentPtr).reduce( _ + " " + _))
            wordPtr = 0
          }
          // If we're all done
          if (sentPtr >= corefNerDoc.words.size) {
            inText = false
          } else {
            // Otherwise, possibly update the current word we're targeting
            currWord = fetchCurrWord(corefNerDoc, sentPtr, wordPtr)
          }
          linePtr += 1
        }
      } else if (line.contains("<" + tagName + ">")) {
        inText = true
      }
    }
    // Drop the last entry if it's empty, which it will be if everything is consumed.
    if (alignments.last.isEmpty && alignments.size >= 2) {
      alignments.remove(alignments.size - 1)
    }
    // Check if all lines are the right length. If not, we should just dump the whole file.
    var catastrophicError = false
    if (alignments.size != corefNerDoc.words.size) {
      Logger.logss("Wrong number of lines! " + alignments.size + " " + corefNerDoc.words.size)
      catastrophicError = true
    }
    for (lineIdx <- 0 until Math.min(alignments.size, corefNerDoc.words.size)) {
      val alignmentLen = alignments(lineIdx).size
      val corefNerDocLen = corefNerDoc.words(lineIdx).size
      if (alignmentLen != corefNerDocLen) {
        Logger.logss("Wrong number of words in line " + lineIdx + "! " + alignments(lineIdx).size + " " + corefNerDoc.words(lineIdx).size)
        // Primarily useful for repairing sentence-final punctuation that was added during preprocessing
        if (alignmentLen == corefNerDocLen - 1 && (alignmentLen == 0 || extractWord(alignments(lineIdx)(alignmentLen - 1), rawXMLLines) == corefNerDoc.words(lineIdx)(alignmentLen - 1))) {
          Logger.logss("Repaired!")
          alignments(lineIdx) += corefNerDoc.words(lineIdx).last
        } else {
          catastrophicError = true
        }
      }
    }
    // If we've had a catastrophic error, just use the raw words rather than standoff annotations.
    val finalAlignments = if (catastrophicError) {
      Logger.logss("XXXXXXXX CATASTROPHIC ERROR XXXXXXXX")
      new ArrayBuffer[ArrayBuffer[String]] ++ corefNerDoc.words.map(new ArrayBuffer[String] ++ _)
    } else {
      alignments
    }
    // Verify that the words are the same
    val reextractedWords = extractWords(finalAlignments, rawXMLLines)
    var someMistake = false
    var standoffCounter = 0
    var missCounter = 0
    for (lineIdx <- 0 until finalAlignments.size) {
      for (wordIdx <- 0 until finalAlignments(lineIdx).size) {
        if (alignmentRe.findFirstIn(finalAlignments(lineIdx)(wordIdx)) != None) {
          standoffCounter += 1
        } else {
          missCounter += 1
        }
        if (reextractedWords(lineIdx)(wordIdx) != corefNerDoc.words(lineIdx)(wordIdx)) {
          Logger.logss("Mismatched word! " + reextractedWords(lineIdx)(wordIdx) + " " + corefNerDoc.words(lineIdx)(wordIdx))
          someMistake = true
        }
      }
    }
    Logger.logss(badCharactersSkipped + " bad characters skipped")
    Logger.logss(standoffCounter + " standoffs, " + missCounter + " raw strings")
    if (!someMistake) {
      Logger.logss("******** ALIGNED CORRECTLY! ********")
    } else {
      Logger.logss("XXXXXXXX ALIGNED INCORRECTLY! XXXXXXXX")
    }
    finalAlignments
  }
  
  def reverseStandoffAnnotations(corefNerFile: String, rawXMLFile: String) = {
    val standoffConllDoc = new ConllDocReader(Language.ENGLISH).readConllDocs(corefNerFile).head
    val rawXMLLines = IOUtils.readLinesHard(rawXMLFile).asScala
    val words = extractWords(new ArrayBuffer[ArrayBuffer[String]] ++ standoffConllDoc.words.map(new ArrayBuffer[String] ++ _), rawXMLLines)
    val reconstitutedConllDoc = new ConllDoc(standoffConllDoc.docID,
                                             standoffConllDoc.docPartNo,
                                             words,
                                             standoffConllDoc.pos,
                                             standoffConllDoc.trees,
                                             standoffConllDoc.nerChunks,
                                             standoffConllDoc.corefChunks,
                                             standoffConllDoc.speakers)
    reconstitutedConllDoc
  }
  
  // Only needed once, to convert PTB parses to CoNLL docs
//  def convertParsesToDocs
}
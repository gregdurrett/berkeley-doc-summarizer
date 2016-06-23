package edu.berkeley.nlp.summ.data

import java.io.BufferedReader
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.entity.ConllDoc
import edu.berkeley.nlp.entity.DepConstTree
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeReader
import edu.berkeley.nlp.entity.lang.Language
import edu.berkeley.nlp.entity.ConllDocReader

/**
 * Named DepParse for legacy reasons -- actually just a tagged sentence.
 */
trait DepParse extends Serializable {
  def size: Int
  def getWord(idx: Int): String;
  def getWords: Array[String];
  def getPos(idx: Int): String;
  def getPoss: Array[String];
}

object DepParse {
  
  def readFromFile(file: String): Seq[DepParseRaw] = {
    val reader = IOUtils.openInHard(file)
    val parses = readFromFile(reader)
    reader.close()
    parses
  }
  
  def readFromFile(reader: BufferedReader): Seq[DepParseRaw] = {
    val lineItr = IOUtils.lineIterator(reader)
    val sents = new ArrayBuffer[DepParseRaw];
    val currWords = new ArrayBuffer[String];
    val currPoss = new ArrayBuffer[String];
    val currParents = new ArrayBuffer[Int];
    val currLabels = new ArrayBuffer[String];
    while (lineItr.hasNext) {
      val line = lineItr.next
      if (line.trim.isEmpty) {
        sents += new DepParseRaw(currWords.toArray, currPoss.toArray)
        currWords.clear()
        currPoss.clear()
        currParents.clear()
        currLabels.clear()
      } else {
        val splitLine = line.split("\\s+")
//        println(splitLine.toSeq)
        require(splitLine.size == 10, "Wrong number of fields in split line " + splitLine.size + "; splits = " + splitLine.toSeq)
        currWords += splitLine(1)
        currPoss += splitLine(4)
        currParents += splitLine(6).toInt - 1
        currLabels += splitLine(7)
      }
    }
    if (!currWords.isEmpty) {
      sents += new DepParseRaw(currWords.toArray, currPoss.toArray)
    }
    sents
  }
  
  def readFromConstFile(file: String): Seq[DepParseRaw] = {
    val sents = new ArrayBuffer[DepParseRaw];
    val reader = IOUtils.openInHard(file)
    val lineItr = IOUtils.lineIterator(reader)
    while (lineItr.hasNext) {
      val tree = PennTreeReader.parseHard(lineItr.next, false)
      val words = tree.getYield().asScala
      val pos = tree.getYield().asScala
      sents += new DepParseRaw(words.toArray, pos.toArray)
    }
    reader.close()
    sents
  }
  
  def readFromConllFile(file: String): Seq[DepParseConllWrapped] = {
    val conllDoc = new ConllDocReader(Language.ENGLISH).readConllDocs(file, -1).head
    (0 until conllDoc.numSents).map(i => new DepParseConllWrapped(conllDoc, i))
  }
  
  def fromDepConstTree(depConstTree: DepConstTree) = {
    // Build a dep parse with no labels
    val parents = Array.tabulate(depConstTree.size)(i => depConstTree.childParentDepMap(i))
    new DepParseRaw(depConstTree.words.toArray, depConstTree.pos.toArray)
  }
}

@SerialVersionUID(1565751158494431431L)
class DepParseRaw(val words: Array[String],
                  val poss: Array[String]) extends DepParse {
  
  def size = words.size
  def getWord(idx: Int): String = words(idx)
  def getWords = words
  def getPos(idx: Int): String = poss(idx)
  def getPoss = poss
}

@SerialVersionUID(4946839606729754922L)
class DepParseConllWrapped(val conllDoc: ConllDoc,
                           val sentIdx: Int) extends DepParse {
  
  def size = conllDoc.words(sentIdx).size
  def getWord(idx: Int): String = conllDoc.words(sentIdx)(idx)
  def getWords = conllDoc.words(sentIdx).toArray
  def getPos(idx: Int): String = conllDoc.pos(sentIdx)(idx)
  def getPoss = conllDoc.pos(sentIdx).toArray
}

package edu.berkeley.nlp.summ.data

import java.io.File

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer

import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Logger

object DiscourseTreeReader {
  
  def readDisFile(path: String) = {
    val file = new File(path)
    val reader = IOUtils.openInHard(file)
    val iter = IOUtils.lineIterator(reader)
    
    val currNodeStack = new ArrayBuffer[DiscourseNode]
    while (iter.hasNext) {
      var line = iter.next
      if (line.trim.startsWith(")")) {
        val finishedNode = currNodeStack.remove(currNodeStack.size - 1)
        if (!currNodeStack.isEmpty) {
          currNodeStack.last.children += finishedNode
        } else {
          currNodeStack += finishedNode
        }
      } else { // (
        val text = if (line.contains("(text _!")) {
          val textStart = line.indexOf("_!") + 2
          val textEnd = line.lastIndexOf("_!")
          val result = line.substring(textStart, textEnd)
          line = line.substring(0, textStart) + line.substring(textEnd)
          result
        } else {
          ""
        }
        val lineCut = line.trim.drop(1).trim
        val label = lineCut.substring(0, lineCut.indexOf(" "))
        require(label == DiscourseNode.Nucleus || label == DiscourseNode.Satellite || label == DiscourseNode.Root, "Bad label: " + label + "; " + line)
        val spanText = readField(line, "span")
        var isLeaf = false
        val span = if (!spanText.isEmpty) {
          isLeaf = false
          makeSpan(spanText)
        } else {
          isLeaf = true
          val leafText = readField(line, "leaf")
          require(!leafText.isEmpty, line)
          (leafText.toInt - 1) -> leafText.toInt
        }
        val rel2par = readField(line, "rel2par")
        val newNode = new DiscourseNode(label, rel2par, span, text, ArrayBuffer[DiscourseNode]())
        if (currNodeStack.isEmpty || !isLeaf) {
          currNodeStack += newNode
        } else {
          currNodeStack.last.children += newNode
        }
      }
    }
    require(currNodeStack.size == 1, currNodeStack.size)
    reader.close()
    new DiscourseTree(file.getName, currNodeStack.head)
  }
  
  def writeDisFile(tree: DiscourseTree) {
    writeDisFileHelper(tree.rootNode, "")
    Logger.logss("PARENTS: " + tree.parents.toSeq)
  }
  
  def writeDisFileHelper(node: DiscourseNode, currStr: String) {
    if (node.children.isEmpty) {
      val leafIdx = node.span._1 // node.span._2 is what is used in the dataset
      Logger.logss(currStr + "( " + node.label + " (leaf " + leafIdx + ") (rel2par " + node.rel2par + ") (text _!" + node.leafText + "_!) ) (HEAD = " + node.head + ")")
    } else {
      val spanStart = node.span._1 // node.span._1 + 1 is what is used in the dataset
      Logger.logss(currStr + "( " + node.label + " (span " + spanStart + " " + node.span._2 + ") (rel2par " + node.rel2par + ") (HEAD = " + node.head + ")")
      for (child <- node.children) {
        writeDisFileHelper(child, currStr + "  ")
      }
      Logger.logss(currStr + ")")
    }
  }
  
  def makeSpan(str: String) = {
    val strSplit = str.split("\\s+")
    require(strSplit.size == 2, strSplit)
    (strSplit(0).toInt - 1) -> strSplit(1).toInt
  }
  
  def readField(line: String, label: String) = {
    if (line.contains("(" + label + " ")) {
      val start = line.indexOf("(" + label + " ")
      val end = line.indexOf(")", start)
//      Logger.logss(line + " " + start + " " + end)
      line.substring(start + label.size + 2, end)
    } else {
      ""
    }
  }
  
  def readAllAlignAndFilter(preprocDocsPath: String, discourseTreesPath: String, mpc: MentionPropertyComputer): Seq[DiscourseDepEx] = {
    val allTreeFiles = new File(discourseTreesPath).listFiles.sortBy(_.getName).filter(_.getName.endsWith(".out.dis"))
    val allTrees = allTreeFiles.map(file => DiscourseTreeReader.readDisFile(file.getAbsolutePath))
    val allSummDocFiles = new File(preprocDocsPath).listFiles.sortBy(_.getName)
    val allSummDocs = allSummDocFiles.map(file => SummDoc.readSummDocNoAbstract(file.getAbsolutePath, mpc, filterSpuriousDocs = false, filterSpuriousSummSents = false))
    require(allTrees.size == allSummDocs.size)
    val badFiles = new ArrayBuffer[String]
    val exs = new ArrayBuffer[DiscourseDepEx]
    for (i <- 0 until allTrees.size) {
      require(allTreeFiles(i).getName.dropRight(4) == allSummDocFiles(i).getName, allTreeFiles(i).getName.dropRight(4) + " " + allSummDocFiles(i).getName)
      Logger.logss(allSummDocFiles(i).getName)
      try {
        val alignments = EDUAligner.align(allTrees(i).leafWords, allSummDocs(i).doc)
        exs += new DiscourseDepEx(allSummDocs(i), allTrees(i), alignments)
      } catch {
        case e: Exception => {
          Logger.logss(e)
          badFiles += allSummDocFiles(i).getName
        }
      }
    } 
    Logger.logss("Read in " + exs.size + " out of " + allTrees.size + " possible")
    Logger.logss(badFiles.size + " bad files: " + badFiles)
    exs;
  }
  
  def readHiraoParents(file: String) = {
    val lines = IOUtils.readLines(file).asScala
    val parents = Array.fill(lines.size)(-100)
    for (line <- lines) {
      val childParent = line.replace("leaf", "").replace("->", "").split(" ")
      parents(childParent(0).toInt - 1) = childParent(1).toInt - 1
    }
    require(parents.filter(_ == -100).size == 0, "Null lingered")
    parents
  }
  
  def main(args: Array[String]) {
    val allFiles = new File("data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES/").listFiles.filter(_.getName.endsWith(".out.dis"))
    val hiraoDir = new File("data/dependency/")
    val hiraoFiles = hiraoDir.listFiles.map(_.getName)
    var totalEduLen = 0
    var numEdus = 0
    for (file <- allFiles) {
      println("====================")
      println(file.getName)
      val tree = readDisFile(file.getAbsolutePath)
      if (hiraoFiles.contains(file.getName)) {
        val referenceParents = readHiraoParents(hiraoDir.getAbsolutePath + "/" + file.getName)
        Logger.logss("REFERENCE: " + referenceParents.toSeq)
        Logger.logss("   NORMAL: " + tree.parents.toSeq)
        require(tree.parents.size == referenceParents.size, tree.parents.size + " " + referenceParents.size)
        def hammingDist(candParents: Seq[Int]) = (0 until referenceParents.size).map(i => if (candParents(i) != referenceParents(i)) 1 else 0).reduce(_ + _)
        Logger.logss("Normal parents: Hamming dist on " + file.getName + ": " + hammingDist(tree.parents) + " / " + referenceParents.size)
      }
//      writeDisFile(tree)
      totalEduLen += tree.leaves.foldLeft(0)(_ + _.leafWords.size)
      numEdus += tree.leaves.size
    }
    Logger.logss(totalEduLen + " / " + numEdus + " = " + (totalEduLen.toDouble / numEdus))
//    val root = readDisFile("data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_0600.out.dis")
//    writeDisFile(root)
  }
}
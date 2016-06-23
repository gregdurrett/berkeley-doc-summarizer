package edu.berkeley.nlp.summ.compression

import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.mutable.HashMap
import scala.collection.JavaConverters
import edu.berkeley.nlp.futile.LightRunner
import java.util.IdentityHashMap
import edu.berkeley.nlp.futile.syntax.Tree
import scala.collection.JavaConverters._
import edu.berkeley.nlp.summ.data.SummDoc

object SyntacticCompressor {
  
  def findComprParents(comprs: Seq[(Int,Int)]): Seq[Int] = {
    var someUnset = true
    val parents = Array.tabulate(comprs.size)(i => -2)
    while (someUnset) {
      var someUnsetThisItr = false
      for (idx <- 0 until comprs.size) {
        if (parents(idx) == -2) {
          var containedInNone = true
          var containedIdx = -1
          var containerSize = Int.MaxValue
          val compr = comprs(idx)
          for (comparisonIdx <- 0 until comprs.size) {
            if (comparisonIdx != idx) {
              val possibleParent = comprs(comparisonIdx)
              val possibleParentSize = possibleParent._2 - possibleParent._1
              if (isContained(compr, possibleParent) && possibleParentSize < containerSize)  {
                containedInNone = false
                containedIdx = comparisonIdx
                containerSize = possibleParentSize
              }
            }
          }
          if (containedInNone) {
            parents(idx) = -1 
          } else if (containedIdx != -1) {
            parents(idx) = containedIdx
          } else {
            someUnsetThisItr = true
          }
        }
      }
      someUnset = someUnsetThisItr
    }
    parents
  }
  
  def isContained(containee: (Int,Int), container: (Int,Int)) = {
    container._1 <= containee._1 && containee._2 <= container._2 
  }
  
  def identifyCuts(tree: Tree[String]) = {
    val processedTree = TreeProcessor.processTree(tree)
    require(processedTree.getYield.size == tree.getYield.size)
    val indicesToCut = new ArrayBuffer[(Int,Int)]
    val parentsMap = SentenceCompressor.getParentTrees(processedTree)
    identifyCutsHelper(processedTree, 0, processedTree.getYield().size, parentsMap)
  }
  
  val emptyCuts = new ArrayBuffer[(Int,Int)]
  
  def identifyCutsHelper(tree: Tree[String], startIdx: Int, endIdx: Int, parentsMap: IdentityHashMap[Tree[String],Tree[String]]): ArrayBuffer[(Int,Int)] = {
    if (tree.isLeaf) {
      emptyCuts
    } else {
      val legalCuts = new ArrayBuffer[(Int,Int)]
      val thisTreeFeats = SentenceCompressor.getCutFeatures(tree, startIdx, endIdx, parentsMap)
      if (thisTreeFeats.size > 0) {
        legalCuts += (startIdx -> endIdx)
      }
      val children = tree.getChildren()
      var currStartIdx = startIdx
      for (child <- children.asScala) {
        legalCuts ++= identifyCutsHelper(child, currStartIdx, currStartIdx + child.getYield.size, parentsMap)
        currStartIdx += child.getYield.size
      }
      legalCuts
    }
  }

  def compress(rawDoc: SummDoc): (Seq[((Int,Int),(Int,Int))], Seq[Int], Seq[String]) = {
    val conllDoc = rawDoc.corefDoc.rawDoc
    val possibleCompressionsEachSent: Seq[Set[(Int,Int)]] = (0 until conllDoc.numSents).map(i => identifyCuts(conllDoc.trees(i).constTree).toSet)
    compress(possibleCompressionsEachSent, conllDoc.words.map(_.size))
  }
  
  def compress(possibleCompressionsEachSent: Seq[Set[(Int,Int)]], sentLens: Seq[Int]): (Seq[((Int,Int),(Int,Int))], Seq[Int], Seq[String]) = {
    val numSents = sentLens.size
    val chunks = new ArrayBuffer[((Int,Int),(Int,Int))]
    val parents = new ArrayBuffer[Int]
    val labels = new ArrayBuffer[String]
    for (sentIdx <- 0 until numSents) {
      val (sentChunks, sentParents, sentLabels) = compressSentence(possibleCompressionsEachSent(sentIdx), sentLens(sentIdx))
      val sentOffset = chunks.size
      chunks ++= sentChunks.map(chunk => (sentIdx -> chunk._1) -> (sentIdx -> chunk._2))
      // Offset all chunks by the number of chunks occurring earlier in the document, same with
      parents ++= sentParents.map(parent => if (parent == -1) -1 else parent + sentOffset)
      labels ++= sentLabels
    }
    (chunks.toSeq, parents.toSeq, labels.toSeq)
  }
  
  def compressSentence(possibleCompressions: Set[(Int,Int)], sentLen: Int) = {
//    Logger.logss(possibleCompressions + " " + sentLen)
    val orderedComprs = Seq((0, sentLen)) ++ possibleCompressions.toSeq.sortBy(_._1)
    val sentComprParents = findComprParents(orderedComprs)
    // Segments start at 0 and wherever something happens with the compressions. The last always ends
    // at the end of the sentence but this doesn't appear in the list
    val boundaries = (Seq(0) ++ orderedComprs.map(_._1) ++ orderedComprs.map(_._2).filter(_ != sentLen)).toSet.toSeq.sorted
    val segmentToCompressionMapping = Array.tabulate(boundaries.size)(idx => {
      val start = boundaries(idx)
      val end = if (idx == boundaries.size - 1) sentLen else boundaries(idx+1)
      var segIdx = -1
      var comprLen = Int.MaxValue
      // Find the smallest compression containing this segment
      for (comprIdx <- 0 until orderedComprs.size) {
//          if (isContained((start, end), orderedComprs(comprIdx)) && (sentComprParents(comprIdx) == -1 || !isContained((start, end), orderedComprs(sentComprParents(comprIdx))))) {
        val newComprLen = orderedComprs(comprIdx)._2 - orderedComprs(comprIdx)._1
        if (isContained((start, end), orderedComprs(comprIdx)) && newComprLen < comprLen) {
          segIdx = comprIdx
          comprLen = newComprLen
        }
      }
      segIdx
    })
//      Logger.logss("B: " + boundaries)
//      Logger.logss("STCM: " + segmentToCompressionMapping.toSeq)
    val compressionToSegmentMapping = Array.tabulate(orderedComprs.size)(comprIdx => {
      (0 until segmentToCompressionMapping.size).filter(segIdx => segmentToCompressionMapping(segIdx) == comprIdx)
    })
    val sentParents = new ArrayBuffer[Int]
    val sentLabels = new ArrayBuffer[String]
    // If it's not in a compression, hook it to the first one of the sentence with =SameUnit
    // If it is a compression, hook it to its immediately larger compression or to the first one
    // of the sentence with Compression
    for (i <- 0 until boundaries.size) {
      val myCompr = segmentToCompressionMapping(i)
      val myOtherSegs = compressionToSegmentMapping(myCompr)
      if (myOtherSegs.indexOf(i) == 0) {
        if (sentComprParents(myCompr) == -1) {
          sentParents += -1
          sentLabels += ""
        } else {
          var parentWithSegs = sentComprParents(myCompr)
          while (parentWithSegs != -1 && compressionToSegmentMapping(parentWithSegs).isEmpty) {
            parentWithSegs = sentComprParents(parentWithSegs)
          }
          if (parentWithSegs == -1) {
            sentParents += -1
            sentLabels += "Compression"
          } else {
            sentParents += compressionToSegmentMapping(parentWithSegs).head
            sentLabels += "Compression"
          }
        }
        // Return the parent
      } else {
        sentParents += myOtherSegs(0)
        sentLabels += "=SameUnit"
      }
    }
    val sentChunks = (0 until boundaries.size).map(i => boundaries(i) -> (if (i == boundaries.size - 1) sentLen else boundaries(i+1)))
    (sentChunks, sentParents, sentLabels)
  }
  
  def refineEdus(rawDoc: SummDoc, eduAlignments: Seq[((Int,Int),(Int,Int))], parents: Seq[Int], labels: Seq[String]): (Seq[((Int,Int),(Int,Int))], Seq[Int], Seq[String]) = {
    val conllDoc = rawDoc.corefDoc.rawDoc
    val possibleCompressionsEachSent: Seq[Set[(Int,Int)]] = (0 until conllDoc.numSents).map(i => identifyCuts(conllDoc.trees(i).constTree).toSet)
    refineEdus(possibleCompressionsEachSent, conllDoc.words.map(_.size), eduAlignments, parents, labels)
  }
  
  def refineEdus(possibleCompressionsEachSent: Seq[Set[(Int,Int)]], sentLens: Seq[Int], edus: Seq[((Int,Int),(Int,Int))], parents: Seq[Int], labels: Seq[String]) = {
    require(edus.size == parents.size && edus.size == labels.size, edus.size + " " + parents.size + " " + labels.size)
    val numSents = sentLens.size
    val newChunks = new ArrayBuffer[((Int,Int),(Int,Int))]
    val newParents = new ArrayBuffer[Int]
    val newLabels = new ArrayBuffer[String]
    for (sentIdx <- 0 until numSents) {
      val sentEduStartIdx = edus.filter(_._1._1 < sentIdx).size
      val sentEduEndIdx = edus.filter(_._1._1 <= sentIdx).size
      require(sentEduEndIdx > sentEduStartIdx)
      val origEdus = edus.slice(sentEduStartIdx, sentEduEndIdx).map(edu => edu._1._2 -> edu._2._2)
      val adjustedParents = parents.slice(sentEduStartIdx, sentEduEndIdx).map(parent => if (parent < sentEduStartIdx || parent >= sentEduEndIdx) -1 else (parent - sentEduStartIdx))
      val (sentChunks, sentParents, sentLabels) = refineEdusInSentence(possibleCompressionsEachSent(sentIdx), origEdus, adjustedParents, labels.slice(sentEduStartIdx, sentEduEndIdx))
      val sentOffset = newChunks.size
      newChunks ++= sentChunks.map(chunk => (sentIdx -> chunk._1) -> (sentIdx -> chunk._2))
      // Offset all chunks by the number of chunks occurring earlier in the document, same with
      newParents ++= sentParents.map(parent => if (parent == -1) -1 else parent + sentOffset)
      newLabels ++= sentLabels
    }
    (newChunks.toSeq, newParents.toSeq, newLabels.toSeq)
  }
  
  def refineEdusInSentence(possibleCompressions: Set[(Int,Int)], edus: Seq[(Int,Int)], parents: Seq[Int], labels: Seq[String]): (Seq[(Int,Int)], Seq[Int], Seq[String]) = {
    require(edus.size == parents.size && edus.size == labels.size, edus.size + " " + parents.size + " " + labels.size)
    // For each EDU, try to refine it and make a little parent structure
//    val newParents = new ArrayBuffer[Int]
//    val newLabels = new ArrayBuffer[String]
    val newGroupEdus = new ArrayBuffer[Seq[(Int,Int)]]
    val newGroupParents = new ArrayBuffer[Seq[Int]]
    val newGroupLabels = new ArrayBuffer[Seq[String]]
    
    var parentOffset = 0
    for (eduIdx <- 0 until edus.size) {
      val edu = edus(eduIdx)
      val possibleCompressionsThisEdu = possibleCompressions.filter(compr => compr._1 >= edu._1 && compr._2 <= edu._2 && compr != edu).map(compr => (compr._1 - edu._1) -> (compr._2 - edu._1))
      if (possibleCompressionsThisEdu.isEmpty) {
        newGroupEdus += Seq(edu)
        newGroupParents += Seq(-1)
        require(eduIdx < labels.size, edus.size + " " + parents.size + " " + labels.size + " " + eduIdx) 
        newGroupLabels += Seq(labels(eduIdx))
      } else {
        val (eduChunks, eduParents, eduLabels) = compressSentence(possibleCompressionsThisEdu, edu._2 - edu._1)
        newGroupEdus += eduChunks.map(eduChunk => (eduChunk._1 + edu._1) -> (eduChunk._2 + edu._1))
        newGroupParents += eduParents 
        newGroupLabels += eduLabels
      }
    }
    require(newGroupEdus.size == parents.size, newGroupEdus.size + " " + parents.size)
    val cumEdus = new ArrayBuffer[Int]
    var currNum = 0
    for (i <- 0 until newGroupEdus.size) {
      require(newGroupParents(i).contains(-1), "No root of the subtree! NGE: " + newGroupEdus + "; NGP: " + newGroupParents + "; NGL: " + newGroupLabels)
      cumEdus += currNum
      currNum += newGroupEdus(i).size
    }
    // Note that we can have multiple heads if you have (0, 6) and (6, 9) fully spanning an EDU.
    val eduHeads = Array.tabulate(newGroupEdus.size)(i => {
      (0 until newGroupEdus(i).size).filter(j => newGroupParents(i)(j) == -1).map(cumEdus(i) + _)
    })
    // We need to update the parents. Within each EDUs sub-parts, we have parents
    val newParents = new ArrayBuffer[Int]
    val newLabels = new ArrayBuffer[String]
    for (i <- 0 until newGroupEdus.size) {
      for (j <- 0 until newGroupEdus(i).size) {
        if (newGroupParents(i)(j) == -1) {
          val parent = parents(i)
          if (parent == -1) {
            newParents += -1
          } else {
            newParents += eduHeads(parent).head
          }
          // When you have multiple heads of an EDU, that means it's fully spanned by possible
          // compressions. Handle this appropriately
          if (eduHeads(i).size > 1) {
            newLabels += "Compression"
          } else {
            newLabels += labels(i)
          }
        } else {
          newParents += cumEdus(i) + newGroupParents(i)(j)
          newLabels += newGroupLabels(i)(j)
        }
      }
    }
    val newEdus = newGroupEdus.flatten.toSeq
    (newEdus, newParents.toSeq, newLabels)
  }
  
  def main(args: Array[String]) {
    LightRunner.initializeOutput(SyntacticCompressor.getClass())
    Logger.logss(", stuff".drop(1).trim)
    // Should be 2, -1, 1, -1
    val cuts = Seq((0, 1), (0, 3), (0, 2), (5, 6))
    println(findComprParents(cuts).toSeq)
    // Should be -1, 3, 0, 2, 0
    println(findComprParents(Seq((0, 6), (0, 1), (0, 3), (0, 2), (5, 6))).toSeq)
    println(compress(Seq(cuts.toSet), Seq(6)))
    println(compress(Seq(cuts.toSet), Seq(7)))
    val cuts2 = Seq((0, 3), (1, 2), (5, 6))
    println(compress(Seq(cuts2.toSet), Seq(7)))
    println(compress(Seq(cuts.toSet, cuts2.toSet), Seq(7, 7)))
    val cuts3 = Seq((0, 6), (6, 9))
    println(compress(Seq(cuts3.toSet), Seq(9)))
    val cuts4 = Seq((0, 6), (6, 9), (0, 9))
    println(compress(Seq(cuts4.toSet), Seq(10)))
    
    // Refining EDUs
    println(refineEdusInSentence(Set((0, 1), (6, 7)), Seq((0, 5), (5, 10)), Seq(-1, 0), Seq("None", "Elaboration")))
    LightRunner.finalizeOutput()
    
  }
}
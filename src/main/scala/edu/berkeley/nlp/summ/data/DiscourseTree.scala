package edu.berkeley.nlp.summ.data

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet

import edu.berkeley.nlp.futile.tokenizer.PTBLineLexer


case class DiscourseTree(val name: String,
                         val rootNode: DiscourseNode) extends Serializable {
  
  // Set parent of each node
  DiscourseTree.setParentsHelper(rootNode)
  
  // Set head of each node
  // This is according to the method described in Hirao et al. (2013),
  // but it doesn't actually lead to sensible heads.
  val leaves = DiscourseTree.getLeaves(rootNode)
  def numLeaves = leaves.size
  def leafWords = leaves.map(_.leafWords.toSeq)
  val leafStatuses = leaves.map(_.label)
  private def setHiraoHeadsHelper(node: DiscourseNode) {
    // TODO: Actually percolate heads, only break ties if two Ns
    var leftmostN = -1
    for (idx <- node.span._1 until node.span._2) {
      if (leftmostN == -1 && leafStatuses(idx) == "Nucleus") {
        leftmostN = idx;
      }
    }
    node.hiraoHead = if (leftmostN == -1) {
      node.span._1
    } else {
      leftmostN
    }
    for (child <- node.children) {
      setHiraoHeadsHelper(child)
    }
  }
  setHiraoHeadsHelper(rootNode)
  DiscourseTree.setRecursiveHeadsHelper(rootNode)
  
  // Determine dependency structure
  // This is the method specified in Hirao et al. but it
  // doesn't seem to do well; you tend to end up with pretty
  // shallow structures and they look a bit weird overall.
  val hiraoParents = Array.tabulate(leafStatuses.size)(i => {
    val leaf = leaves(i)
    if (leafStatuses(i) == DiscourseNode.Nucleus) {
      // Find the nearest dominating S, then assign to
      // the head that S's parent. This is because only
      // S's are in subordinating relations so we need to find
      // one in order to establish the hierarchy.
      var node = leaf.parent
      while (node != null && node.label != DiscourseNode.Satellite) {
        node = node.parent
      }
      if (node == null) {
        -1
      } else {
        node.parent.head
      }
    } else {
      require(leafStatuses(i) == DiscourseNode.Satellite)
      // Find the first parent with a head that's not this
      var node = leaf.parent
      while (node.head == leaf.span._1) {
        node = node.parent
      }
      node.head
    }
  })
  // "Best" parent method, where your depth ends up being the number of
  // Ss between you and the root + 1
  private val advancedDepTree = DiscourseTree.setAdvancedParents(rootNode, leafStatuses.size, false)
  private val advancedParents = advancedDepTree.parents
  private val advancedLabels = advancedDepTree.labels.map(DiscourseTree.getCoarseLabel(_))
  private val apRootIndices = advancedParents.zipWithIndex.filter(_._1 == -1).map(_._2)
  private val advancedParentsOneRoot = Array.tabulate(advancedParents.size)(i => if (apRootIndices.contains(i) && apRootIndices(0) != i) apRootIndices(0) else advancedParents(i))

  private val advancedDepTreeMNLinks = DiscourseTree.setAdvancedParents(rootNode, leafStatuses.size, true)
  private val advancedParentsMNLinks = advancedDepTreeMNLinks.parents
  private val advancedLabelsMNLinks = advancedDepTreeMNLinks.labels.map(DiscourseTree.getCoarseLabel(_))
  
  val parents = advancedParentsOneRoot // Current best
  val parentsMultiRoot = advancedParents
  val labels = advancedLabels
  val childrenMap = DiscourseTree.makeChildrenMap(parents)
  
  
  def getParents(useMultinuclearLinks: Boolean) = {
    if (useMultinuclearLinks) advancedDepTreeMNLinks.parents else advancedParentsOneRoot
  }
  
  def getLabels(useMultinuclearLinks: Boolean) = {
    if (useMultinuclearLinks) advancedLabelsMNLinks else advancedLabels
  }
}

object DiscourseTree {
  
  def getCoarseLabel(label: String) = {
    if (label == null) {
      "root"
    } else if (label.contains("-")) {
      label.substring(0, label.indexOf("-"))
    } else {
      label
    }
  }
  
  def getLeaves(rootNode: DiscourseNode): ArrayBuffer[DiscourseNode] = {
    val leaves = new ArrayBuffer[DiscourseNode]
    getLeavesHelper(rootNode, leaves)
  }
  
  def getLeavesHelper(rootNode: DiscourseNode, leaves: ArrayBuffer[DiscourseNode]): ArrayBuffer[DiscourseNode] = {
    if (rootNode.isLeaf) {
      leaves += rootNode
      leaves
    } else {
      for (child <- rootNode.children) {
        getLeavesHelper(child, leaves)
      }
      leaves
    }
  }
  
  def setParentsHelper(node: DiscourseNode) {
    for (child <- node.children) {
      child.parent = node
      setParentsHelper(child)
    }
  }
  
  def setRecursiveHeadsHelper(node: DiscourseNode): Int = {
    if (node.isLeaf) {
      node.head = node.span._1
      node.head
    } else {
      var parentHeadIdx = -1
      for (child <- node.children) {
        val childHead = setRecursiveHeadsHelper(child)
        if (parentHeadIdx == -1 && child.label == DiscourseNode.Nucleus) {
          parentHeadIdx = childHead
        }
      }
      require(parentHeadIdx != -1)
      node.head = parentHeadIdx
      parentHeadIdx
    }
  }
  
  def setAdvancedParents(node: DiscourseNode, numLeaves: Int, addMultinuclearLinks: Boolean): DiscourseDependencyTree = {
    val depTree = new DiscourseDependencyTree(Array.fill(numLeaves)(-1), new Array[String](numLeaves), new ArrayBuffer[(Int,Int)])
    setAdvancedParentsHelper(node, depTree, addMultinuclearLinks)
    depTree
  }
  
  /**
   * Set parents according to the "advanced" strategy, which by definition
   * produces a tree such that the depth of each node is 1 + the number of Ss
   * between it and the root. This helper method returns the set of unbound
   * nodes at this point in the recursion; ordinarily in parsing this would just
   * be one head, but it can be multiple in the case of N => N N rules.
   */
  def setAdvancedParentsHelper(node: DiscourseNode, depTree: DiscourseDependencyTree, addMultinuclearLinks: Boolean): Seq[Int] = {
    // Leaf node
    if (node.children.size == 0) {
      Seq(node.span._1)
    } else if (node.children.size == 2) {
      ////////////
      // BINARY //
      ////////////
      // Identify the satellite (if it exists) and link up all exposed heads from
      // the satellite to the nucleus. The rel2par of the satellite encodes the relation.
      val leftExposed = setAdvancedParentsHelper(node.children(0), depTree, addMultinuclearLinks)
      val rightExposed = setAdvancedParentsHelper(node.children(1), depTree, addMultinuclearLinks)
      val ruleType = node.children(0).label + " " + node.children(1).label
      // BINUCLEAR
      if (ruleType == DiscourseNode.Nucleus + " " + DiscourseNode.Nucleus) {
        if (addMultinuclearLinks) {
          require(leftExposed.size == 1 && rightExposed.size == 1, "Bad structure!")
          depTree.parents(rightExposed(0)) = leftExposed.head
          // All labels of multinuclear things start with =
          depTree.labels(rightExposed(0)) = "=" + node.children(1).rel2par
          leftExposed
        } else {
          if (node.children(0).rel2par == "Same-Unit" && node.children(1).rel2par == "Same-Unit") {
            // There can be multiple if one Same-Unit contains some coordination
            for (leftIdx <- leftExposed) {
              for (rightIdx <- rightExposed) {
                depTree.sameUnitPairs += leftIdx -> rightIdx
              }
            }
          }
          leftExposed ++ rightExposed
        }
      } else if (ruleType == DiscourseNode.Nucleus + " " + DiscourseNode.Satellite) {
        // Mononuclear, left-headed
        val head = leftExposed.head
//        val head = leftExposed.last // This works a bit worse
        for (rightIdx <- rightExposed) {
          depTree.parents(rightIdx) = head
          depTree.labels(rightIdx) = node.children(1).rel2par
        }
        leftExposed
      } else {
        // Mononuclear, right-headed
        require(ruleType == DiscourseNode.Satellite + " " + DiscourseNode.Nucleus)
        val head = rightExposed.head
        for (leftIdx <- leftExposed) {
          depTree.parents(leftIdx) = head
          depTree.labels(leftIdx) = node.children(0).rel2par
        }
        rightExposed
      }
    } else {
      //////////////////
      // HIGHER ARITY //
      //////////////////
      val allChildrenAreNuclei = !node.children.map(_.label == DiscourseNode.Satellite).reduce(_ || _)
      val oneChildIsNucleus = node.children.map(_.label).filter(_ == DiscourseNode.Nucleus).size == 1
      require(allChildrenAreNuclei || oneChildIsNucleus, "Bad higher-arity: " + node.children.map(_.label).toSeq)
      // Higher-arity, all nuclei. Can be Same-Unit, mostly List
      if (allChildrenAreNuclei) {
        val allChildrenExposedIndices = node.children.map(child => setAdvancedParentsHelper(child, depTree, addMultinuclearLinks))
        // Link up all pairs of exposed indices across the children
        val allExposed = new ArrayBuffer[Int]
        if (addMultinuclearLinks) {
          // Add links in sequence a <- b <- c ... (child points to parent here)
          // There should only be one exposed index in this case
          for (childIdx <- 0 until allChildrenExposedIndices.size) {
            require(allChildrenExposedIndices(childIdx).size == 1)
            if (childIdx > 0) {
              depTree.parents(allChildrenExposedIndices(childIdx).head) = allChildrenExposedIndices(childIdx - 1).head
              // All labels of multinuclear things start with =
              depTree.labels(allChildrenExposedIndices(childIdx).head) = "=" + node.children(childIdx).rel2par
            }
          }
          allExposed += allChildrenExposedIndices(0).head
        } else {
          // Pass all children up
          for (exposedIndices <- allChildrenExposedIndices) {
            allExposed ++= exposedIndices
          }
        }
        allExposed
      } else {
        // Higher-arity, one nucleus. Typically standard relations that simply have arity > 2
        val nucleusIdx = node.children.map(_.label).zipWithIndex.filter(_._1 == DiscourseNode.Nucleus).head._2
        val nucleusExposed = setAdvancedParentsHelper(node.children(nucleusIdx), depTree, addMultinuclearLinks)
        for (i <- 0 until node.children.size) {
          if (i != nucleusIdx) {
            val satelliteExposed = setAdvancedParentsHelper(node.children(i), depTree, addMultinuclearLinks)
//            val nucleusHead = if (i < nucleusIdx) nucleusExposed.head else nucleusExposed.last // This works a bit worse
            val nucleusHead = nucleusExposed.head
            for (satelliteIdx <- satelliteExposed) {
              depTree.parents(satelliteIdx) = nucleusHead
              depTree.labels(satelliteIdx) = node.children(i).rel2par
            }
          }
        }
        nucleusExposed
      }
    }
  }
  
  def makeChildrenMap(parents: Seq[Int]) = {
    val childrenMap = new HashMap[Int,ArrayBuffer[Int]]
    for (i <- 0 until parents.size) {
      childrenMap.put(i, new ArrayBuffer[Int])
    }
    for (i <- 0 until parents.size) {
      if (parents(i) != -1) {
        childrenMap(parents(i)) += i
      }
    }
    childrenMap
  }
  
  def computeDepths(parents: Seq[Int]): Array[Int] = computeDepths(parents, Array.fill(parents.size)(""), false)
  
  def computeDepths(parents: Seq[Int], labels: Seq[String], flattenMultinuclear: Boolean): Array[Int] = {
    val depths = Array.tabulate(parents.size)(i => -1)
    var unassignedDepths = true
    while (unassignedDepths) {
      unassignedDepths = false
      for (i <- 0 until parents.size) {
        if (depths(i) == -1) {
          if (parents(i) == -1) {
            depths(i) = 1
          } else if (depths(parents(i)) != -1) {
            depths(i) = if (flattenMultinuclear && labels(i).startsWith("=")) depths(parents(i)) else depths(parents(i)) + 1
          } else {
            unassignedDepths = true
          }
        }
      }
    }
//    for (i <- 0 until depths.size) {
//      require(depths(i) == computeDepth(parents, labels, flattenMultinuclear, i))
//    }
    depths
  }
  
  def computeDepth(parents: Seq[Int], labels: Seq[String], flattenMultinuclear: Boolean, idx: Int) = {
    var node = idx
    var depth = 0
    // The root of the tree is at depth 1
    while (node != -1) {
      if (!flattenMultinuclear || !labels(node).startsWith("=")) {
        depth += 1
      }
      node = parents(node)
    }
    depth
  }
  
  def computeNumDominated(parents: Seq[Int], idx: Int) = {
    val childrenMap = makeChildrenMap(parents)
    val children = childrenMap(idx)
    var totalChildren = 0
    var newFrontier = new HashSet[Int] ++ children
    var frontier = new HashSet[Int]
    while (!newFrontier.isEmpty) {
      frontier = newFrontier
      newFrontier = new HashSet[Int]
      for (child <- frontier) {
        totalChildren += 1
        newFrontier ++= childrenMap(child)
      }
    }
    totalChildren
  }
}

case class DiscourseDependencyTree(val parents: Array[Int],
                                   val labels: Array[String],
                                   val sameUnitPairs: ArrayBuffer[(Int,Int)]) {
}

case class DiscourseNode(val label: String,
                         val rel2par: String,
                         val span: (Int,Int),
                         val leafText: String,
                         val children: ArrayBuffer[DiscourseNode]) extends Serializable {
  var head: Int = -1
  var hiraoHead: Int = -1
  var parent: DiscourseNode = null
  
  // N.B. If anything changes here, should rerun EDUAligner and make sure things aren't worse
  val leafTextPreTok = leafText.replace("<P>", "")
  val leafWordsWhitespace = leafTextPreTok.split("\\s+").filter(_ != "<P>")
  // Adding the period fixes a bug where "buy-outs" is treated differently sentence-internally than it is
  // when it ends an utterance; generally this makes the tokenizer more consistent on fragments
  val leafWordsPTBLL = if (leafTextPreTok.split("\\s+").last.contains("-")) {
    new PTBLineLexer().tokenize(leafTextPreTok + " .").toArray(Array[String]()).dropRight(1)
  } else {
    new PTBLineLexer().tokenize(leafTextPreTok).toArray(Array[String]()) 
  }
  // There are still some spaces in some tokens; get rid of these
  val leafWordsPTB = if (leafTextPreTok != "") {
    leafWordsPTBLL.flatMap(_.split("\\s+")).filter(_ != "<P>")
  } else {
    Array[String]()
  }
  
//  def leafWords = leafWordsWhitespace
  def leafWords = leafWordsPTB
//  def leafWords = leafWordsPTBLL
  
  def isLeaf = span._2 - span._1 == 1
  
}

object DiscourseNode {
  val Nucleus = "Nucleus"
  val Satellite = "Satellite"
  val Root = "Root"
}

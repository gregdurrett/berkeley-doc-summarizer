package edu.berkeley.nlp.summ.compression;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.entity.lang.ModCollinsHeadFinder;
import edu.berkeley.nlp.futile.syntax.Tree;
import edu.berkeley.nlp.summ.Pair;


public class SentenceCompressor {
  
  public static final ModCollinsHeadFinder headFinder = new ModCollinsHeadFinder();
  public static final Set<String> daysOfWeek = new HashSet<String>();
  public static final Set<String> dayOfWeekNPWords = new HashSet<String>();
  public static final Set<String> attributionVPHeads = new HashSet<String>();
  public static final Set<String> attributionPPHeads = new HashSet<String>();
  public static final Set<String> parentNPdeletablePPHeads = new HashSet<String>();
  public static final Set<String> phraseTypesDeletableInCoord = new HashSet<String>();
  public static final Set<String> nonDeletableCoordinators = new HashSet<String>();
  public static final Set<String> dayOfWeekPhrasePPHeads = new HashSet<String>();
  public static final Set<String> initialSBARDeletableHeads = new HashSet<String>();
  static {
    daysOfWeek.add("monday");
    daysOfWeek.add("tuesday");
    daysOfWeek.add("wednesday");
    daysOfWeek.add("thursday");
    daysOfWeek.add("friday");
    daysOfWeek.add("saturday");
    daysOfWeek.add("sunday");
    daysOfWeek.add("today");
    daysOfWeek.add("yesterday");
    daysOfWeek.add("tomorrow");

    dayOfWeekNPWords.add("monday");
    dayOfWeekNPWords.add("tuesday");
    dayOfWeekNPWords.add("wednesday");
    dayOfWeekNPWords.add("thursday");
    dayOfWeekNPWords.add("friday");
    dayOfWeekNPWords.add("saturday");
    dayOfWeekNPWords.add("sunday");
    dayOfWeekNPWords.add("today");
    dayOfWeekNPWords.add("yesterday");
    dayOfWeekNPWords.add("tomorrow");
    dayOfWeekNPWords.add("last");
    dayOfWeekNPWords.add("early");
    dayOfWeekNPWords.add("late");
    dayOfWeekNPWords.add("morning");
    dayOfWeekNPWords.add("evening");
    dayOfWeekNPWords.add("night");
    dayOfWeekNPWords.add("afternoon");
    
    dayOfWeekPhrasePPHeads.add("on");
    dayOfWeekPhrasePPHeads.add("before");
    dayOfWeekPhrasePPHeads.add("since");
    dayOfWeekPhrasePPHeads.add("after");

    attributionVPHeads.add("confirmed");
    attributionVPHeads.add("reported");
    attributionVPHeads.add("said");
    attributionVPHeads.add("told");
    attributionVPHeads.add("has");

    attributionPPHeads.add("according");
    
    parentNPdeletablePPHeads.add("in");
    parentNPdeletablePPHeads.add("on");
    parentNPdeletablePPHeads.add("after");
    parentNPdeletablePPHeads.add("before");
    parentNPdeletablePPHeads.add("during");
    parentNPdeletablePPHeads.add("since");
    
    initialSBARDeletableHeads.add("where");
    initialSBARDeletableHeads.add("while");
    initialSBARDeletableHeads.add("though");
    initialSBARDeletableHeads.add("although");
    initialSBARDeletableHeads.add("when");
    initialSBARDeletableHeads.add("since");
    initialSBARDeletableHeads.add("because");
    initialSBARDeletableHeads.add("as");
    
    phraseTypesDeletableInCoord.add("S");
    phraseTypesDeletableInCoord.add("NP");
    phraseTypesDeletableInCoord.add("VP");
    phraseTypesDeletableInCoord.add("SBAR");
    
    nonDeletableCoordinators.add("nor");
  }

  public static IdentityHashMap<Tree<String>,Tree<String>> getParentTrees(Tree<String> inputTree) {
    IdentityHashMap<Tree<String>,Tree<String>> parentsMap = new IdentityHashMap<Tree<String>,Tree<String>>();
    getParentTreesHelper(inputTree, parentsMap);
    return parentsMap;
  }

  private static void getParentTreesHelper(Tree<String> inputTree, IdentityHashMap<Tree<String>,Tree<String>> parentsMap) {
    if (!inputTree.isLeaf()) {
      List<Tree<String>> children = inputTree.getChildren();
      for (Tree<String> child : children) {
        parentsMap.put(child, inputTree);
        getParentTreesHelper(child, parentsMap);
      }
    }
  }
  
//  public static List<List<String>> getPossibleCompressions(Tree<String> inputTree) {
//    List<Pair<Integer,Integer>> spansToCut = new ArrayList<Pair<Integer,Integer>>();
//    
//  }
  
  public static List<Pair<String,Double>> getCutFeatures(Tree<String> nodeTree, int startIdx, int endIdx, IdentityHashMap<Tree<String>,Tree<String>> parentsMap) {
    // Usually only true at the root
    if (!parentsMap.containsKey(nodeTree)) {
      return new ArrayList<Pair<String,Double>>();
//      return false;
    }
    String nodeLabel = nodeTree.getLabel();
    Tree<String> parentTree = parentsMap.get(nodeTree);
    String parentLabel = parentTree.getLabel();
//    List<Integer> leftSiblings = new ArrayList<Integer>();
//    List<Integer> rightSiblings = new ArrayList<Integer>();
//    if (p != -1) {
//      parentTree = getSubtreeByNode()[p];
//      parentLabel = parentTree.getLabel();
//      int[] siblingNodes = getNodeChildren()[p];
//      int ni=-1;
//      for (int si=0; si<siblingNodes.length; ++si) if (siblingNodes[si] == n) {ni = si; break;}
//      for (int si=ni-1; si>=0; --si) leftSiblings.add(siblingNodes[si]);
//      for (int si=ni+1; si<siblingNodes.length; ++si) rightSiblings.add(siblingNodes[si]);
//    }

    List<Tree<String>> leftSiblingTrees = new ArrayList<Tree<String>>();
    List<Tree<String>> rightSiblingTrees = new ArrayList<Tree<String>>();
    List<Tree<String>> siblingTrees = new ArrayList<Tree<String>>();
    List<String> leftSiblingLabels = new ArrayList<String>();
    List<String> rightSiblingLabels = new ArrayList<String>();
    List<String> siblingLabels = new ArrayList<String>();
    
    List<Tree<String>> siblings = parentTree.getChildren();
    boolean isLeft = true;
    for (int i = 0; i < siblings.size(); i++) {
      if (siblings.get(i) == nodeTree) {
        isLeft = false;
      } else {
        if (isLeft) {
          leftSiblingTrees.add(siblings.get(i));
          leftSiblingLabels.add(siblings.get(i).getLabel());
        } else {
          rightSiblingTrees.add(siblings.get(i));
          rightSiblingLabels.add(siblings.get(i).getLabel());
        }
        siblingTrees.add(siblings.get(i));
        siblingLabels.add(siblings.get(i).getLabel());
      }
    }
    
    Tree<String> headTree = getHeadTree(nodeTree);
    String headWord = getHeadWord(nodeTree);
    
    Tree<String> parentHeadTree = getHeadTree(parentTree);
//    String parentHeadLabel = null;
//    String parentHeadWord = null;
//    String parentHeadPOS = null;
//    if (p != -1) {
//      parentHeadTree = getHeadTree(parentTree);
//      parentHeadLabel = (parentHeadTree != null ? parentHeadTree.getLabel() : null);
//      parentHeadWord = getHeadWord(parentTree);
//      parentHeadPOS = getHeadPOS(parentTree);
//    }
    
    int parentHeadChildIndex = -1;
    int parentNodeChildIndex = -1;
    for (int sib=0; sib<parentTree.getChildren().size(); ++sib) {
      if (parentTree.getChild(sib) == parentHeadTree) {
        parentHeadChildIndex = sib;
      }
      if (parentTree.getChild(sib) == nodeTree) {
        parentNodeChildIndex = sib;
      }
    }
    int distanceToParentHead = parentHeadChildIndex - parentNodeChildIndex;
    
//    boolean startsSentence = true;
//    Tree<String> node = nodeTree;
//    while (startsSentence && parentsMap.containsKey(node)) {
//      // Needs to be on the left edge all the way down
//      if (parentsMap.get(node).getChildren().indexOf(node) != 0) {
//        startsSentence = false;
//      }
//      node = parentsMap.get(node);
//    }
    
    boolean isCoordinated = false;
    boolean firstChildIsCC = false;
    String coordinatingCCHead = null;
    if (!nodeTree.isLeaf() && nodeTree.getChild(0).getLabel().equals("CC")) {
      isCoordinated = nodeLabel.equals(parentLabel) && !leftSiblingLabels.isEmpty() && leftSiblingLabels.get(0).equals(nodeLabel);
      coordinatingCCHead = getHeadWord(nodeTree.getChild(0));
      firstChildIsCC = true;
    } else {
      if (!rightSiblingLabels.isEmpty()) {
        for (Tree<String> sibling : rightSiblingTrees) {
          if (sibling.getLabel().equals(nodeLabel) && !sibling.isLeaf() && sibling.getChild(0).getLabel().equals("CC")) {
            coordinatingCCHead = getHeadWord(sibling.getChild(0));
            isCoordinated = true;
            break;
          }
          if (!sibling.getLabel().equals(nodeLabel)) {
            break;
          }
        }
      }
      isCoordinated = isCoordinated && nodeLabel.equals(parentLabel);
    }
    
    int numberOfSiblingsOfSameType = 0;
    for (String label : siblingLabels) if (nodeLabel.equals(label)) numberOfSiblingsOfSameType++;
    
    //////////////////////////////////////////////////////////////////////////////////////
    
    List<Pair<String,Double>> features = new ArrayList<Pair<String,Double>>();

    // Final attribution VP and NP deletion

    if (nodeLabel.equals("SATTR")) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("SATTR"), 1.0));
      features.add(Pair.makePair(String.format("SATTR_Head%s", headWord), 1.0));
    } 


    // Attribution PP deletion

    else if (nodeLabel.equals("PP") 
        && attributionPPHeads.contains(headWord)) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_attr"), 1.0));
      features.add(Pair.makePair(String.format("PP_attr_Head%s", headWord), 1.0));
    }


    // Day of week PP deletion

    else if (nodeLabel.equals("PP") 
        && dayOfWeekPhrasePPHeads.contains(headWord)
        && nodeTree.getChildren().size() >= 2
        && nodeTree.getChild(1).getLabel().equals("NP") 
        && isDayOfWeekNP(nodeTree.getChild(1))) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_dayOfWeek"), 1.0));
      features.add(Pair.makePair(String.format("PP_dayOfWeek_Head%s", headWord), 1.0));
    }


    // Day of week NP deletion

    else if (nodeLabel.equals("NP") 
        && isDayOfWeekNP(nodeTree)
        && (leftSiblingLabels.isEmpty() 
            || !leftSiblingLabels.get(0).equals("VP"))
            && !parentLabel.equals("PP") 
            && (!parentsMap.containsKey(parentTree) 
            || !parentsMap.get(parentTree).getLabel().equals("PP"))) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("NP_dayOfWeek"), 1.0));
    }


    // Coordinated phrase deletion

    else if (isCoordinated 
        && phraseTypesDeletableInCoord.contains(nodeLabel) 
        && firstChildIsCC 
        && numberOfSiblingsOfSameType == 1
        && !nonDeletableCoordinators.contains(coordinatingCCHead)) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("Coord%s", nodeLabel), 1.0));
      features.add(Pair.makePair(String.format("Coord%s_CCHead%s", nodeLabel, coordinatingCCHead), 1.0));
    }


    // Sentence-initial connective phrase deletion

    else if (nodeLabel.equals("CC") 
        && parentLabel.equals("S") 
        && startIdx == 0) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("CC_initial"), 1.0));
      features.add(Pair.makePair(String.format("CC_initial_Head%s", headWord), 1.0));
    }

    else if (nodeLabel.equals("ADVP") 
        && parentLabel.equals("S") 
        && distanceToParentHead >= 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("ADVP_initial"), 1.0));
      features.add(Pair.makePair(String.format("ADVP_initial_Head%s", headWord), 1.0));
    }

    else if (nodeLabel.equals("SBAR") 
        && initialSBARDeletableHeads.contains(headWord)
        && parentLabel.equals("S") 
        && nodeTree.getYield().size() <= 6
        && distanceToParentHead >= 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("SBAR_initial"), 1.0));
      features.add(Pair.makePair(String.format("SBAR_initial_Head%s", headWord), 1.0));
    }

    else if (nodeLabel.equals("PP") 
        && parentLabel.equals("S") 
        && distanceToParentHead >= 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_initial"), 1.0));
      features.add(Pair.makePair(String.format("PP_initial_Head%s", headWord), 1.0));
    }


    // SBAR deletion

    else if (nodeLabel.equals("SBAR") 
        && parentLabel.equals("NP")
        && nodeTree.getYield().size() <= 6
        && headTree.getLabel().startsWith("WH")) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("SBAR"), 1.0));
      features.add(Pair.makePair(String.format("SBAR_Head%s", headWord), 1.0));
    }

    ////////////////////////
    // TODO: Maybe use this, but may need it to be locked down
    else if (nodeLabel.equals("PP") 
        && (parentLabel.equals("NP"))
        && nodeTree.getYield().size() <= 6
        && Math.abs(distanceToParentHead) >= 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_Far"), 1.0));
      features.add(Pair.makePair(String.format("PP_Far_Parent%s", parentLabel), 1.0));
      features.add(Pair.makePair(String.format("PP_Far_Head%s", headWord), 1.0));
      features.add(Pair.makePair(String.format("PP_Far_Parent%s_Head%s", parentLabel, headWord), 1.0));
    }

    else if (nodeLabel.equals("PP") 
        && (parentLabel.equals("NP"))
        && !headWord.equals("of")
        && !headWord.equals("for")
        && nodeTree.getYield().size() <= 6
        && Math.abs(distanceToParentHead) < 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_Close"), 1.0));
      features.add(Pair.makePair(String.format("PP_Close_Parent%s", parentLabel), 1.0));
      features.add(Pair.makePair(String.format("PP_Close_Head%s", headWord), 1.0));
      features.add(Pair.makePair(String.format("PP_Close_Parent%s_Head%s", parentLabel, headWord), 1.0));
    }

    else if (nodeLabel.equals("PP") 
        && (parentLabel.equals("VP"))
        && nodeTree.getYield().size() <= 6
        && !parentHeadTree.isPreTerminal()
        && !headWord.equals("to")
        && Math.abs(distanceToParentHead) >= 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_Far"), 1.0));
      features.add(Pair.makePair(String.format("PP_Far_Parent%s", parentLabel), 1.0));
      features.add(Pair.makePair(String.format("PP_Far_Head%s", headWord), 1.0));
      features.add(Pair.makePair(String.format("PP_Far_Parent%s_Head%s", parentLabel, headWord), 1.0));
    }

    else if (nodeLabel.equals("PP") 
        && (parentLabel.equals("VP"))
        && nodeTree.getYield().size() <= 6
        && !parentHeadTree.isPreTerminal()
        && !headWord.equals("to")
        && Math.abs(distanceToParentHead) < 2) {
      features.add(Pair.makePair(String.format("CutBias"), 1.0));
      features.add(Pair.makePair(String.format("PP_Close"), 1.0));
      features.add(Pair.makePair(String.format("PP_Close_Parent%s", parentLabel), 1.0));
      features.add(Pair.makePair(String.format("PP_Close_Head%s", headWord), 1.0));
      features.add(Pair.makePair(String.format("PP_Close_Parent%s_Head%s", parentLabel, headWord), 1.0));
    }

    ////////////////////////
    
    return features;
    
  }
  
  

  public static boolean isDayOfWeekNP(Tree<String> tree) {
    boolean isDayOfWeekNP = true;
    for (String word : tree.getYield()) {
      if (!dayOfWeekNPWords.contains(word.toLowerCase())) {
        isDayOfWeekNP = false;
      }
    }
    return isDayOfWeekNP;
  }

  public static String getHeadWord(Tree<String> tree) {
    Tree<String> headWordTree = tree;
    while (headWordTree != null && !headWordTree.isLeaf()) {
      headWordTree = headFinder.determineHead(headWordTree);
    }
    if (headWordTree == null) return null;
    return headWordTree.getLabel().toLowerCase();
  }
  
  public static String getHeadPOS(Tree<String> tree) {
    if (tree.isLeaf()) return null;
    Tree<String> headPOSTree = tree;
    while (headPOSTree != null && !headPOSTree.isPreTerminal()) {
      headPOSTree = headFinder.determineHead(headPOSTree);
    }
    if (headPOSTree == null) return null;
    return headPOSTree.getLabel();
  }
  
  public static Tree<String> getHeadTree(Tree<String> tree) {
    return headFinder.determineHead(tree);
  }
}

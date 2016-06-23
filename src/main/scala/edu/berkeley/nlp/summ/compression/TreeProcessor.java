package edu.berkeley.nlp.summ.compression;


import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.futile.syntax.Tree;

public class TreeProcessor {
	
	public static Tree<String> processTree(Tree<String> tree) {
		tree = tree.clone();
		bundleFinalAttributionClause(tree);
		bundleCCAndCoordinatedPhrase(tree);
//		displayTree(tree);
		return tree;
	}
	
	public static void bundleFinalAttributionClause(Tree<String> tree) {
		if (tree.isLeaf()) return;
		for (Tree<String> child : tree.getChildren()) bundleFinalAttributionClause(child);
		
		int numChildren = tree.getChildren().size();
		if ((tree.getLabel().equals("S") || tree.getLabel().equals("SINV")) && numChildren >= 3 && tree.getChild(numChildren-3).getLabel().equals("S") && tree.getChild(numChildren-2).getLabel().equals("NP") && tree.getChild(numChildren-1).getLabel().equals("VP")) {
			Tree<String> NPTree = tree.getChild(numChildren-2);
			Tree<String> VPTree = tree.getChild(numChildren-1);
			List<String> VPChildLabels = new ArrayList<String>();
			for (Tree<String> child : VPTree.getChildren()) {
				VPChildLabels.add(child.getLabel());
			}
			String VPHeadWord = SentenceCompressor.getHeadWord(VPTree);
			
			Tree<String> NPTreeInObjectPosition = null;
			Tree<String> SorSBARTreeInObjectPosition = null;
			if (VPTree.getChildren().size() >= 2) {
				for (int c=1; c<VPTree.getChildren().size(); ++c) {
					if (VPTree.getChild(c).getLabel().equals("NP")) {
						NPTreeInObjectPosition = VPTree.getChild(c);
					}
					if (VPTree.getChild(c).getLabel().equals("S") || VPTree.getChild(c).getLabel().equals("SBAR")) {
						SorSBARTreeInObjectPosition = VPTree.getChild(c);
					}
				}
			}
			
			if ((SorSBARTreeInObjectPosition == null || SentenceCompressor.getHeadPOS(SorSBARTreeInObjectPosition).equals("IN")) 
				&& (NPTreeInObjectPosition == null || SentenceCompressor.isDayOfWeekNP(NPTreeInObjectPosition) || VPHeadWord.equals("told"))) {

				if (SentenceCompressor.attributionVPHeads.contains(VPHeadWord)) {
					Tree<String> newChild = new Tree<String>("SATTR", new ArrayList<Tree<String>>());
					newChild.getChildren().add(NPTree);
					newChild.getChildren().add(VPTree);
					tree.getChildren().remove(numChildren-1);
					tree.getChildren().remove(numChildren-2);
					tree.getChildren().add(newChild);
				}
			}
		}
		if ((tree.getLabel().equals("S") || tree.getLabel().equals("SINV")) && numChildren >= 3 && tree.getChild(numChildren-3).getLabel().equals("S") && tree.getChild(numChildren-2).getLabel().equals("VP") && tree.getChild(numChildren-1).getLabel().equals("NP")) {
			Tree<String> NPTree = tree.getChild(numChildren-1);
			Tree<String> VPTree = tree.getChild(numChildren-2);
			List<String> VPChildLabels = new ArrayList<String>();
			for (Tree<String> child : VPTree.getChildren()) {
				VPChildLabels.add(child.getLabel());
			}
			String VPHeadWord = SentenceCompressor.getHeadWord(VPTree);
			
			if (VPTree.getYield().size() == 1) {
				if (SentenceCompressor.attributionVPHeads.contains(VPHeadWord)) {
					Tree<String> newChild = new Tree<String>("SATTR", new ArrayList<Tree<String>>());
					newChild.getChildren().add(VPTree);
					newChild.getChildren().add(NPTree);
					tree.getChildren().remove(numChildren-1);
					tree.getChildren().remove(numChildren-2);
					tree.getChildren().add(newChild);
				}
			}
		}
	}
	
	public static void bundleCCAndCoordinatedPhrase(Tree<String> tree) {
		if (tree.isLeaf()) return;
		for (Tree<String> child : tree.getChildren()) bundleCCAndCoordinatedPhrase(child);
		
		List<Tree<String>> oldChildren = tree.getChildren();
		List<Tree<String>> newChildren = new ArrayList<Tree<String>>();
		for (int c=0; c<oldChildren.size(); ++c) {
			Tree<String> child = oldChildren.get(c);
			if (child.getLabel().equals("CC") && c < oldChildren.size()-1 && oldChildren.get(c+1).getLabel().equals(tree.getLabel())) {
				oldChildren.get(c+1).getChildren().add(0, child);
			} else {
				newChildren.add(child);
			}
		}
		tree.setChildren(newChildren);
	}
}

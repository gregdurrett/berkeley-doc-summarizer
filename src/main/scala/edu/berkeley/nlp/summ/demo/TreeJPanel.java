package edu.berkeley.nlp.summ.demo;

import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;

import edu.berkeley.nlp.futile.fig.basic.Pair;
import edu.berkeley.nlp.futile.syntax.Tree;

/**
 * Class for displaying a Tree.
 * 
 * @author Dan Klein
 */

public class TreeJPanel extends JPanel {
  
  private static final long serialVersionUID = 1L;

  static double sisterSkip = 2.5;
  static double parentSkip = 0.5;
  static double belowLineSkip = 0.1;
  static double aboveLineSkip = 0.1;
  
  static boolean drawTrianglesAtBottom = true;

  public static String nodeToString(Tree<String> t) {
    if (t == null) {
      return " ";
    }
    Object l = t.getLabel();
    if (l == null) {
      return " ";
    }
    String str = (String) l;
    if (str == null) {
      return " ";
    }
    return str;
  }

  static class WidthResult {
    double width = 0.0;
    double nodeTab = 0.0;
    double nodeCenter = 0.0;
    double childTab = 0.0;
  }

  public static WidthResult widthResult(Tree<String> tree, FontMetrics fM) {
    WidthResult wr = new WidthResult();
    if (tree == null) {
      wr.width = 0.0;
      wr.nodeTab = 0.0;
      wr.nodeCenter = 0.0;
      wr.childTab = 0.0;
      return wr;
    }
    double local = fM.stringWidth(nodeToString(tree));
    if (tree.isLeaf()) {
      wr.width = local;
      wr.nodeTab = 0.0;
      wr.nodeCenter = local / 2.0;
      wr.childTab = 0.0;
      return wr;
    }
    double sub = 0.0;
    double nodeCenter = 0.0;
    double childTab = 0.0;
    for (int i = 0; i < tree.getChildren().size(); i++) {
      WidthResult subWR = widthResult(tree.getChildren()
          .get(i), fM);
      if (i == 0) {
        nodeCenter += (sub + subWR.nodeCenter) / 2.0;
      }
      if (i == tree.getChildren().size() - 1) {
        nodeCenter += (sub + subWR.nodeCenter) / 2.0;
      }
      sub += subWR.width;
      if (i < tree.getChildren().size() - 1) {
        sub += sisterSkip * fM.stringWidth(" ");
      }
    }
    double localLeft = local / 2.0;
    double subLeft = nodeCenter;
    double totalLeft = Math.max(localLeft, subLeft);
    double localRight = local / 2.0;
    double subRight = sub - nodeCenter;
    double totalRight = Math.max(localRight, subRight);
    wr.width = totalLeft + totalRight;
    wr.childTab = totalLeft - subLeft;
    wr.nodeTab = totalLeft - localLeft;
    wr.nodeCenter = nodeCenter + wr.childTab;
    return wr;
  }
  
  /**
   * GREG ADDED
   */
  public static boolean isContained(int spanStart, int spanEnd, List<Pair<Integer,Integer>> spans) {
    for (Pair<Integer,Integer> span : spans) {
      if (span.getFirst().intValue() <= spanStart && spanEnd <= span.getSecond().intValue()) {
        return true;
      }
    }
    return false;
  }
  

  public static List<Integer> getContainedIndices(int spanStart, int spanEnd, List<Pair<Integer,Integer>> subSpans) {
    List<Integer> containedIndices = new ArrayList<Integer>();
    for (int i = 0; i < subSpans.size(); i++) {
      Pair<Integer,Integer> span = subSpans.get(i);
      if (spanStart <= span.getFirst().intValue() && span.getSecond().intValue() <= spanEnd) {
        containedIndices.add(i);
      }
    }
    return containedIndices;
  }
  
  /**
   * GREG'S VERSION
   */
  public static Pair<Double,Double> paintTreeModified(Tree<String> t, Point2D start, Graphics2D g2, FontMetrics fM, int charOffset, Color color,
                                         List<Pair<Integer,Integer>> compressedSpans, List<Pair<Integer,Integer>> pronReplacedSpans, List<String> pronReplacements) {
    g2.setColor(color);
    String nodeStr = nodeToString(t);
    double nodeWidth = fM.stringWidth(nodeStr);
    double nodeHeight = fM.getHeight();
    double nodeAscent = fM.getAscent();
    WidthResult wr = widthResult(t, fM);
    double treeWidth = wr.width;
    double nodeTab = wr.nodeTab;
    double childTab = wr.childTab;
    double nodeCenter = wr.nodeCenter;
    g2.drawString(nodeStr, (float) (nodeTab + start.getX()), (float) (start.getY() + nodeAscent));
    if (t.isLeaf()) {
      return Pair.makePair(new Double(nodeWidth), new Double(nodeHeight));
    }
    double layerMultiplier = (1.0 + belowLineSkip + aboveLineSkip + parentSkip);
    double layerHeight = nodeHeight * layerMultiplier;
    double childStartX = start.getX() + childTab;
    double childStartY = start.getY() + layerHeight;
    double lineStartX = start.getX() + nodeCenter;
    double lineStartY = start.getY() + nodeHeight * (1.0 + belowLineSkip);
    double lineEndY = lineStartY + nodeHeight * parentSkip;
    // recursively draw children
    int currCharOffset = charOffset;
    double maxChildHeight = 0;
    for (int i = 0; i < t.getChildren().size(); i++) {
      Tree<String> child = t.getChildren().get(i);
      int childSize = 0;
      for (String leaf : child.getYield()) {
        childSize += leaf.length() + 1;
      }
      boolean isChildCompressed = isContained(currCharOffset, currCharOffset + childSize - 1, compressedSpans);
      Color childColor = (isChildCompressed ? Color.gray : child.isLeaf() ? SummarizerDemo.alignmentColor() : color);
      Pair<Double,Double> childWidthHeight = paintTreeModified(child, new Point2D.Double(childStartX, childStartY), g2, fM, currCharOffset, childColor,
                                        compressedSpans, pronReplacedSpans, pronReplacements);
      double cWidth = childWidthHeight.getFirst().doubleValue();
      maxChildHeight = Math.max(maxChildHeight, childWidthHeight.getSecond().doubleValue());
      // draw connectors
      wr = widthResult(child, fM);
      g2.setColor((isChildCompressed ? Color.gray : color));
      if (drawTrianglesAtBottom && child.isLeaf()) {
        double triangleStartX = childStartX;
        double triangleEndX = childStartX + wr.width;
        int[] xs = {(int)lineStartX, (int)triangleStartX, (int)triangleEndX, (int)lineStartX};
        int[] ys = {(int)lineStartY, (int)lineEndY, (int)lineEndY, (int)lineStartY};
        g2.drawPolyline(xs, ys, 4);
      } else {
        double lineEndX = childStartX + wr.nodeCenter;
        g2.draw(new Line2D.Double(lineStartX, lineStartY, lineEndX,
            lineEndY));
      }
      childStartX += cWidth;
      if (i < t.getChildren().size() - 1) {
        childStartX += sisterSkip * fM.stringWidth(" ");
      }
      currCharOffset += childSize;
    }
    return Pair.makePair(new Double(treeWidth), new Double(lineEndY - start.getY() + maxChildHeight));
  }
}
package mstparser;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.futile.fig.basic.Pair;

public class KsummKBestParseForest {

  public static int rootType;

  public ParseForestItem[][][][][] chart;

  private int start, end;

  private int K;

  public KsummKBestParseForest(int start, int end, int K) {
    this.K = K;
    chart = new ParseForestItem[end + 1][end + 1][2][2][K];
    this.start = start;
    this.end = end;
  }

  public boolean add(int s, int type, int dir, double score, FeatureVector fv) {

    boolean added = false;

    if (chart[s][s][dir][0][0] == null) {
      for (int i = 0; i < K; i++)
        chart[s][s][dir][0][i] = new ParseForestItem(s, type, dir, Double.NEGATIVE_INFINITY, null);
    }

    if (chart[s][s][dir][0][K - 1].prob > score)
      return false;

    for (int i = 0; i < K; i++) {
      if (chart[s][s][dir][0][i].prob < score) {
        ParseForestItem tmp = chart[s][s][dir][0][i];
        chart[s][s][dir][0][i] = new ParseForestItem(s, type, dir, score, fv);
        for (int j = i + 1; j < K && tmp.prob != Double.NEGATIVE_INFINITY; j++) {
          ParseForestItem tmp1 = chart[s][s][dir][0][j];
          chart[s][s][dir][0][j] = tmp;
          tmp = tmp1;
        }
        added = true;
        break;
      }
    }

    return added;
  }

  public boolean add(int s, int r, int t, int type, int dir, int comp, double score,
          FeatureVector fv, ParseForestItem p1, ParseForestItem p2) {

    boolean added = false;

    if (chart[s][t][dir][comp][0] == null) {
      for (int i = 0; i < K; i++)
        chart[s][t][dir][comp][i] = new ParseForestItem(s, r, t, type, dir, comp,
                Double.NEGATIVE_INFINITY, null, null, null);
    }

    if (chart[s][t][dir][comp][K - 1].prob > score)
      return false;

    for (int i = 0; i < K; i++) {
      if (chart[s][t][dir][comp][i].prob < score) {
        ParseForestItem tmp = chart[s][t][dir][comp][i];
        chart[s][t][dir][comp][i] = new ParseForestItem(s, r, t, type, dir, comp, score, fv, p1, p2);
        for (int j = i + 1; j < K && tmp.prob != Double.NEGATIVE_INFINITY; j++) {
          ParseForestItem tmp1 = chart[s][t][dir][comp][j];
          chart[s][t][dir][comp][j] = tmp;
          tmp = tmp1;
        }
        added = true;
        break;
      }

    }

    return added;

  }

  public double getProb(int s, int t, int dir, int comp) {
    return getProb(s, t, dir, comp, 0);
  }

  public double getProb(int s, int t, int dir, int comp, int i) {
    if (chart[s][t][dir][comp][i] != null)
      return chart[s][t][dir][comp][i].prob;
    return Double.NEGATIVE_INFINITY;
  }

  public double[] getProbs(int s, int t, int dir, int comp) {
    double[] result = new double[K];
    for (int i = 0; i < K; i++)
      result[i] = chart[s][t][dir][comp][i] != null ? chart[s][t][dir][comp][i].prob
              : Double.NEGATIVE_INFINITY;
    return result;
  }

  public ParseForestItem getItem(int s, int t, int dir, int comp) {
    return getItem(s, t, dir, comp, 0);
  }

  public ParseForestItem getItem(int s, int t, int dir, int comp, int k) {
    if (chart[s][t][dir][comp][k] != null)
      return chart[s][t][dir][comp][k];
    return null;
  }

  public ParseForestItem[] getItems(int s, int t, int dir, int comp) {
    if (chart[s][t][dir][comp][0] != null)
      return chart[s][t][dir][comp];
    return null;
  }

  public Object[] getBestParse() {
    Object[] d = new Object[2];
    d[0] = getFeatureVector(chart[0][end][0][0][0]);
    d[1] = getDepString(chart[0][end][0][0][0]);
    return d;
  }
  
  public List<Pair<Integer,Integer>> getBestParseDepLinks() {
    return getDepLinks(chart[0][end][0][0][0]);
  }

  public Object[][] getBestParses() {
    Object[][] d = new Object[K][2];
    for (int k = 0; k < K; k++) {
      if (chart[0][end][0][0][k].prob != Double.NEGATIVE_INFINITY) {
        d[k][0] = getFeatureVector(chart[0][end][0][0][k]);
        d[k][1] = getDepString(chart[0][end][0][0][k]);
      } else {
        d[k][0] = null;
        d[k][1] = null;
      }
    }
    return d;
  }

  public FeatureVector getFeatureVector(ParseForestItem pfi) {
    if (pfi.left == null)
      return pfi.fv;

    return cat(pfi.fv, cat(getFeatureVector(pfi.left), getFeatureVector(pfi.right)));
  }

  public String getDepString(ParseForestItem pfi) {
    if (pfi.left == null)
      return "";

    if (pfi.comp == 0) {
      return (getDepString(pfi.left) + " " + getDepString(pfi.right)).trim();
    } else if (pfi.dir == 0) {
      return ((getDepString(pfi.left) + " " + getDepString(pfi.right)).trim() + " " + pfi.s + "|"
              + pfi.t + ":" + pfi.type).trim();
    } else {
      return (pfi.t + "|" + pfi.s + ":" + pfi.type + " " + (getDepString(pfi.left) + " " + getDepString(pfi.right))
              .trim()).trim();
    }
  }

  public List<Pair<Integer,Integer>> getDepLinks(ParseForestItem pfi) {
    List<Pair<Integer,Integer>> links = new ArrayList<Pair<Integer,Integer>>();
    if (pfi.left == null)
      return links;

    links.addAll(getDepLinks(pfi.left));
    links.addAll(getDepLinks(pfi.right));
    if (pfi.comp == 0) {
      // Do nothing
    } else if (pfi.dir == 0) {
      links.add(new Pair<Integer,Integer>(pfi.s, pfi.t));
    } else {
      links.add(new Pair<Integer,Integer>(pfi.t, pfi.s));
    }
    return links;
  }

  public FeatureVector cat(FeatureVector fv1, FeatureVector fv2) {
    return fv1.cat(fv2);
  }

  // returns pairs of indeces and -1,-1 if < K pairs
  public int[][] getKBestPairs(ParseForestItem[] items1, ParseForestItem[] items2) {
    // in this case K = items1.length

    boolean[][] beenPushed = new boolean[K][K];

    int[][] result = new int[K][2];
    for (int i = 0; i < K; i++) {
      result[i][0] = -1;
      result[i][1] = -1;
    }

    if (items1 == null || items2 == null || items1[0] == null || items2[0] == null)
      return result;

    BinaryHeap heap = new BinaryHeap(K + 1);
    int n = 0;
    ValueIndexPair vip = new ValueIndexPair(items1[0].prob + items2[0].prob, 0, 0);

    heap.add(vip);
    beenPushed[0][0] = true;

    while (n < K) {
      vip = heap.removeMax();

      if (vip.val == Double.NEGATIVE_INFINITY)
        break;

      result[n][0] = vip.i1;
      result[n][1] = vip.i2;

      n++;
      if (n >= K)
        break;

      if (!beenPushed[vip.i1 + 1][vip.i2]) {
        heap.add(new ValueIndexPair(items1[vip.i1 + 1].prob + items2[vip.i2].prob, vip.i1 + 1,
                vip.i2));
        beenPushed[vip.i1 + 1][vip.i2] = true;
      }
      if (!beenPushed[vip.i1][vip.i2 + 1]) {
        heap.add(new ValueIndexPair(items1[vip.i1].prob + items2[vip.i2 + 1].prob, vip.i1,
                vip.i2 + 1));
        beenPushed[vip.i1][vip.i2 + 1] = true;
      }

    }

    return result;
  }
  
  private static class ValueIndexPair {
    public double val;

    public int i1, i2;

    public ValueIndexPair(double val, int i1, int i2) {
      this.val = val;
      this.i1 = i1;
      this.i2 = i2;
    }

    public int compareTo(ValueIndexPair other) {
      if (val < other.val)
        return -1;
      if (val > other.val)
        return 1;
      return 0;
    }

  }

  // Max Heap
  // We know that never more than K elements on Heap
  private static class BinaryHeap {
    private int DEFAULT_CAPACITY;

    private int currentSize;

    private ValueIndexPair[] theArray;

    public BinaryHeap(int def_cap) {
      DEFAULT_CAPACITY = def_cap;
      theArray = new ValueIndexPair[DEFAULT_CAPACITY + 1];
      // theArray[0] serves as dummy parent for root (who is at 1)
      // "largest" is guaranteed to be larger than all keys in heap
      theArray[0] = new ValueIndexPair(Double.POSITIVE_INFINITY, -1, -1);
      currentSize = 0;
    }

    public ValueIndexPair getMax() {
      return theArray[1];
    }

    private int parent(int i) {
      return i / 2;
    }

    private int leftChild(int i) {
      return 2 * i;
    }

    private int rightChild(int i) {
      return 2 * i + 1;
    }

    public void add(ValueIndexPair e) {

      // bubble up:
      int where = currentSize + 1; // new last place
      while (e.compareTo(theArray[parent(where)]) > 0) {
        theArray[where] = theArray[parent(where)];
        where = parent(where);
      }
      theArray[where] = e;
      currentSize++;
    }

    public ValueIndexPair removeMax() {
      ValueIndexPair min = theArray[1];
      theArray[1] = theArray[currentSize];
      currentSize--;
      boolean switched = true;
      // bubble down
      for (int parent = 1; switched && parent < currentSize;) {
        switched = false;
        int leftChild = leftChild(parent);
        int rightChild = rightChild(parent);

        if (leftChild <= currentSize) {
          // if there is a right child, see if we should bubble down there
          int largerChild = leftChild;
          if ((rightChild <= currentSize)
              && (theArray[rightChild].compareTo(theArray[leftChild])) > 0) {
            largerChild = rightChild;
          }
          if (theArray[largerChild].compareTo(theArray[parent]) > 0) {
            ValueIndexPair temp = theArray[largerChild];
            theArray[largerChild] = theArray[parent];
            theArray[parent] = temp;
            parent = largerChild;
            switched = true;
          }
        }
      }
      return min;
    }

  }

}
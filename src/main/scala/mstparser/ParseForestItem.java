package mstparser;

public class ParseForestItem {

  public int s, r, t, dir, comp, length, type;

  public double prob;

  public FeatureVector fv;

  public ParseForestItem left, right;

  // productions
  public ParseForestItem(int i, int k, int j, int type, int dir, int comp, double prob,
          FeatureVector fv, ParseForestItem left, ParseForestItem right) {
    this.s = i;
    this.r = k;
    this.t = j;
    this.dir = dir;
    this.comp = comp;
    this.type = type;
    length = 6;

    this.prob = prob;
    this.fv = fv;

    this.left = left;
    this.right = right;

  }

  // preproductions
  public ParseForestItem(int s, int type, int dir, double prob, FeatureVector fv) {
    this.s = s;
    this.dir = dir;
    this.type = type;
    length = 2;

    this.prob = prob;
    this.fv = fv;

    left = null;
    right = null;

  }

  public ParseForestItem() {
  }

  public void copyValues(ParseForestItem p) {
    p.s = s;
    p.r = r;
    p.t = t;
    p.dir = dir;
    p.comp = comp;
    p.prob = prob;
    p.fv = fv;
    p.length = length;
    p.left = left;
    p.right = right;
    p.type = type;
  }

  // way forest works, only have to check rule and indeces
  // for equality.
  public boolean equals(ParseForestItem p) {
    return s == p.s && t == p.t && r == p.r && dir == p.dir && comp == p.comp && type == p.type;
  }

  public boolean isPre() {
    return length == 2;
  }

}

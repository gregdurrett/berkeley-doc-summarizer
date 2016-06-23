///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2007 University of Texas at Austin and (C) 2005
// University of Pennsylvania and Copyright (C) 2002, 2003 University
// of Massachusetts Amherst, Department of Computer Science.
//
// This software is licensed under the terms of the Common Public
// License, Version 1.0 or (at your option) any subsequent version.
//
// The license is approved by the Open Source Initiative, and is
// available from their website at http://www.opensource.org.
///////////////////////////////////////////////////////////////////////////////

package mstparser;

import gnu.trove.TLinkableAdaptor;

/**
 * A simple class holding a feature index and value that can be used in a TLinkedList.
 * 
 * <p>
 * Created: Sat Nov 10 15:25:10 2001
 * </p>
 * 
 * @author Jason Baldridge
 * @version $Id: TLinkedList.java,v 1.5 2005/03/26 17:52:56 ericdf Exp $
 * @see mstparser.FeatureVector
 */

public final class Feature extends TLinkableAdaptor {

  public int index;

  public double value;

  public Feature(int i, double v) {
    index = i;
    value = v;
  }

  @Override
  public final Feature clone() {
    return new Feature(index, value);
  }

  public final Feature negation() {
    return new Feature(index, -value);
  }

  @Override
  public final String toString() {
    return index + "=" + value;
  }

}

package edu.berkeley.nlp.summ.data

import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer

trait DepParseDoc extends Serializable {
  
  def name: String
  def doc: Seq[DepParse]
  def summary: Seq[DepParse]

  override def toString() = {
    toString(Int.MaxValue)
  }
  
  def toString(maxNumSentences: Int) = {
    "DOCUMENT:\n" + doc.map(_.getWords.reduce(_ + " " + _)).slice(0, Math.min(maxNumSentences, doc.size)).reduce(_ + "\n" + _) +
    "\nSUMMARY:\n" + summary.map(_.getWords.reduce(_ + " " + _)).slice(0, Math.min(maxNumSentences, doc.size)).reduce(_ + "\n" + _)
  }
}

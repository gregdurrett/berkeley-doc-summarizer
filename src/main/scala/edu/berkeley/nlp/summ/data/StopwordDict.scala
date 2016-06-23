package edu.berkeley.nlp.summ.data

import scala.collection.mutable.HashSet

object StopwordDict {

  // N.B. This set was extracted from the RST treebank (train and test) mostly to reproduce
  // Hirao's results; it shouldn't really be used for other things
  val stopwords = Set("!", "", "#", "$", "%", "&", "'", "''", "'S", "'s", "()", ",", "-", "--", "-owned", ".", "", ":", ";", "<", "?", "",
                      "A", "A.", "", "AND", "After", "All", "Am", "An", "And", "Any", "As", "At", "BE", "Between", "Both", "But", "By", "Each",
                      "Few", "For", "From", "Had", "He", "Here", "How", "I", "If", "In", "Is", "It", "Its", "MORE", "More", "Most", "NO", "No", "No.",
                      "Not", "OF", "Of", "On", "One", "Only", "Or", "Other", "Our", "Over", "She", "So", "Some", "Such", "THE", "Than", "That", "The",
                      "Their", "Then", "There", "These", "They", "Those", "To", "UPS", "Under", "Until", "WHY", "We", "What", "When", "While", "Why",
                      "Would", "You", "`It", "``", "a", "about", "above", "after", "again", "again.", "", "against", "all", "am", "an", "and", "any",
                      "as", "at", "be", "been", "being", "below", "between", "both", "but", "by", "ca", "can", "could", "did", "do", "doing", "down",
                      "each", "few", "for", "from", "further", "had", "have", "having", "he", "her", "here", "herself", "him", "him.", "", "himself",
                      "how", "if", "in", "into", "is", "it", "its", "itself", "let", "lets", "me", "more", "most", "must", "my", "n't", "no", "nor",
                      "not", "of", "off", "on", "one", "ones", "only", "or", "other", "others", "ought", "our", "out", "over", "own", "owned", "owns",
                      "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "those",
                      "through", "to", "too", "under", "until", "up", "very", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
                      "why", "with", "wo", "would", "you", "your", "yourself", "{", "}")
  // Leave $ in there
  val stopwordTags = new HashSet[String] ++ Array("CC", "DT", "EX", "IN", "LS", "MD", "PDT", "POS", "PRN", "PRP", "PRP$", "RP", "SYM",
                                                  "TO", "WDT", "WP", "WP$", "WRB", ".", ",", "``", "''", ";", ":", "-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-")
                                                  
}
package edu.berkeley.nlp.summ.demo

import edu.berkeley.nlp.futile.fig.basic.IOUtils
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.util.Logger
import javax.swing.JLabel
import javax.swing.JFrame
import javax.swing.JPanel
import java.awt.Graphics
import java.awt.Dimension
import java.awt.Image
import java.awt.Font
import java.awt.RenderingHints
import java.awt.Graphics2D
import javax.swing.JScrollPane
import java.awt.Color
import javax.swing.JButton
import javax.swing.AbstractButton
import java.awt.event.ActionListener
import java.awt.event.ActionEvent
import java.awt.BorderLayout
import java.awt.FlowLayout
import javax.swing.JTextField
import javax.swing.JCheckBox
import javax.swing.JTextArea
import javax.swing.BorderFactory
import org.gnu.glpk.GLPK
import javax.swing.border.EmptyBorder
import javax.swing.text.SimpleAttributeSet
import javax.swing.text.StyleConstants
import javax.swing.JTextPane
import edu.berkeley.nlp.futile.LightRunner
import javax.swing.text.BadLocationException
import javax.swing.ScrollPaneConstants
import javax.swing.JSplitPane
import edu.berkeley.nlp.futile.syntax.Tree
import java.awt.geom.Point2D
import javax.swing.JComboBox
import java.awt.CardLayout
import java.awt.BasicStroke
import javax.swing.JSeparator
import javax.swing.SwingConstants
import edu.berkeley.nlp.summ.CompressiveAnaphoraSummarizer
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed
import edu.berkeley.nlp.summ.RougeFileMunger
import edu.berkeley.nlp.summ.data.PronounReplacement
import edu.berkeley.nlp.summ.Main

/**
 * GUI that shows the summarizer's output graphically and how it relates to the
 * preprocessing
 */
object SummarizerDemo {
  
  val resolutionX = 1000
  val resolutionY = 600
  val startingSizeX = 1000
  val startingSizeY = 600
  val summPanelSizeX = 300
  
  val preprocDataPath = "preproc-cached-realsplit.ser.gz"
  val modelPath = "models/summarizer-extractive.ser.gz"
  val noCorefModelPath = "models/summarizer-extractive-compressive.ser.gz"
  val sentModelPath = "models/summarizer-full.ser.gz"
  
  val docFontSize = 18
  val summFontSize = 24
  
  val initialYOffset = (docFontSize * 1.5).toInt
  val yDelta = (docFontSize * 1.5).toInt

  val useOneModel = true
  
  val alignmentColor = new Color(170, 0, 0)
  val corefColor = new Color(0, 66, 170)
  
  val systemNames = Seq("Document prefix baseline", "Sentence extractive", "Extractive and compressive", "Full system")
  
  val customFont = new Font("Times", Font.PLAIN, summFontSize);
  
  def main(args: Array[String]) {
    LightRunner.populateScala(SummarizerDemo.getClass(), args)
    
    Logger.logss("Loading examples")
    val (_, evalExs) = IOUtils.readObjFileHard(preprocDataPath).asInstanceOf[(ArrayBuffer[DiscourseDepExProcessed],ArrayBuffer[DiscourseDepExProcessed])]
    Logger.logss("Loading models...")
    val model = IOUtils.readObjFile(modelPath).asInstanceOf[CompressiveAnaphoraSummarizer]
    Logger.logss("1 of 3...")
    val noCorefModel = if (useOneModel) model else IOUtils.readObjFile(noCorefModelPath).asInstanceOf[CompressiveAnaphoraSummarizer]
//    val noCorefModel = model
    Logger.logss("2 of 3...")
    val sentModel = if (useOneModel) model else IOUtils.readObjFile(sentModelPath).asInstanceOf[CompressiveAnaphoraSummarizer]
//    val sentModel = model
    Logger.logss("Done!")
    
    val goodIDSeq = Seq("1815761", "1815757", "1815974")
    val newExs = new ArrayBuffer[DiscourseDepExProcessed]
    for (id <- goodIDSeq) {
      newExs ++= evalExs.filter(_.rawDoc.corefDoc.rawDoc.docID == id)
    }
    newExs ++= evalExs.filter(ex => !goodIDSeq.contains(ex.rawDoc.corefDoc.rawDoc.docID))
    createAndShowRealGUI(newExs, model, noCorefModel, sentModel)
  }
  
  def createAndShowRealGUI(allExs: Seq[DiscourseDepExProcessed],
                           model: CompressiveAnaphoraSummarizer,
                           noCorefModel: CompressiveAnaphoraSummarizer,
                           sentModel: CompressiveAnaphoraSummarizer) {
    val frame: JFrame = new JFrame("Berkeley Summarizer");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    val outerPanel = new JPanel
    val summaryPanel = new SummPanel(detailsPane = false)
    val detailsSummaryPanel = new SummPanel(detailsPane = true)
    
    val comprCheckbox = new JCheckBox("Use compression")
    val corefCheckbox = new JCheckBox("Use coreference")
    val comboBox = new JComboBox(systemNames.toArray);
    
    
    val rendererPanel = new RendererPanel(allExs.iterator, model, noCorefModel, sentModel, summaryPanel, Some(detailsSummaryPanel), comprCheckbox, corefCheckbox, comboBox)
    
    val scrollPane = new JScrollPane()
    scrollPane.setViewportView(rendererPanel)
    scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS)
    scrollPane.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS)
    outerPanel.add(scrollPane)
    val compressButton = new JButton("Compression")
    compressButton.setActionCommand("compression")
    compressButton.addActionListener(rendererPanel)
    val corefButton = new JButton("Coreference")
    corefButton.setActionCommand("coref")
    corefButton.addActionListener(rendererPanel)
    val summarizeButton = new JButton("Summarize")
    summarizeButton.setActionCommand("summarize")
    summarizeButton.addActionListener(rendererPanel)
    val nextExButton = new JButton("Next")
    nextExButton.setActionCommand("nextex")
    nextExButton.addActionListener(rendererPanel)
    val parsesButton = new JButton("Parses")
    parsesButton.setActionCommand("parses")
    parsesButton.addActionListener(rendererPanel)
    
    val buttonPane = new JPanel(new FlowLayout())
    
    buttonPane.add(comboBox)
    buttonPane.add(summarizeButton)
    buttonPane.add(compressButton)
    buttonPane.add(corefButton)
    buttonPane.add(parsesButton)
    buttonPane.add(nextExButton)
    
    
    val bottomPane = new JPanel(new BorderLayout())
    bottomPane.add(buttonPane, BorderLayout.SOUTH)
    
    val contentPane = new JPanel(new BorderLayout());
    
    contentPane.add(bottomPane, BorderLayout.SOUTH)
    
    val innerSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, scrollPane, detailsSummaryPanel);
    val outerSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, innerSplitPane, summaryPanel);
    innerSplitPane.setOneTouchExpandable(true)
    outerSplitPane.setOneTouchExpandable(true)
    // WHAT WORKS: Setting the minimum size of the scroll pane
    val scrollPaneSize = new Dimension(startingSizeX - summPanelSizeX, startingSizeY - 100)
    val detailsSize = new Dimension(summPanelSizeX, 100)
    val summarySize = new Dimension(summPanelSizeX, 100)
    scrollPane.setPreferredSize(scrollPaneSize)
    innerSplitPane.setPreferredSize(new Dimension((scrollPaneSize.getWidth() + detailsSize.getWidth()).toInt, (scrollPaneSize.getHeight() + detailsSize.getHeight()).toInt))
    detailsSummaryPanel.setPreferredSize(detailsSize)
    summaryPanel.setPreferredSize(summarySize)
    // DOES NOT WORK
    summaryPanel.setMaximumSize(summarySize)
    contentPane.add(outerSplitPane)
    
    // BORDERS
    val raisedbevel = BorderFactory.createRaisedBevelBorder();
    val loweredbevel = BorderFactory.createLoweredBevelBorder();
    val compound = BorderFactory.createCompoundBorder(raisedbevel, loweredbevel);
    
    val compoundWithSpace = BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(contentPane.getBackground(), 2), compound)
    scrollPane.setBorder(compoundWithSpace)
    val compoundWithSpaceAndWhite = BorderFactory.createCompoundBorder(compoundWithSpace, BorderFactory.createLineBorder(Color.WHITE, 4))
    summaryPanel.setBorder(compoundWithSpaceAndWhite)
    detailsSummaryPanel.setBorder(compoundWithSpaceAndWhite)


    frame.setContentPane(contentPane);
    frame.setPreferredSize(new Dimension(startingSizeX, startingSizeY))
    frame.pack();
    frame.setVisible(true);
  }
}

class ParsePanel extends JPanel {
  var ex: DiscourseDepExProcessed = null
  var includedEdus: Seq[Int] = null
  var compressionSpans: Seq[(Int,Int)] = null
  var pronReplacedSpans: Seq[(Int,Int)] = null
  var pronReplacements: Seq[String] = null
  
  val treesIndent = 20
  val treesVerticalGap = 60
  
  override def paintComponent(g: Graphics) {
    val g2 = g.asInstanceOf[Graphics2D]
    g2.setColor(Color.WHITE);
    g2.fillRect(0, 0, getWidth(), getHeight());
    g2.setColor(getForeground());
    g2.setFont(SummarizerDemo.customFont)
    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    if (ex != null && includedEdus != null) {
      val oldFont = g2.getFont();
      g2.setFont(new Font(oldFont.getName(), oldFont.getStyle(), 14));    
      val includedSents = includedEdus.map(ex.eduAlignments(_)._1._1).distinct
      var currX = treesIndent
      var currY = treesIndent
      var charOffset = 0
      val lineHeights = new ArrayBuffer[Int]
      var maxTreeWidth = 0
      for (sentIdx <- includedSents) {
        val constTreeRaw = ex.rawDoc.corefDoc.rawDoc.trees(sentIdx).constTree
        val constTree = if (constTreeRaw.getLabel() == "TOP") constTreeRaw.getChild(0) else constTreeRaw
        val sentEdus = ex.eduAlignments.filter(_._1._1 == sentIdx).map(edu => edu._1._2 -> edu._2._2)
        val widthHeight = TreeJPanel.paintTreeModified(flattenTree(constTree, sentEdus), new Point2D.Double(currX, currY), g2, g2.getFontMetrics(), charOffset, Color.black,
                                     compressionSpans.map(tuple => new edu.berkeley.nlp.futile.fig.basic.Pair(new Integer(tuple._1), new Integer(tuple._2))).asJava,
                                     pronReplacedSpans.map(tuple => new edu.berkeley.nlp.futile.fig.basic.Pair(new Integer(tuple._1), new Integer(tuple._2))).asJava,
                                     pronReplacements.asJava)
        maxTreeWidth = Math.max(maxTreeWidth, widthHeight.getFirst().intValue)
        // Count both characters and spaces between words
        charOffset += constTree.getYield().asScala.map(_.size).foldLeft(0)(_ + _) + constTree.getYield().size
        currY += widthHeight.getSecond().intValue + treesVerticalGap
        lineHeights += currY - treesVerticalGap/2
      }
      val g2Copy = g.create().asInstanceOf[Graphics2D];
      val dashed = new BasicStroke(2, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, Array(9F), 0);
      g2Copy.setStroke(dashed);
      for (height <- lineHeights) {
        g2Copy.drawLine(0, height, maxTreeWidth + treesIndent * 2, height)
      }
      g2Copy.dispose()
      g2.setFont(oldFont);
      this.setPreferredSize(new Dimension(maxTreeWidth + treesIndent * 2, currY))
    } else {
      this.setPreferredSize(new Dimension(200, 600))
    }
  }
  
  // HELPER METHODS FOR TREES
  
  def flattenTree(tree: Tree[String], eduBoundaries: Seq[(Int,Int)]): Tree[String] = {
    flattenTree(eduBoundaries, tree, 0)
  }
  
  def flattenTree(eduBoundaries: Seq[(Int,Int)], subtree: Tree[String], startIdx: Int): Tree[String] = {
    val endIdx = startIdx + subtree.getYield().size
    // If the tree is wholly contained in an EDU, just triangle it out
    if (eduBoundaries.filter(edu => edu._1 <= startIdx && endIdx <= edu._2).size > 0) {
      new Tree[String](subtree.getLabel(), Seq(new Tree[String](subtree.getYield().asScala.foldLeft("")(_ + " " + _).trim, Seq().asJava)).asJava)
    } else {
      // Recurse
      val newChildren = new ArrayBuffer[Tree[String]]
      var currStartIdx = startIdx
      for (child <- subtree.getChildren().asScala) {
        newChildren += flattenTree(eduBoundaries, child, currStartIdx)
        currStartIdx += child.getYield().size
      }
      new Tree[String](subtree.getLabel(), newChildren.asJava)
    }
  }
}

class SummPanel(detailsPane: Boolean) extends JPanel {
  
  val textPane = new JTextPane()
  val parsePane = new ParsePanel()
  val parseScrollPane = new JScrollPane()
  parseScrollPane.setViewportView(parsePane)
  parseScrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS)
  
  textPane.setText("")
  textPane.setOpaque(true)
  textPane.setBackground(Color.WHITE)
  textPane.setEditable(false)
  
  parsePane.setOpaque(true)
  parsePane.setBackground(Color.WHITE)
  
  this.setLayout(new CardLayout)
  add(textPane, SummPanel.TEXT_CARD)
  add(parseScrollPane, SummPanel.PARSE_CARD)
  
  val set = new SimpleAttributeSet(textPane.getParagraphAttributes());
  StyleConstants.setLineSpacing(set, 0.3F);
  textPane.setParagraphAttributes(set, true)
  textPane.setFont(SummarizerDemo.customFont)
  
  var ex: DiscourseDepExProcessed = null
  var text = ""
  var includedEdus: Seq[Int] = null
  var compressionSpans: Seq[(Int,Int)] = null
  var pronReplacedSpans: Seq[(Int,Int)] = null
  var pronReplacements: Seq[String] = null
  
  var showTreeStructure = false
  
  def refreshExample(newEx: DiscourseDepExProcessed) {
    this.ex = newEx
    parsePane.ex = newEx
    clearSummaryText()
  }
  
  def clearSummaryText() {
    setDetailedSummaryText("", Seq(), Seq(), Seq(), Seq())
  }
  
  def setNormalSummaryText(text: String) {
    setDetailedSummaryText(text, Seq(), Seq(), Seq(), Seq())
  }
  
  def setDetailedSummaryText(text: String, includedEdus: Seq[Int], compressionSpans: Seq[(Int,Int)], pronReplacedSpans: Seq[(Int,Int)], pronReplacements: Seq[String]) {
    this.text = text
    this.includedEdus = includedEdus
    this.compressionSpans = compressionSpans
    this.pronReplacedSpans = pronReplacedSpans
    this.pronReplacements = pronReplacements
    parsePane.includedEdus = includedEdus
    parsePane.compressionSpans = compressionSpans
    parsePane.pronReplacedSpans = pronReplacedSpans
    parsePane.pronReplacements = pronReplacements
    updateTextPane()
  }
  
  def updateTextPane() {
    textPane.setText("")
    val doc = textPane.getStyledDocument();
    val style = textPane.addStyle("", null);
    val spanStartsSet = (Seq(0, text.size) ++ compressionSpans.map(_._1) ++ compressionSpans.map(_._2) ++ pronReplacedSpans.map(_._1) ++ pronReplacedSpans.map(_._2)).sorted
    var pronRepIdx = 0
    for (i <- 0 until spanStartsSet.size - 1) {
      val start = spanStartsSet(i)
      val end = spanStartsSet(i+1)
      var inPronRep = false
      if (compressionSpans.map(_._1).contains(start)) {
        StyleConstants.setForeground(style, Color.gray);
      } else if (pronReplacedSpans.map(_._1).contains(start)) {
        StyleConstants.setForeground(style, SummarizerDemo.corefColor);
        inPronRep = true
      } else {
        StyleConstants.setForeground(style, Color.black);
      }
      try {
        val nextSnippet = text.substring(start, end)
        val nextStr = if (inPronRep) {
          pronRepIdx += 1
          "[" + nextSnippet + " \u2192 " + pronReplacements(pronRepIdx - 1) + "]"
        } else {
          nextSnippet
        }
        doc.insertString(doc.getLength(), nextStr, style);
      } catch {
        case e: BadLocationException => throw new RuntimeException(e)
      }
    }
  }
  
  override def paintComponent(g: Graphics) {
    g.asInstanceOf[Graphics2D].setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    if (detailsPane && showTreeStructure) {
//      parsePane.repaint()
      parseScrollPane.repaint()
      parseScrollPane.revalidate()
    } else {
      textPane.getGraphics().asInstanceOf[Graphics2D].setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_GASP)
      textPane.repaint()
    }
  }
}

object SummPanel {
  val TEXT_CARD = "text"
  val PARSE_CARD = "parse"
}

class RendererPanel(allExs: Iterator[DiscourseDepExProcessed],
                    model: CompressiveAnaphoraSummarizer,
                    noCorefModel: CompressiveAnaphoraSummarizer,
                    sentModel: CompressiveAnaphoraSummarizer,
                    summaryPanel: SummPanel,
                    maybeSecondSummaryPanel: Option[SummPanel],
                    comprCheckBox: JCheckBox,
                    corefCheckBox: JCheckBox,
                    comboBox: JComboBox[String]) extends JPanel with ActionListener {
  
  var showCompressions: Boolean = false
  var showCoref: Boolean = false
  
  var isCurrSummarySentence = true
  var currSummary = Seq[Int]()
  
  setLayout(null);
  val customFont = new Font("Times", Font.PLAIN, SummarizerDemo.docFontSize);
    
  val corefPredictor = model.computer.corefPredictor
  
  var ex = allExs.next
  summaryPanel.refreshExample(ex)
  if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.refreshExample(ex)
  var conllDoc = ex.rawDoc.corefDoc.rawDoc
  var mentions = ex.rawDoc.corefDoc.goldMentions
  var pronReps = ex.identifyPronounReplacements(model.computer.replaceWithNEOnly, corefPredictor, model.computer.corefConfidenceThreshold)
  var fragileProns = ex.identifyFragilePronouns(corefPredictor)
  
  val maxNumSents = 100
    
  setOpaque(true)
  setBackground(Color.WHITE)
  
  
  def refreshExample(newEx: DiscourseDepExProcessed) {
    Logger.logss("Loading " + newEx.rawDoc.corefDoc.rawDoc.docID)
    ex = newEx
    conllDoc = ex.rawDoc.corefDoc.rawDoc
    mentions = ex.rawDoc.corefDoc.goldMentions
    pronReps = ex.identifyPronounReplacements(model.computer.replaceWithNEOnly, corefPredictor, model.computer.corefConfidenceThreshold)
    fragileProns = ex.identifyFragilePronouns(corefPredictor)
    summaryPanel.refreshExample(newEx)
    if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.refreshExample(newEx)
    currSummary = Seq[Int]()
    repaint()
  }
  
  def actionPerformed(e: ActionEvent) {
    val refSummary = ex.rawDoc.summSents
    val refLen = refSummary.map(_.size).foldLeft(0)(_ + _)
    if ("compression".equals(e.getActionCommand())) {
      showCompressions = !showCompressions
      this.repaint()
    }
    if ("coref".equals(e.getActionCommand())) {
      showCoref = !showCoref
      this.repaint()
    }
    if ("parses".equals(e.getActionCommand())) {
      if (maybeSecondSummaryPanel.isDefined) {
        val cl = maybeSecondSummaryPanel.get.getLayout().asInstanceOf[CardLayout];
        maybeSecondSummaryPanel.get.showTreeStructure = !maybeSecondSummaryPanel.get.showTreeStructure
        if (maybeSecondSummaryPanel.get.showTreeStructure) {
          cl.show(maybeSecondSummaryPanel.get, SummPanel.PARSE_CARD);
        } else {
          cl.show(maybeSecondSummaryPanel.get, SummPanel.TEXT_CARD);
        }
        maybeSecondSummaryPanel.get.repaint()
      }
    }
    if ("summarize".equals(e.getActionCommand())) {
      Logger.logss("Summarizing...")
      val selection = comboBox.getSelectedItem
      val startTime = System.nanoTime()
      ex.cachedSummFeats = null
      ex.cachedPronFeats = null
      ex.cachedBigramFeats = null
      // FULL SYSTEM
      if (selection == SummarizerDemo.systemNames(3)) {
        isCurrSummarySentence = false
        val (edus, prons) = model.computer.decode(ex, model.weights, refLen)
        val pronRepsUsed = prons.map(pronReps(_))
        currSummary = edus
        val (summTextRaw, fullText, comprSpans, pronRepSpans) = ex.getSummaryTextWithPronounsReplacedDemo(edus, pronRepsUsed)
        val summText = RougeFileMunger.detokenizeSentence(summTextRaw.foldLeft("")(_ + " " + _).trim)
        summaryPanel.setNormalSummaryText(summText)
        if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.setDetailedSummaryText(fullText, edus, comprSpans, pronRepSpans, pronRepsUsed.map(_.replacementWords.foldLeft("")(_ + " " + _).trim))
      } else if (selection == SummarizerDemo.systemNames(2)) {
        isCurrSummarySentence = false
        val (edus, prons) = noCorefModel.computer.decode(ex, noCorefModel.weights, refLen)
        val pronRepsUsed = prons.map(pronReps(_))
        currSummary = edus
        val (summTextRaw, fullText, comprSpans, pronRepSpans) = ex.getSummaryTextWithPronounsReplacedDemo(edus, Seq[PronounReplacement]())
        val summText = RougeFileMunger.detokenizeSentence(summTextRaw.foldLeft("")(_ + " " + _).trim)
        summaryPanel.setNormalSummaryText(summText)
        if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.setDetailedSummaryText(fullText, edus, comprSpans, pronRepSpans, pronRepsUsed.map(_.replacementWords.foldLeft("")(_ + " " + _).trim))
      } else if (selection == SummarizerDemo.systemNames(1)) {
        isCurrSummarySentence = true
        val sentenceEx = DiscourseDepExProcessed.makeTrivial(ex.rawDoc)
        val (edus, prons) = sentModel.computer.decode(sentenceEx, sentModel.weights, refLen)
        currSummary = edus
        val summTextRaw = sentenceEx.getSummaryTextWithPronounsReplaced(edus, Seq[PronounReplacement](), true)
        val summText = RougeFileMunger.detokenizeSentence(summTextRaw.foldLeft("")(_ + " " + _).trim)
        summaryPanel.setNormalSummaryText(summText)
        if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.clearSummaryText()
      } else {
        isCurrSummarySentence = true
        val sentenceEx = DiscourseDepExProcessed.makeTrivial(ex.rawDoc)
        val edus = Main.getFirstKSentences(sentenceEx, refLen + 5)
        currSummary = edus
        val summTextRaw = sentenceEx.getSummaryTextWithPronounsReplaced(edus, Seq[PronounReplacement](), true)
        val summText = RougeFileMunger.detokenizeSentence(summTextRaw.foldLeft("")(_ + " " + _).trim)
        summaryPanel.setNormalSummaryText(summText)
        if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.clearSummaryText()
      }
      Logger.logss("Done summarizing! Summarized " + ex.rawDoc.corefDoc.rawDoc.numSents + " sentences in " + (System.nanoTime() - startTime)/1000000 + " millis")
      this.repaint()
      summaryPanel.repaint()
      if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.repaint()
    }
    if ("nextex".equals(e.getActionCommand())) {
      if (allExs.hasNext) {
        refreshExample(allExs.next())
      }
      summaryPanel.repaint()
      if (maybeSecondSummaryPanel.isDefined) maybeSecondSummaryPanel.get.repaint()
      this.repaint()
    }
  }

  override def paintComponent(g: Graphics) {
    val mentXStarts = Array.tabulate(mentions.size)(i => 0)
    val mentXEnds = Array.tabulate(mentions.size)(i => 0)
    
    g.setFont(customFont)
    g.asInstanceOf[Graphics2D].setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_GASP)
    g.asInstanceOf[Graphics2D].setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    
    var maxXOffset = 0
    var yOffset = SummarizerDemo.initialYOffset
    for (i <- 0 until Math.min(conllDoc.numSents, maxNumSents)) {
      val eduStartIdx = ex.eduAlignments.filter(_._1._1 < i).size
      val eduEndIdx = ex.eduAlignments.filter(_._1._1 <= i).size
      
      val sentSplit = ex.eduAlignments.filter(_._1._1 == i).map(eduAlignment => eduAlignment._1._2 -> eduAlignment._2._2)
      val sentParents = ex.parents.slice(eduStartIdx, eduEndIdx).map(_ - eduStartIdx)
      val sentParentLabels = ex.parentLabels.slice(eduStartIdx, eduEndIdx)
      
      val eduSpans = ex.eduAlignments.slice(eduStartIdx, eduEndIdx).map(edu => edu._1._2 -> edu._2._2)
      var xOffset = 10
      val fm = g.getFontMetrics
      for (spanIdx <- 0 until eduSpans.size) {
        var depth = 0
        var parent = sentParents(spanIdx)
        while (parent >= 0 && parent < sentParents.size) {
          if (!sentParentLabels(parent).startsWith("=")) {
            depth += 1
          }
          parent = sentParents(parent)
        }
        val spanStart = eduSpans(spanIdx)._1
        val spanEnd = eduSpans(spanIdx)._2
        val mentIndicesStartingHere = (0 until mentions.size).filter(mentIdx => mentions(mentIdx).sentIdx == i && mentions(mentIdx).startIdx >= spanStart && mentions(mentIdx).startIdx < spanEnd)
        val mentIndicesEndingHere = (0 until mentions.size).filter(mentIdx => mentions(mentIdx).sentIdx == i && mentions(mentIdx).endIdx > spanStart && mentions(mentIdx).endIdx <= spanEnd)
        for (mentIdx <- mentIndicesStartingHere) {
          val prefixStr = conllDoc.words(i).slice(spanStart, mentions(mentIdx).startIdx).foldLeft("")(_ + " " + _).trim
          mentXStarts(mentIdx) = xOffset + fm.stringWidth(prefixStr)
        }
        for (mentIdx <- mentIndicesEndingHere) {
          val prefixStr = conllDoc.words(i).slice(spanStart, mentions(mentIdx).endIdx).foldLeft("")(_ + " " + _).trim
          mentXEnds(mentIdx) = xOffset + fm.stringWidth(prefixStr)
        }
        val eduStr = conllDoc.words(i).slice(spanStart, spanEnd).reduce(_ + " " + _)
        val isInSummary = (isCurrSummarySentence && currSummary.contains(i)) || (!isCurrSummarySentence && currSummary.contains(eduStartIdx + spanIdx))
        if (showCompressions) {
          if (depth > 0) {
            if (isInSummary) {
              g.setColor(new Color(250, 120, 120))
            } else {
              g.setColor(Color.GRAY)
            }
          } else if (isInSummary) {
            g.setColor(SummarizerDemo.alignmentColor)
          } else {
            g.setColor(Color.BLACK)
          }
          g.drawString(eduStr, xOffset, yOffset)
        } else {
          if (isInSummary) {
            g.setColor(SummarizerDemo.alignmentColor)
          }
          g.drawString(eduStr, xOffset, yOffset)
        }
        g.setColor(Color.BLACK)
        
        // N.B. 6 is the right number for a space 
        xOffset += fm.stringWidth(eduStr) + 6
      }
      maxXOffset = Math.max(maxXOffset, xOffset)
      yOffset += SummarizerDemo.yDelta
    }
    this.setPreferredSize(new Dimension(maxXOffset, yOffset))
    g.setColor(Color.BLACK)
    if (showCoref) {
      for (pronRep <- pronReps) {
        if (mentions(pronRep.mentIdx).sentIdx < maxNumSents) {
          drawMentionLine(g, pronRep.mentIdx, pronRep.antIdx, mentXStarts, mentXEnds, SummarizerDemo.corefColor)
        }
      }
      for (fragilePron <- fragileProns) {
        if (mentions(fragilePron.mentIdx).sentIdx < maxNumSents) {
          for (antIdx <- fragilePron.antecedentMentIndices.get) {
            drawMentionLine(g, fragilePron.mentIdx, antIdx, mentXStarts, mentXEnds, Color.BLACK)
          }
        }
      }
    }
  }
  
  def drawMentionLine(g: Graphics, startMentIdx: Int, endMentIdx: Int, mentXStarts: Seq[Int], mentXEnds: Seq[Int], color: Color) {
    val corefDoc = ex.rawDoc.corefDoc
    val startSentIdx = corefDoc.goldMentions(startMentIdx).sentIdx
    val endSentIdx = corefDoc.goldMentions(endMentIdx).sentIdx
    val startYOffset = SummarizerDemo.initialYOffset + SummarizerDemo.yDelta * startSentIdx - 10
    val endYOffset = SummarizerDemo.initialYOffset + SummarizerDemo.yDelta * endSentIdx - 10
    
    val x1 = (mentXStarts(startMentIdx) + mentXEnds(startMentIdx))/2
    val x2 = (mentXStarts(endMentIdx) + mentXEnds(endMentIdx))/2
    g.setColor(color)
    if (startYOffset == endYOffset) {
      g.drawLine(x1, startYOffset, x1, startYOffset - 15)
      g.drawLine(x1, startYOffset - 15, x2, endYOffset - 15)
      g.drawLine(x2, endYOffset - 15, x2, endYOffset)
    } else {
      g.drawLine(x1, startYOffset, x2, endYOffset)
    }
    g.setColor(Color.BLACK)
    g.drawRoundRect(mentXStarts(startMentIdx), startYOffset - 10, mentXEnds(startMentIdx) - mentXStarts(startMentIdx), 25, 10, 10)
    g.drawRoundRect(mentXStarts(endMentIdx), endYOffset - 10, mentXEnds(endMentIdx) - mentXStarts(endMentIdx), 25, 10, 10)
  } 
}

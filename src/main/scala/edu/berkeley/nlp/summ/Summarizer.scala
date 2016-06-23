package edu.berkeley.nlp.summ

import java.io.File
import edu.berkeley.nlp.entity.ConllDocReader
import edu.berkeley.nlp.entity.coref.CorefDocAssembler
import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.entity.coref.NumberGenderComputer
import edu.berkeley.nlp.entity.lang.EnglishCorefLanguagePack
import edu.berkeley.nlp.entity.lang.Language
import edu.berkeley.nlp.futile.LightRunner
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.summ.data.SummDoc
import edu.berkeley.nlp.summ.preprocess.DiscourseDependencyParser
import edu.berkeley.nlp.summ.preprocess.EDUSegmenter
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed

/**
 * Main class for running the summarizer on unlabeled data. See run-summarizer.sh for
 * example usage. The most useful arguments are:
 * -inputDir: directory of files (in CoNLL format, with parses/coref/NER) to summarize
 * -outputDir: directory to write summaries
 * -modelPath if you want to use a different version of the summarizer.
 * 
 * Any member of this class can be passed as a command-line argument to the
 * system if it is preceded with a dash, e.g.
 * -budget 100
 */
object Summarizer {
  
  val numberGenderPath = "data/gender.data";
  val segmenterPath = "models/edusegmenter.ser.gz"
  val discourseParserPath = "models/discoursedep.ser.gz"
  val modelPath = "models/summarizer-full.ser.gz"
  
  val inputDir = ""
  val outputDir = ""
  
  // Indicates that we shouldn't do any discourse preprocessing; this is only appropriate
  // for the sentence-extractive version of the system
  val noRst = false
  
  // Summary budget, in words. Set this to whatever you want it to.
  val budget = 50
  
  def main(args: Array[String]) {
    LightRunner.initializeOutput(Summarizer.getClass())
    LightRunner.populateScala(Summarizer.getClass(), args)
    
    Logger.logss("Loading model...")
    val model = IOUtils.readObjFile(modelPath).asInstanceOf[CompressiveAnaphoraSummarizer]
    Logger.logss("Model loaded!")
    val (segmenter, discourseParser) = if (noRst) {
      (None, None)
    } else {
      Logger.logss("Loading segmenter...")
      val tmpSegmenter = IOUtils.readObjFile(segmenterPath).asInstanceOf[EDUSegmenter]
      Logger.logss("Segmenter loaded!")
      Logger.logss("Loading discourse parser...")
      val tmpDiscourseParser = IOUtils.readObjFile(discourseParserPath).asInstanceOf[DiscourseDependencyParser]
      Logger.logss("Discourse parser loaded!")
      (Some(tmpSegmenter), Some(tmpDiscourseParser))
    }
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(numberGenderPath);
    val mpc = new MentionPropertyComputer(Some(numberGenderComputer))
    
    val reader = new ConllDocReader(Language.ENGLISH)
    val assembler = new CorefDocAssembler(new EnglishCorefLanguagePack, true)
    val filesToSummarize = new File(inputDir).listFiles()
    for (file <- filesToSummarize) {
      val conllDoc = reader.readConllDocs(file.getAbsolutePath).head
      val corefDoc = assembler.createCorefDoc(conllDoc, mpc)
      val summDoc = SummDoc.makeSummDoc(conllDoc.docID, corefDoc, Seq())
      val ex = if (noRst) {
        DiscourseDepExProcessed.makeTrivial(summDoc)
      } else {
        DiscourseDepExProcessed.makeWithEduAndSyntactic(summDoc, segmenter.get, discourseParser.get)
      }
      val summaryLines = model.summarize(ex, budget, true)
      val outWriter = IOUtils.openOutHard(outputDir + "/" + file.getName)
      for (summLine <- summaryLines) {
        outWriter.println(summLine)
      }
      outWriter.close
    }
    LightRunner.finalizeOutput()
  }
}
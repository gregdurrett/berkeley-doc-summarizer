package edu.berkeley.nlp.summ

import java.io.File
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import scala.collection.JavaConverters._

/**
 * Handles writing ROUGE-style XML files
 */
object RougeFileMunger {

  val input = "data/RSTDiscourse/sample-outputs/"
  val output = "data/RSTDiscourse/sample-outputs-rouge/"
  val settingsPath = "data/RSTDiscourse/rouge-settings.xml"
  val detokenize = true
  
  def writeSummary(fileName: String, sents: Seq[String], outPath: String, keepFile: Boolean) {
    val outFile = new File(outPath)
    if (!keepFile) outFile.deleteOnExit()
    val outWriter = IOUtils.openOutHard(outFile)
    outWriter.println("<html>")
    outWriter.println("<head><title>" + fileName + "</title></head>")
    outWriter.println("<<body bgcolor=\"white\">")
    var counter = 1
    for (sent <- sents) {
      outWriter.println("<a name=\"" + counter + "\">[" + counter + "]</a> <a href=\"#" + counter + "\" id=" + counter + ">" + sent + "</a>")
      counter += 1
    }
    outWriter.println("</body>")
    outWriter.println("</html>")
    outWriter.close
  }
  
  def detokenizeSentence(line: String) = {
    line.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";").
         replace("`` ", "``").replace(" ''", "''").replace(" '", "'").replace(" \"", "\"").replace("$ ", "$")
  }
  
  def processFiles(rootPath: String, subDir: String) = {
    val refFiles = new File(rootPath + "/" + subDir).listFiles
    for (refFile <- refFiles) {
      val rawName = refFile.getName()
      val name = rawName.substring(0, if (rawName.indexOf("_") == -1) rawName.size else rawName.indexOf("_"))
      val lines = IOUtils.readLinesHard(refFile.getAbsolutePath()).asScala.map(sent => if (detokenize) detokenizeSentence(sent) else sent)
      writeSummary(name, lines, output + "/" + subDir + "/" + refFile.getName, true)
    }
  }
  
  def writeSettings(settingsPath: String, dirPaths: String) {
    val outWriter = IOUtils.openOutHard(settingsPath)
    outWriter.println("""<ROUGE_EVAL version="1.55">""")
    val rawDirName = new File(dirPaths).getName()
    val docs = new File(dirPaths + "/reference").listFiles
    var idx = 0
    for (doc <- docs) {
      val rawName = doc.getName().substring(0, doc.getName.indexOf("_"))
      outWriter.println("<EVAL ID=\"TASK_" + idx + "\">")
      outWriter.println("<MODEL-ROOT>" + rawDirName + "/reference</MODEL-ROOT>")
      outWriter.println("<PEER-ROOT>" + rawDirName + "/system</PEER-ROOT>")
      outWriter.println("<INPUT-FORMAT TYPE=\"SEE\">  </INPUT-FORMAT>")
      outWriter.println("<PEERS>")
      outWriter.println("<P ID=\"1\">" + rawName + "_system1.txt</P>")
      outWriter.println("</PEERS>")
      outWriter.println("<MODELS>")
      outWriter.println("<M ID=\"1\">" + rawName + "_reference1.txt</M>")
      outWriter.println("</MODELS>")
      outWriter.println("</EVAL>")
      idx += 1
    }
    outWriter.println("</ROUGE_EVAL>")
    outWriter.close
  }
  
  def main(args: Array[String]) {
    processFiles(input, "reference")
    processFiles(input, "system")
    writeSettings(settingsPath, output) 
  }
}
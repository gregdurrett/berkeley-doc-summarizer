package edu.berkeley.nlp.summ

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.sys.process.Process
import edu.berkeley.nlp.futile.classify.ClassifyUtils
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.summ.data.StopwordDict

/**
 * Contains methods for both computing ROUGE losses programmatically as well as
 * dispatching to the actual ROUGE scorer for evaluation.
 */
object RougeComputer {
  
  def getBigrams(words: Seq[String]): Seq[(String,String)] = {
    (0 until words.size - 1).map(i => words(i) -> words(i+1))
  }
  
  def getUnigramsNoStopwords(words: Seq[String], poss: Seq[String]): Seq[(String,String)] = {
    val nonStopIndices = (0 until words.size).filter(i => !StopwordDict.stopwordTags.contains(poss(i)))
    // Unigrams are encoded as bigrams because...
    nonStopIndices.map(idx => words(idx) -> "")
  }
  
  def getBigramsNoStopwords(words: Seq[String], poss: Seq[String]): Seq[(String,String)] = {
//    val nonStopIndices = (0 until words.size).filter(i => !stopwordTags.contains(poss(i)))
    val nonStopIndices = (0 until words.size).filter(i => !StopwordDict.stopwordTags.contains(poss(i)))
    (0 until nonStopIndices.size - 1).map(i => words(nonStopIndices(i)) -> words(nonStopIndices(i+1)))
  }

  def computeRouge1SuffStats(sents: Seq[Seq[String]], summSents: Seq[Seq[String]]) = {
    val sourceUnigrams = sents.map(_.toSet).foldLeft(new HashSet[String])(_ ++ _)
    val targetUnigrams = summSents.map(_.toSet).foldLeft(new HashSet[String])(_ ++ _)
    val numHit = (targetUnigrams & sourceUnigrams).size
    (numHit, targetUnigrams.size)
  }
  
  def computeRouge2SuffStats(sents: Seq[Seq[String]], summSents: Seq[Seq[String]]) = {
    val sourceBigrams = getBigramSet(sents)
    val targetBigrams = getBigramSet(summSents)
    val numHit = (targetBigrams & sourceBigrams).size
    (numHit, targetBigrams.size)
  }
  
  def getBigramSet(sents: Seq[Seq[String]]) = {
    val bigrams = new HashSet[(String,String)]
    for (sent <- sents) {
      for (i <- 0 until sent.size - 1) {
        bigrams += sent(i) -> sent(i+1)
      }
    }
    bigrams
  }
  
  def readSummDocJustTextBySents(fileName: String) = {
    val lines = IOUtils.readLines(fileName).asScala
    val results = lines.map(_.trim).filter(_.size != 0).map(_.split("\\s+").toSeq)
    results
  }
  
  def write(fileName: String, data: Seq[Seq[String]]) {
    val printWriter = IOUtils.openOutHard(fileName)
    data.foreach(sent => printWriter.println(sent.foldLeft("")(_ + " " + _).trim))
    printWriter.close()
  }
  
  /**
   * Takes system and reference summaries, writes them to an output directory (which might be deleted),
   * and call the ROUGE scorer on them. Note that although the summaries aren't "tokenized" in the sense
   * that each line is a single String, they should be the result of taking a tokenized string and joining
   * it with spaces, or the ROUGE scorer won't work correctly
   */
  def evaluateRougeNonTok(sysSumms: Seq[Seq[String]],
                          refSumms: Seq[Seq[String]],
                          rougeDirPath: String,
                          skipRougeEvaluation: Boolean = false,
                          keepRougeDirs: Boolean = false,
                          suppressOutput: Boolean = false): Array[Double] = {
    var unigramRecallNum = 0
    var unigramRecallDenom = 0
    var bigramRecallNum = 0
    var bigramRecallDenom = 0
    var totalWordsUsed = 0
    
    val tmpDirPath = Files.createTempDirectory(Paths.get(rougeDirPath), "outputs")
    val tmpDirAbsPath = tmpDirPath.toAbsolutePath.toString
    val tmpDir = new File(tmpDirAbsPath)
    if (!keepRougeDirs) tmpDir.deleteOnExit()
    val sysDir = new File(tmpDirAbsPath + "/system")
    sysDir.mkdir()
    if (!keepRougeDirs) sysDir.deleteOnExit()
    val refDir = new File(tmpDirAbsPath + "/reference")
    refDir.mkdir()
    if (!keepRougeDirs) refDir.deleteOnExit()
    val settingsFile = File.createTempFile("settings", ".xml", new File(rougeDirPath))
    if (!keepRougeDirs) settingsFile.deleteOnExit()
    
    for (i <- 0 until sysSumms.size) {
      val fileName = "" + i
      val sysSumm = sysSumms(i)
      val refSumm = refSumms(i)
      
      val systemPath = tmpDirAbsPath + "/system/" + fileName  + "_system1.txt"
      
      totalWordsUsed += sysSumm.map(_.split("\\s+").size).foldLeft(0)(_ + _)
      val unigramRecallSuffStats = RougeComputer.computeRouge1SuffStats(sysSumm.map(_.split("\\s+").toSeq), refSumm.map(_.split("\\s+").toSeq))
      unigramRecallNum += unigramRecallSuffStats._1
      unigramRecallDenom += unigramRecallSuffStats._2
      val bigramRecallSuffStats = RougeComputer.computeRouge2SuffStats(sysSumm.map(_.split("\\s+").toSeq), refSumm.map(_.split("\\s+").toSeq))
      bigramRecallNum += bigramRecallSuffStats._1
      bigramRecallDenom += bigramRecallSuffStats._2
      RougeFileMunger.writeSummary(fileName, sysSumm, systemPath, keepRougeDirs)
//      write(systemPath, if (runFirstK) firstK else modelSents)
//      val refPath = outDir + "/reference/" + cleanedFileName + "_reference1.txt"
      val refPath = tmpDirAbsPath + "/reference/" + fileName + "_reference1.txt"
      RougeFileMunger.writeSummary(fileName, refSumm, refPath, keepRougeDirs)
//      write(refPath, summ) 
    }
    if (!suppressOutput) Logger.logss("Unigram recall: " + ClassifyUtils.renderNumerDenom(unigramRecallNum, unigramRecallDenom))
    if (!suppressOutput) Logger.logss("Bigram recall: " + ClassifyUtils.renderNumerDenom(bigramRecallNum, bigramRecallDenom))
    if (!suppressOutput) Logger.logss(totalWordsUsed + " words used")
    evaluateRouge(settingsFile.getAbsolutePath, tmpDirAbsPath, rougeDirPath, skipRougeEvaluation, suppressOutput)
  }
  
  def evaluateRouge(settingsFileAbsPath: String, outDirAbsPath: String, rougeDirPath: String, skipScoring: Boolean, suppressOutput: Boolean): Array[Double] = {
    RougeFileMunger.writeSettings(settingsFileAbsPath, outDirAbsPath)
    if (!suppressOutput) Logger.logss("ROUGE OUTPUT: files written to " + outDirAbsPath + " with settings file in " + settingsFileAbsPath)
    if (!skipScoring) {
      import scala.sys.process._
      val output = Process(Seq(rougeDirPath + "/rouge-gillick.sh", settingsFileAbsPath)).lines;
      val lines = new ArrayBuffer[String]
      output.foreach(lines += _)
//      lines.foreach(Logger.logss(_))
      val nums = lines.map(line => {
        val startIdx = line.indexOf(":") + 2
        val endIdx = line.indexOf("(") - 1
        line.substring(startIdx, endIdx)
      })
      if (!suppressOutput) Logger.logss("ROUGE 1 P/R/F1: " + nums(1) + ", " + nums(0) + ", " + nums(2) + "; ROUGE 2 P/R/F1: " + nums(4) + ", " + nums(3) + ", " + nums(5))
      val numsDoubles = nums.map(_.toDouble)
      Array(numsDoubles(1), numsDoubles(0), numsDoubles(2), numsDoubles(4), numsDoubles(3), numsDoubles(5))
    } else {
      Logger.logss("...skipping evaluation, you have to run it yourself")
      Array[Double]()
    }
  }
  
  def bootstrap(worseSumms: Seq[Seq[String]],
                betterSumms: Seq[Seq[String]],
                refSumms: Seq[Seq[String]],
                rougeDirPath: String) {
    val size = worseSumms.size
    require(worseSumms.size == betterSumms.size)
    val worseSuffStats = (0 until size).map(i => evaluateRougeNonTok(Seq(worseSumms(i)), Seq(refSumms(i)), rougeDirPath, suppressOutput = true))
    val betterSuffStats = (0 until size).map(i => evaluateRougeNonTok(Seq(betterSumms(i)), Seq(refSumms(i)), rougeDirPath, suppressOutput = true))
    val numStats = worseSuffStats(0).size
    // Use macro-averages
    val origDiff = (0 until numStats).map(i => betterSuffStats.map(_(i)).foldLeft(0.0)(_ + _)/size - worseSuffStats.map(_(i)).foldLeft(0.0)(_ + _)/size)
    val bootstrapSamples = 10000
    val rng = new scala.util.Random(0)
    val numSig = Array.fill(numStats)(0)
    for (sampleIdx <- 0 until bootstrapSamples) {
      val resampled = (0 until size).map(i => rng.nextInt(size));
      val newDiff = (0 until numStats).map(i => resampled.map(idx => betterSuffStats(idx)(i)).foldLeft(0.0)(_ + _)/size - resampled.map(idx => worseSuffStats(idx)(i)).foldLeft(0.0)(_ + _)/size)
      for (i <- 0 until numStats) {
        if (origDiff(i) >= 0 && newDiff(i) < 2 * origDiff(i)) {
          numSig(i) += 1
        } 
      }
    }
    Logger.logss("ROUGE 1 P/R/F1; 2 P/R/F1: " + (0 until numSig.size).map(i => ClassifyUtils.renderNumerDenom(numSig(i), bootstrapSamples)))
  }
}
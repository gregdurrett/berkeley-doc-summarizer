package edu.berkeley.nlp.summ

import java.io.File
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.entity.GUtil
import edu.berkeley.nlp.entity.coref.MentionPropertyComputer
import edu.berkeley.nlp.entity.coref.NumberGenderComputer
import edu.berkeley.nlp.entity.coref.PairwiseScorer
import edu.berkeley.nlp.futile.LightRunner
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.fig.basic.SysInfoUtils
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.summ.data.DiscourseDepExProcessed
import edu.berkeley.nlp.summ.preprocess.DiscourseDependencyParser
import edu.berkeley.nlp.summ.data.SummaryAligner
import edu.berkeley.nlp.summ.data.SummDoc
import edu.berkeley.nlp.summ.preprocess.EDUSegmenter
import edu.berkeley.nlp.summ.data.DiscourseTreeReader

/**
 * Main class for training the summarizer on unlabeled data. See run-summarizer.sh for
 * example usage. The most useful arguments are:
 * -inputDir: directory of files (in CoNLL format, with parses/coref/NER) to summarize
 * -outputDir: directory to write summaries
 * -modelPath if you want to use a different version of the summarizer.
 * 
 * Any member of this class can be passed as a command-line argument to the
 * system if it is preceded with a dash, e.g.
 * -budget 100
 */
object Main {
  
  val trainDocsPath = "../summ-data/nyt08/trainsm_corefner"
  val trainAbstractsPath = "../summ-data/nyt08/train_proc_abstracts"
  val evalDocsPath = "../summ-data/nyt08/evalsm_corefner"
  val evalAbstractsPath = "../summ-data/nyt08/eval_proc_abstracts"
  
  val abstractsAreConll = false
  
  val noRst = false
  
  val evaluateOnRST = false
  val rstDocsPath = "data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES-PROC2/" 
  val rstTreesPath = "data/RSTDiscourse/data/RSTtrees-WSJ-main-1.0/ALL-FILES/"
  val rstSummsPath = "data/RSTDiscourse/data/summaries-processed/"
  
  val preprocDataPath = "preproc-cached.ser.gz"
  val modelPath = ""
  
  val numberGenderPath = "data/gender.data";
  val rougeDirPath = "rouge/ROUGE"
  
  val trainMax = -1
  val evalMax = -1
  
  val budgetScale = 1.0
  val minBudget = 0
  
  val runBigramCounts = false
  
  val printTKPResults = true
  
  val printSummaries = false
  val printSummariesForTurk = false
  
  val keepRougeDirs = false
  
  val featSpec = "position,basiclex,posnbasiclex,centrality,config,edushape,edulex,discourse,coref,replacement"
  val lexicalCountCutoff = 5
  
  
  val eduSegmenterPath = "models/edusegmenter.ser.gz"
  val discourseDepParserPath = "models/discoursedep.ser.gz"
  
  val useRSTDev = false 
  
  // DATA FILTERING ARGS -- Generally not be changed
  // Prune some additional stuff out of the training set -- achieves the right balance
  // between speed and using a lot of data
  val trainQualityThreshold = 0.5
  val trainMinSummLen = 50
  val trainMaxDocLen = 100
  // Keep all evaluation documents of length >= 50
  val evalQualityThreshold = 0.0
  val evalMinSummLen = 50
  val evalMaxDocLen = Int.MaxValue
  
  // OPTIMIZATION ARGS
  // These hyperparameters are all pretty good
  val parallel = false
  val numItrs = 10
  val eta = 0.1
  val reg = 1e-8
  val batchSize = 1
  
  // COMPRESSION ARGS
  // These shouldn't really have to change
  // Introduces syntactic compressions on top of EDUs
  val addSyntacticCompression = true
  // Uses syntactic compressions only
  val useSyntacticCompression = false
  
  val discourseType = "intra"
  
  val numEqualsConstraints = 2
  val numParentConstraints = 1
  
  // COREF/PRONOUN REPLACEMENT ARGUMENTS
  val corefModelPath = ""
  // These arguments should essentially never have to change unless you're doing experimentation on the system
  val useUnigramRouge = true
  val doPronounReplacement = true
  // More heavily constraint pronouns and don't allow replacement -- we don't
  // use this
  val doPronounConstraints = false
  val useFragilePronouns = true
  val replaceWithNEOnly = true
  val corefConfidenceThreshold = 0.8
  
  
  def main(args: Array[String]) {
    LightRunner.initializeOutput(Main.getClass())
    LightRunner.populateScala(Main.getClass(), args)
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(numberGenderPath);
    val mpc = new MentionPropertyComputer(Some(numberGenderComputer))
    ///////////////////
    // PREPROCESSING //
    ///////////////////
    // Do preprocessing from scratch
    Logger.startTrack("Preprocessing")
    val (trainExs, evalExs) = if (preprocDataPath == "" || !new File(preprocDataPath).exists()) {
      Logger.logss("Assembling train exs (requires EDU segmentation and parsing)...")
      val trainDocFilter = makeDocFilter(trainQualityThreshold, trainMinSummLen, trainMaxDocLen)
      val trainDocsRawPreshuffle = SummDoc.readSummDocs(trainDocsPath, trainAbstractsPath, abstractsAreConll, trainMax, mpc, true, true, trainDocFilter)
      val trainDocs = new scala.util.Random(0).shuffle(trainDocsRawPreshuffle)
//      val trainDocs = filterDocs(new scala.util.Random(0).shuffle(trainDocsRawPreshuffle), trainQualityThreshold, trainMinSummLen)
      val evalDocFilter = makeDocFilter(evalQualityThreshold, evalMinSummLen, evalMaxDocLen)
      val evalDocsRawPreshuffle = SummDoc.readSummDocs(evalDocsPath, evalAbstractsPath, abstractsAreConll, evalMax, mpc, true, true, evalDocFilter)
      // Shuffle dev too! We want the order to be random so that we can take a random subsample for evaluation
      val evalDocs = new scala.util.Random(0).shuffle(evalDocsRawPreshuffle)
//      val evalDocs = filterDocs(new scala.util.Random(0).shuffle(evalDocsRawPreshuffle), evalQualityThreshold, evalMinSummLen)
      Logger.logss("Kept " + trainDocs.size + " train and " + evalDocs.size + " eval docs")
      val overlapNames = trainDocs.map(_.name).toSet & evalDocs.map(_.name).toSet
      Logger.logss("Training set and eval set overlap on " + overlapNames.size + " docs")
      if (overlapNames.size > 0 && overlapNames.size < 100) {
        Logger.logss(overlapNames.toSeq.slice(0, Math.min(100, overlapNames.size)))
      }
      Logger.logss("Number of sentences longer than k on training set:")
      Logger.logss("30: " + trainDocs.filter(_.doc.size > 30).size)
      Logger.logss("50: " + trainDocs.filter(_.doc.size > 50).size)
      Logger.logss("100: " + trainDocs.filter(_.doc.size > 100).size)
      Logger.logss("150: " + trainDocs.filter(_.doc.size > 150).size)
      Logger.logss("200: " + trainDocs.filter(_.doc.size > 200).size)
      val time = System.nanoTime();
      val trainAndEvalExs = if (noRst) {
        val trainExs = new ArrayBuffer[DiscourseDepExProcessed] ++ trainDocs.par.map(doc => DiscourseDepExProcessed.makeTrivial(doc))
        val evalExs = new ArrayBuffer[DiscourseDepExProcessed] ++ evalDocs.par.map(doc => DiscourseDepExProcessed.makeTrivial(doc))
        trainExs -> evalExs
      } else if (useSyntacticCompression) {
        Logger.logss("Assembling compressions for train exs")
        val trainExs = new ArrayBuffer[DiscourseDepExProcessed] ++ trainDocs.par.map(doc => DiscourseDepExProcessed.makeWithSyntacticCompressions(doc))
        Logger.logss("Assembling compressions for test exs")
        val evalExs = new ArrayBuffer[DiscourseDepExProcessed] ++ evalDocs.par.map(doc => DiscourseDepExProcessed.makeWithSyntacticCompressions(doc))
        Logger.logss("Finished processing")
        trainExs -> evalExs
      } else {
        Logger.logss("Memory after loading documents: " + SysInfoUtils.getUsedMemoryStr());
        Logger.logss("Loading EDU segmenter...")
        val eduSegmenter = IOUtils.readObjFileHard(eduSegmenterPath).asInstanceOf[EDUSegmenter]
        Logger.logss("Loading parser...")
        val parser = IOUtils.readObjFileHard(discourseDepParserPath).asInstanceOf[DiscourseDependencyParser]
        Logger.logss("Segmenting and parsing train exs")
        val trainExs = new ArrayBuffer[DiscourseDepExProcessed] ++ trainDocs.par.map(doc => if (addSyntacticCompression) DiscourseDepExProcessed.makeWithEduAndSyntactic(doc, eduSegmenter, parser) else DiscourseDepExProcessed(doc, eduSegmenter, parser))
        Logger.logss("Segmenting and parsing test exs")
        val evalExs = new ArrayBuffer[DiscourseDepExProcessed] ++ evalDocs.par.map(doc => if (addSyntacticCompression) DiscourseDepExProcessed.makeWithEduAndSyntactic(doc, eduSegmenter, parser) else DiscourseDepExProcessed(doc, eduSegmenter, parser))
        Logger.logss("Finished processing")
        trainExs -> evalExs
      }
      Logger.logss("Writing to file...")
      if (preprocDataPath != "") {
        IOUtils.writeObjFileHard(preprocDataPath, trainAndEvalExs)
      }
      Logger.logss("Total millis for preprocessing = " + (System.nanoTime() - time)/1000000)
      trainAndEvalExs
    } else {
      // Load cached preprocessing
      Logger.logss("Reading cached preprocessed examples from " + preprocDataPath)
      val (trainExs, evalExs) = IOUtils.readObjFileHard(preprocDataPath).asInstanceOf[(ArrayBuffer[DiscourseDepExProcessed],ArrayBuffer[DiscourseDepExProcessed])]
      Logger.logss("Done!")
      trainExs -> evalExs
    }
    Logger.logss(trainExs.size + " train and " + evalExs.size + " test examples; " + trainExs.foldLeft(0)(_ + _.eduAlignments.size) + " total EDUs in train")
    for (i <- 0 until Math.min(10, trainExs.size)) {
      Logger.logss(trainExs(i).parentLabels.toSeq)
    }
    Logger.endTrack()
    
    //////////////
    // TRAINING //
    //////////////
    
    // FEATURIZATION
    val featIndexer = new Indexer[String]
    val trainWordCounts = new Counter[String]
    trainExs.foreach(_.rawDoc.doc.foreach(_.getWords.foreach(trainWordCounts.incrementCount(_, 1.0))))
    val featurizer = new CompressiveAnaphoraFeaturizer(featIndexer, featSpec.split(",").toSet, discourseType, trainWordCounts, lexicalCountCutoff)
    Logger.startTrack("Extracting features on train...")
    for (i <- 0 until trainExs.size) {
      if (trainExs.size >= 10 && i % (trainExs.size/10) == 0) {
        Logger.logss("On example " + i + "/" + trainExs.size)
      }
      featurizer.extractFeaturesCached(trainExs(i), true)
    }
    Logger.logss("Memory after featurization: " + SysInfoUtils.getUsedMemoryStr());
    val maybeCorefModel = if (corefModelPath != "") Some(GUtil.load(corefModelPath).asInstanceOf[PairwiseScorer]) else None
    if (doPronounReplacement) {
      Logger.logss("Pronoun featurization")
      for (i <- 0 until trainExs.size) {
        if (trainExs.size >= 10 && i % (trainExs.size/10) == 0) {
          Logger.logss("On example " + i + "/" + trainExs.size)
        }
        featurizer.extractPronounFeaturesCached(trainExs(i), trainExs(i).identifyPronounReplacements(replaceWithNEOnly, maybeCorefModel, corefConfidenceThreshold), true)
      }
    } else {
      None
    }
    if (featSpec.contains("ngramtype")) {
      Logger.logss("N-gram featurization")
      for (i <- 0 until trainExs.size) {
        if (trainExs.size >= 10 && i % (trainExs.size/10) == 0) {
          Logger.logss("On example " + i + "/" + trainExs.size)
        }
        featurizer.extractBigramFeaturesCached(trainExs(i), trainExs(i).getDocBigramsSeq(useUnigramRouge), true)
      }
    }
    Logger.endTrack()
    // INSTANTIATING AND TRAINING SUMMARIZER
    val computer = new CompressiveAnaphoraSummarizerComputer(featurizer, -1, budgetScale, discourseType, numEqualsConstraints, numParentConstraints,
                                                                  doPronounReplacement, doPronounConstraints, useFragilePronouns, replaceWithNEOnly, maybeCorefModel,
                                                                  corefConfidenceThreshold, useUnigramRouge)
    val initialWeights = computer.getInitialWeights(0.01)
    Logger.logss(initialWeights.size + " total sparse features")
    val weights = new GeneralTrainer(parallel).trainAdagradSparse(trainExs, computer, eta, reg, batchSize, numItrs, initialWeights, verbose = true);
    val summarizer = new CompressiveAnaphoraSummarizer(computer, new AdagradWeightVector(weights, 0, 0))
    
    if (modelPath != "") {
      Logger.logss("Wrote model to " + modelPath)
      IOUtils.writeObjFileHard(modelPath, summarizer)
    }
    
    /////////////////
    // EVALUATION //
    ////////////////
    
    evaluate(evalExs, summarizer, printSummaries)
    
    // RST EVALUATION
    if (evaluateOnRST) {
      Logger.logss("RST EVALUATION")
      val exs = DiscourseTreeReader.readAllAlignAndFilter(rstDocsPath, rstTreesPath, mpc).sortBy(_.rawDoc.corefDoc.rawDoc.docID)
      // N.B. Shuffled now
      val testExs = new scala.util.Random(0).shuffle(exs.filter(ex => DiscourseDependencyParser.summDocNames.contains(ex.discourseTreeFileName)))
      val exNames = exs.map(_.discourseTreeFileName)
      val badTestExs = DiscourseDependencyParser.summDocNames.filter(name => !exNames.contains(name))
      Logger.logss("Bad test exs: " + badTestExs)
      val refSummaries: Seq[Seq[Seq[String]]] = testExs.map(ex => RougeComputer.readSummDocJustTextBySents(rstSummsPath + "/" + ex.discourseTreeFileName.dropRight(8) + ".short-abs"))
      val refSummariesDetok: Seq[Seq[String]] = refSummaries.map(_.map(sent => sent.reduce(_ + " " + _)))
      val firstKSummaries = new ArrayBuffer[Seq[String]]
      val firstKWordsSummaries = new ArrayBuffer[Seq[String]]
      val hiraoSummariesGoldEdus = new ArrayBuffer[Seq[String]]
      val sysSummariesGoldEdus = new ArrayBuffer[Seq[String]]
      val hiraoSummaries = new ArrayBuffer[Seq[String]]
      val sysSummaries = new ArrayBuffer[Seq[String]]
      Logger.logss("Loading EDU segmenter...")
      val eduSegmenter = IOUtils.readObjFileHard(eduSegmenterPath).asInstanceOf[EDUSegmenter]
      Logger.logss("Loading parser...")
      val parser = IOUtils.readObjFileHard(discourseDepParserPath).asInstanceOf[DiscourseDependencyParser]
      for (idx <- 0 until testExs.size) {
        if (testExs.size >= 10 && idx % (testExs.size/10) == 0) {
          Logger.logss("On example " + idx + "/" + testExs.size)
        }
        val rawTestEx = testExs(idx)
        val budget = refSummaries(idx).map(_.size).foldLeft(0)(_ + _)
        TreeKnapsackSummarizer.summarizeFirstK(testExs(idx).getLeaves, budget)
        
        val evalExGoldEdusTKP = DiscourseDepExProcessed(rawTestEx.rawDoc, rawTestEx.eduAlignments, parser)
        val evalExGoldEdus = if (addSyntacticCompression) {
          DiscourseDepExProcessed.makeWithEduAndSyntactic(rawTestEx.rawDoc, rawTestEx.eduAlignments, evalExGoldEdusTKP.parents, evalExGoldEdusTKP.parentLabels)
        } else {
          evalExGoldEdusTKP
        }
        
//        val sysSummaryGoldEdus = evalExGoldEdus.getSummaryText(computer.decode(evalExGoldEdus, wrappedWeights, budget))
        val sysSummaryGoldEdus = summarizer.summarize(evalExGoldEdus, budget, false)
        sysSummariesGoldEdus += sysSummaryGoldEdus
        
        val evalExTKP = DiscourseDepExProcessed(rawTestEx.rawDoc, eduSegmenter, parser)
        val evalEx = if (addSyntacticCompression) {
          DiscourseDepExProcessed.makeWithEduAndSyntactic(rawTestEx.rawDoc, evalExTKP.eduAlignments, evalExTKP.parents, evalExTKP.parentLabels)
        } else {
          evalExTKP
        }
        val sysSummary = summarizer.summarize(evalEx, budget, false)
//        val sysSummary = evalEx.getSummaryText(computer.decode(evalEx, wrappedWeights, budget))
        sysSummaries += sysSummary
        firstKSummaries += evalExGoldEdus.getSummaryText(getFirstK(evalExGoldEdus, budget), false)
        firstKWordsSummaries += evalExGoldEdus.getFirstKWords(budget, false)
        
        val leafWordsSysEdus = evalEx.eduAlignments.map(edu => evalEx.rawDoc.corefDoc.rawDoc.words(edu._1._1).slice(edu._1._2, edu._2._2))
        val hirao = TreeKnapsackSummarizer.summarizeILP(leafWordsSysEdus.map(_.size), evalEx.parents, budget, TreeKnapsackSummarizer.computeEduValuesUseStopwordSet(leafWordsSysEdus, evalEx.parents), false)._1
        hiraoSummaries += TreeKnapsackSummarizer.extractSummary(leafWordsSysEdus, hirao)
        
        val leafWordsGoldEdus = evalExGoldEdusTKP.eduAlignments.map(edu => evalExGoldEdusTKP.rawDoc.corefDoc.rawDoc.words(edu._1._1).slice(edu._1._2, edu._2._2))
        val hiraoGoldEdus = TreeKnapsackSummarizer.summarizeILP(leafWordsGoldEdus.map(_.size), evalExGoldEdusTKP.parents, budget, TreeKnapsackSummarizer.computeEduValuesUseStopwordSet(leafWordsGoldEdus, evalExGoldEdusTKP.parents), false)._1
        hiraoSummariesGoldEdus += TreeKnapsackSummarizer.extractSummary(leafWordsGoldEdus, hiraoGoldEdus)
      }
      if (useRSTDev) {
        val numDevDocs = 10
        def evaluateOnDevAndTest(summs: Seq[Seq[String]], refs: Seq[Seq[String]]) {
          RougeComputer.evaluateRougeNonTok(summs.slice(0, numDevDocs), refs.slice(0, 10), rougeDirPath, false, false, keepRougeDirs)
          RougeComputer.evaluateRougeNonTok(summs.slice(numDevDocs, refs.size), refs.slice(10, refs.size), rougeDirPath, false, false, keepRougeDirs)
        }
        Logger.logss("FIRST K EDUS")
        evaluateOnDevAndTest(firstKSummaries, refSummariesDetok)
        Logger.logss("FIRST K WORDS")
        evaluateOnDevAndTest(firstKWordsSummaries, refSummariesDetok)
        Logger.logss("HIRAO WITH GOLD EDUS")
        evaluateOnDevAndTest(hiraoSummariesGoldEdus, refSummariesDetok)
        Logger.logss("SYSTEM WITH GOLD EDUS")
        evaluateOnDevAndTest(sysSummariesGoldEdus, refSummariesDetok)
        Logger.logss("STAT SIG")
        RougeComputer.bootstrap(hiraoSummariesGoldEdus, sysSummariesGoldEdus, refSummariesDetok, rougeDirPath)
        RougeComputer.bootstrap(sysSummariesGoldEdus, hiraoSummariesGoldEdus, refSummariesDetok, rougeDirPath)
        Logger.logss("HIRAO")
        evaluateOnDevAndTest(hiraoSummaries, refSummariesDetok)
        Logger.logss("SYSTEM")
        evaluateOnDevAndTest(sysSummaries, refSummariesDetok)
      } else {
        Logger.logss("FIRST K EDUS")
        RougeComputer.evaluateRougeNonTok(firstKSummaries, refSummariesDetok, rougeDirPath, false, keepRougeDirs)
        Logger.logss("FIRST K WORDS")
        RougeComputer.evaluateRougeNonTok(firstKWordsSummaries, refSummariesDetok, rougeDirPath, false, keepRougeDirs)
        Logger.logss("HIRAO WITH GOLD EDUS")
        RougeComputer.evaluateRougeNonTok(hiraoSummariesGoldEdus, refSummariesDetok, rougeDirPath, false, keepRougeDirs)
        Logger.logss("SYSTEM WITH GOLD EDUS")
        RougeComputer.evaluateRougeNonTok(sysSummariesGoldEdus, refSummariesDetok, rougeDirPath, false, keepRougeDirs)
        Logger.logss("STAT SIG")
        RougeComputer.bootstrap(firstKSummaries, sysSummariesGoldEdus, refSummariesDetok, rougeDirPath)
        Logger.logss("HIRAO")
        RougeComputer.evaluateRougeNonTok(hiraoSummaries, refSummariesDetok, rougeDirPath, false, keepRougeDirs)
        Logger.logss("SYSTEM")
        RougeComputer.evaluateRougeNonTok(sysSummaries, refSummariesDetok, rougeDirPath, false, keepRougeDirs)
      }
    }
    
    
    
    
    
    LightRunner.finalizeOutput() 
  }
  
  def makeDocFilter(alignedFraction: Double, minSummLen: Int, maxDocLen: Int) = {
    (doc: SummDoc) => {
      val alignment = SummaryAligner.alignDocAndSummary(doc, false);
      val isAlignmentGood = alignment.filter(_ != -1).size >= alignment.size * alignedFraction 
      val isSummLenGood = doc.summary.map(_.size).foldLeft(0)(_ + _) >= minSummLen
      val isDocLenGood = doc.doc.size <= maxDocLen
      isAlignmentGood && isSummLenGood && isDocLenGood
    }
  }
  
  ////////////////
  // EVALUATION //
  ////////////////
  def evaluate(evalExs: Seq[DiscourseDepExProcessed], summarizer: DiscourseDepExSummarizer, printSummaries: Boolean) {
    // Evaluation
    val refSummaries = new ArrayBuffer[Seq[String]]
    val firstKSentsSummaries = new ArrayBuffer[Seq[String]]
    val firstKEdusSummaries = new ArrayBuffer[Seq[String]]
    val firstKWordsSummaries = new ArrayBuffer[Seq[String]]
    val sysSummaries = new ArrayBuffer[Seq[String]]
    val bigramCountSummaries = new ArrayBuffer[Seq[String]]
    val tkpSummaries = new ArrayBuffer[Seq[String]]
    val bigramCountSummarizer = new BigramCountSummarizer()
    var totalBudget = 0
    var numInFirstK = 0
    var totalEdusSelected = 0
    for (evalEx <- evalExs.sortBy(_.rawDoc.name)) {
//    for (evalEx <- evalExs) {
      Logger.logss("Decoding " + evalEx.rawDoc.name)
      val refSummary = evalEx.rawDoc.summSents
      val refLen = refSummary.map(_.size).foldLeft(0)(_ + _)
      val budget = Math.max(minBudget, (refLen * budgetScale).toInt)
      totalBudget += budget
      val refSummaryNonTok = refSummary.map(_.reduce(_ + " " + _))
      refSummaries += refSummaryNonTok
      val firstKSentsSummaryNonTok = evalEx.getSummaryText(getFirstKSentences(evalEx, budget))
      firstKSentsSummaries += firstKSentsSummaryNonTok
      val firstKEdusSummaryNonTok = evalEx.getSummaryText(getFirstK(evalEx, budget))
      firstKEdusSummaries += firstKEdusSummaryNonTok
      val firstKWordsSummaryNonTok = evalEx.getFirstKWords(budget)
      firstKWordsSummaries += firstKWordsSummaryNonTok
      val sysSummaryNonTok = summarizer.summarize(evalEx, budget)
      sysSummaries += sysSummaryNonTok
      if (runBigramCounts) {
        val bigramCountSummaryNonTok = bigramCountSummarizer.summarize(evalEx, budget)
        bigramCountSummaries += bigramCountSummaryNonTok
      }
      if (printTKPResults) {
        val parents = evalEx.getParents(discourseType)
        val leafWords = evalEx.eduAlignments.map(edu => evalEx.rawDoc.doc(edu._1._1).getWords.slice(edu._1._2, edu._2._2).toSeq)
        val leafPoss = evalEx.eduAlignments.map(edu => evalEx.rawDoc.doc(edu._1._1).getPoss.slice(edu._1._2, edu._2._2).toSeq)
//        val eduScores = DiscourseSummarizer.computeEduValuesUseStopwordSet(leafWords, parents)
        val eduScores = TreeKnapsackSummarizer.computeEduValuesUsePoss(leafWords, leafPoss, parents)
        val tkpEdus = TreeKnapsackSummarizer.summarizeILP(evalEx.leafSizes, parents, evalEx.parentLabels, budget, eduScores, 1, 1)._1
        val tkpSummaryNonTok = evalEx.getSummaryText(tkpEdus)
        tkpSummaries += tkpSummaryNonTok
      }
      if (printSummaries) {
        Logger.logss("------------------")
        Logger.logss("FIRST K EDUS")
        for (j <- 0 until firstKEdusSummaryNonTok.size) {
          Logger.logss(j + ": " + firstKEdusSummaryNonTok(j))
        }
        Logger.logss("FIRST K WORDS")
        for (j <- 0 until firstKWordsSummaryNonTok.size) {
          Logger.logss(j + ": " + firstKWordsSummaryNonTok(j))
        }
        Logger.logss("SYSTEM")
        summarizer.display(evalEx, budget)
        Logger.logss("REFERENCE")
        refSummary.foreach(lineWords => Logger.logss(lineWords.foldLeft("")(_ + " " + _).trim))
      }
      if (printSummariesForTurk) {
        Logger.logss("FIRST K SENTS: " + RougeFileMunger.detokenizeSentence(firstKSentsSummaryNonTok.foldLeft("")(_ + " " + _).trim))
        Logger.logss("FIRST K EDUS: " + RougeFileMunger.detokenizeSentence(firstKEdusSummaryNonTok.foldLeft("")(_ + " " + _).trim))
        Logger.logss("FIRST K WORDS: " + RougeFileMunger.detokenizeSentence(firstKWordsSummaryNonTok.foldLeft("")(_ + " " + _).trim))
        if (runBigramCounts) {
          Logger.logss("BIGRAM COUNTS: " + RougeFileMunger.detokenizeSentence(bigramCountSummaries.last.foldLeft("")(_ + " " + _).trim))
        }
        Logger.logss("SYSTEM: " + RougeFileMunger.detokenizeSentence(sysSummaryNonTok.foldLeft("")(_ + " " + _).trim))
        Logger.logss("REFERENCE: " + RougeFileMunger.detokenizeSentence(refSummaryNonTok.foldLeft("")(_ + " " + _).trim))
        if (printTKPResults) {
          Logger.logss("TKP: " + RougeFileMunger.detokenizeSentence(tkpSummaries.last.foldLeft("")(_ + " " + _)))
        }
      }
//      Logger.logss(firstKSummary.map(_.size).foldLeft(0)(_ + _) + " in first-k, " + sysSummary.map(_.size).foldLeft(0)(_ + _) + " in sys, budget = " + budget)
    }
    Logger.logss("Total budget: " + totalBudget)
    if (printTKPResults) {
      Logger.logss("TKP RESULTS")
      RougeComputer.evaluateRougeNonTok(tkpSummaries, refSummaries, rougeDirPath, false, keepRougeDirs)
    }
    Logger.logss("FIRST K SENTS VS REFERENCE")
    RougeComputer.evaluateRougeNonTok(firstKSentsSummaries, refSummaries, rougeDirPath, false, keepRougeDirs)
    Logger.logss("FIRST K EDUS VS REFERENCE")
    RougeComputer.evaluateRougeNonTok(firstKEdusSummaries, refSummaries, rougeDirPath, false, keepRougeDirs)
    Logger.logss("FIRST K WORDS VS REFERENCE")
    RougeComputer.evaluateRougeNonTok(firstKWordsSummaries, refSummaries, rougeDirPath, false, keepRougeDirs)
    if (runBigramCounts) {
      Logger.logss("BIGRAM COUNTS VS REFERENCE")
      RougeComputer.evaluateRougeNonTok(bigramCountSummaries, refSummaries, rougeDirPath, false, keepRougeDirs)
    }
    summarizer.printStatistics()
    Logger.logss("SYSTEM VS REFERENCE")
    RougeComputer.evaluateRougeNonTok(sysSummaries, refSummaries, rougeDirPath, false, keepRougeDirs)
  }
  
  def getFirstK(ex: DiscourseDepExProcessed, budget: Int): Seq[Int] = {
    var currSize = 0
    var stoppingPoint = 0
    while (stoppingPoint < ex.leafSizes.size && currSize + ex.leafSizes(stoppingPoint) <= budget) {
      currSize += ex.leafSizes(stoppingPoint)
      stoppingPoint += 1
    }
    0 until stoppingPoint
  }
  
  def getFirstKSentences(ex: DiscourseDepExProcessed, budget: Int): Seq[Int] = {
    val firstK = getFirstK(ex, budget)
    if (firstK.isEmpty) {
      firstK
    } else {
      val firstKLastEdu = firstK.last
      if (firstKLastEdu == ex.eduAlignments.size - 1) {
        firstK
      } else {
        // Walk back until we find an EDU that finishes a sentence
        var currEdu = firstKLastEdu
        while (currEdu >= 0 && ex.eduAlignments(currEdu)._1._1 == ex.eduAlignments(currEdu+1)._1._1) {
          currEdu -= 1
        }
        0 until currEdu + 1
      }
    }
  }
}
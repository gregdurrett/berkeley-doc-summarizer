package edu.berkeley.nlp.summ

import java.util.Arrays

import scala.collection.JavaConverters.iterableAsScalaIterableConverter
import scala.util.Random

import edu.berkeley.nlp.futile.fig.basic.SysInfoUtils
import edu.berkeley.nlp.futile.math.CachingDifferentiableFunction
import edu.berkeley.nlp.futile.math.LBFGSMinimizer
import edu.berkeley.nlp.futile.util.IntCounter
import edu.berkeley.nlp.futile.util.Logger

trait LikelihoodAndGradientComputer[T] {
  
  def getInitialWeights(initialWeightsScale: Double): Array[Double]
  
  /**
   * Accumulates the gradient on this example into gradient and returns the log likelihood
   * of this example
   */
  def accumulateGradientAndComputeObjective(ex: T, weights: Array[Double], gradient: Array[Double]): Double;
  
  /**
   * Just computes the objective; lighter-weight method that clients may want to implement
   * more efficiently
   */
  def computeObjective(ex: T, weights: Array[Double]): Double;
  
  /**
   * Allows for modification of the weights to do things like clipping or printing
   */
  def weightsUpdateCallback(weights: Array[Double]) = {}
  
  /**
   * Allows for modification of the weights to do things like clipping or printing
   */
  def iterationEndCallback(weights: Array[Double]) = {}
}

trait LikelihoodAndGradientComputerSparse[T] {
  
  def getInitialWeights(initialWeightsScale: Double): Array[Double]
  
  /**
   * Accumulates the gradient on this example into gradient and returns the log likelihood
   * of this example
   */
  def accumulateGradientAndComputeObjective(ex: T, weights: AdagradWeightVector, gradient: IntCounter): Double
  
  /**
   * Just computes the objective; lighter-weight method that clients may want to implement
   * more efficiently
   */
  def computeObjective(ex: T, weights: AdagradWeightVector): Double;
  
  /**
   * Allows for modification of the weights to do things like clipping or printing
   */
  def weightsUpdateCallback(weights: AdagradWeightVector) = {}
  
  /**
   * Allows for modification of the weights to do things like clipping or printing
   */
  def iterationEndCallback(weights: AdagradWeightVector) = {}
}

/**
 * N.B. This is not threadsafe currently! access() will potentially have race
 * conditions.
 */
@SerialVersionUID(1L)
class AdagradWeightVector(val weights: Array[Double],
                          val lambda: Double,
                          val eta: Double) extends Serializable {
  var nanos = 0L
  val lastIterTouched = Array.tabulate(weights.size)(i => 0)
  var currIter = 0
  val diagGt = Array.tabulate(weights.size)(i => 0.0)
  
  def applyGradientUpdate(gradient: IntCounter, batchSize: Int) {
//    Logger.logss(gradient.toString)
    // Precompute this so dividing by batch size is a multiply and not a divide
    val batchSizeMultiplier = 1.0/batchSize;
    currIter += 1
    // TODO: Potentially optimizable
    for (key <- gradient.keySet.asScala) {
      val i = key.intValue
      val xti = weights(i);
      // N.B. We negate the gradient here because the Adagrad formulas are all for minimizing
      // and we're trying to maximize, so think of it as minimizing the negative of the objective
      // which has the opposite gradient
      // Equation (25) in http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf
      // eta is the step size, lambda is the regularization
      val gti = -gradient.get(i) * batchSizeMultiplier;
      // Update diagGt
      val oldEtaOverHtii = eta / (1 + Math.sqrt(diagGt(i)).toDouble)
      diagGt(i) += gti * gti;
      val Htii = 1 + Math.sqrt(diagGt(i)).toDouble;
      // Avoid divisions at all costs...
      val etaOverHtii = eta / Htii;
      val newXti = xti - etaOverHtii * gti;
      // Apply the regularizer for every iteration since touched
      val itersSinceTouched = currIter - lastIterTouched(i)
      lastIterTouched(i) = currIter
      weights(i) = Math.signum(newXti) * Math.max(0, Math.abs(newXti) - lambda * etaOverHtii - (itersSinceTouched - 1) * lambda * oldEtaOverHtii);
    }
  }
  
  def access(i: Int) = {
    if (lastIterTouched(i) != currIter) {
      val xti = weights(i)
      val Htii = 1 + Math.sqrt(diagGt(i)).toDouble;
      val etaOverHtii = eta / Htii;
      val itersSinceTouched = currIter - lastIterTouched(i)
      lastIterTouched(i) = currIter
      weights(i) = Math.signum(xti) * Math.max(0, Math.abs(xti) - itersSinceTouched * lambda * eta * etaOverHtii);
    }
    weights(i)
  }
  
  def score(feats: Array[Int]) = scoreWithPosnOffset(feats, 0)
  
  def scoreWithPosnOffset(feats: Array[Int], offset: Int) = {
    var i = 0
    var score = 0.0
    while (i < feats.size) {
      score += access(feats(i) + offset)
      i += 1
    }
    score
  }
  
  def finalizeWeights: Array[Double] = Array.tabulate(weights.size)(i => access(i))
}

class GeneralTrainer[T](val parallel: Boolean = false) {
  
  var inferenceNanos = 0L;
  var adagradNanos = 0L;
  
  
  
  def displayWeights(weights: Array[Double]) {
    Logger.logss("NONZERO WEIGHTS: " + weights.foldRight(0)((weight, count) => if (Math.abs(weight) > 1e-15) count + 1 else count));
    Logger.logss("WEIGHT VECTOR NORM: " + weights.foldRight(0.0)((weight, norm) => norm + weight * weight));
  }
  
  def displayTime(iter: Int, startTime: Long, inferenceNanos: Long, adagradNanos: Long) {
    Logger.logss("MILLIS FOR ITER " + iter + ": " + (System.nanoTime() - startTime) / 1000000.0 +
              " (" + inferenceNanos / 1000000.0 + " for inference and " + adagradNanos / 1000000.0 + " for Adagrad)");
    Logger.logss("MEMORY AFTER ITER " + iter + ": " + SysInfoUtils.getUsedMemoryStr());
  }
  
  def computeObjectiveL1R(trainExs: Seq[T],
                          computer: LikelihoodAndGradientComputer[T],
                          weights: Array[Double],
                          lambda: Double): Double = {
    var objective = (if (parallel) trainExs.par else trainExs).aggregate(0.0)((currLL, ex) => currLL + computer.computeObjective(ex, weights), _ + _)
    objective + computeRegularizationTermL1R(weights, lambda)
  }
  
  def computeObjectiveL1RSparse(trainExs: Seq[T],
                                computer: LikelihoodAndGradientComputerSparse[T],
                                weights: AdagradWeightVector,
                                lambda: Double): Double = {
    var objective = (if (parallel) trainExs.par else trainExs).aggregate(0.0)((currLL, ex) => currLL + computer.computeObjective(ex, weights), _ + _)
    objective + computeRegularizationTermL1R(weights.weights, lambda)
  }
  
  def computeRegularizationTermL1R(weights: Array[Double], lambda: Double): Double = {
    var regTerm = 0.0
    for (weight <- weights) {
      regTerm -= lambda * Math.abs(weight);
    }
    regTerm;
  }
  
  ///////////////////////////
  // MINIBATCH COMPUTATION //
  ///////////////////////////
  
  def getMinibatchObjectiveAndGradient(exs: Seq[T], computer: LikelihoodAndGradientComputer[T], weights: Array[Double], gradientArray: Array[Double]) = {
    var nanoTime = System.nanoTime();
    val objective = if (parallel) {
      parallelGetMinibatchObjectiveAndGradient(exs, computer, weights, gradientArray)
    } else {
      serialGetMinibatchObjectiveAndGradient(exs, computer, weights, gradientArray)
    }
    inferenceNanos += (System.nanoTime() - nanoTime);
    objective
  }
  
  private def serialGetMinibatchObjectiveAndGradient(exs: Seq[T], computer: LikelihoodAndGradientComputer[T], weights: Array[Double], gradientArray: Array[Double]) = {
    var objective = 0.0
    for (ex <- exs) {
      objective += computer.accumulateGradientAndComputeObjective(ex, weights, gradientArray);
    }
    objective
  }
  
  case class SuffStats(var ll: Double, val gradient: Array[Double]) {
    def incrementLL(increment: Double) { ll += increment }
    def +=(other: SuffStats) = {
      ll += other.ll
      var i = 0
      while (i < gradient.size) {
        gradient(i) += other.gradient(i)
        i += 1
      }
      this
    }
  }
  
  def parallelGetMinibatchObjectiveAndGradient(exs: Seq[T], computer: LikelihoodAndGradientComputer[T], weights: Array[Double], gradientArray: Array[Double]) = {
    val finalSS = exs.par.aggregate(null: SuffStats)((currSS, ex) => {
      val ss = if (currSS ne null) currSS else new SuffStats(0.0, Array.tabulate(gradientArray.size)(i => 0.0))
      val ll = computer.accumulateGradientAndComputeObjective(ex, weights, ss.gradient);
      ss.incrementLL(ll)
      ss
    }, { (a, b) => if (a eq null) b else if (b eq null) a else b += a })
    System.arraycopy(finalSS.gradient, 0, gradientArray, 0, gradientArray.size)
    finalSS.ll
  }
  
  
  //////////////////////////////////
  // SPARSE MINIBATCH COMPUTATION //
  //////////////////////////////////
  
  
  def getMinibatchObjectiveAndGradientSparse(exs: Seq[T], computer: LikelihoodAndGradientComputerSparse[T], weights: AdagradWeightVector, gradientCounter: IntCounter) = {
    var nanoTime = System.nanoTime();
    val objective = if (parallel) {
      parallelGetMinibatchObjectiveAndGradientSparse(exs, computer, weights, gradientCounter)
    } else {
      serialGetMinibatchObjectiveAndGradientSparse(exs, computer, weights, gradientCounter)
    }
    inferenceNanos += (System.nanoTime() - nanoTime);
    objective
  }
  
  private def serialGetMinibatchObjectiveAndGradientSparse(exs: Seq[T], computer: LikelihoodAndGradientComputerSparse[T], weights: AdagradWeightVector, gradientCounter: IntCounter) = {
    var objective = 0.0
    for (ex <- exs) {
      objective += computer.accumulateGradientAndComputeObjective(ex, weights, gradientCounter);
    }
    objective
  }
  
  case class SuffStatsSparse(var ll: Double, val gradient: IntCounter) {
    def incrementLL(increment: Double) { ll += increment }
    def +=(other: SuffStatsSparse) = {
      ll += other.ll
      var i = 0
      while (i < gradient.size) {
        gradient.incrementAll(other.gradient)
        i += 1
      }
      this
    }
  }
  
  def parallelGetMinibatchObjectiveAndGradientSparse(exs: Seq[T], computer: LikelihoodAndGradientComputerSparse[T], weights: AdagradWeightVector, gradientCounter: IntCounter) = {
    val finalSS = exs.par.aggregate(null: SuffStatsSparse)((currSS, ex) => {
      val ss = if (currSS ne null) currSS else new SuffStatsSparse(0.0, new IntCounter)
      val ll = computer.accumulateGradientAndComputeObjective(ex, weights, ss.gradient);
      ss.incrementLL(ll)
      ss
    }, { (a, b) => if (a eq null) b else if (b eq null) a else b += a })
    gradientCounter.incrementAll(finalSS.gradient)
    finalSS.ll
  }
  
  /////////////
  // ADAGRAD //
  /////////////

  def trainAdagrad(trainExs: Seq[T],
                   computer: LikelihoodAndGradientComputer[T],
                   eta: Double,
                   lambda: Double,
                   batchSize: Int,
                   numItrs: Int,
                   initialWeights: Array[Double],
                   verbose: Boolean = true): Array[Double] = {
//    val weights = Array.fill(pairwiseIndexingFeaturizer.featureIndexer.size)(0.0);
    val weights = initialWeights;
    val reusableGradientArray = Array.fill(initialWeights.size)(0.0);
    val diagGt = Array.fill(initialWeights.size)(0.0);
    for (i <- 0 until numItrs) {
      Logger.logss("ITERATION " + i);
      val startTime = System.nanoTime();
      inferenceNanos = 0;
      adagradNanos = 0;
      if (verbose) Logger.startTrack("Computing gradient");
      var cumulativeObjective = 0.0
      var currIdx = 0;
      var currBatchIdx = 0;
      val printFreq = (trainExs.size / batchSize) / 10 // Print progress 10 times per pass through the data
      while (currIdx < trainExs.size) {
        if (verbose && (printFreq == 0 || currBatchIdx % printFreq == 0)) {
          Logger.logs("Computing gradient on " + currIdx + " (batch " + currBatchIdx + " / " + (trainExs.size / batchSize) + ")");
        }
        cumulativeObjective += takeAdagradStepL1R(trainExs.slice(currIdx, Math.min(trainExs.size, currIdx + batchSize)),
                                                  computer,
                                                  weights,
                                                  reusableGradientArray,
                                                  diagGt,
                                                  eta,
                                                  lambda);
        computer.weightsUpdateCallback(weights)
        currIdx += batchSize;
        currBatchIdx += 1;
      }
//      for (weight <- weights) {
//        cumulativeObjective -= lambda * Math.abs(weight);
//      }
      cumulativeObjective += computeRegularizationTermL1R(weights, lambda)
      Logger.logss("APPROXIMATE OBJECTIVE: " + cumulativeObjective + " (avg = " + cumulativeObjective/trainExs.size + ")")
      if (verbose) {
        Logger.endTrack();
        displayWeights(weights)
        displayTime(i, startTime, inferenceNanos, adagradNanos)
      }
      computer.iterationEndCallback(weights)
    }
    if (verbose) {
      Logger.logss("FINAL TRAIN OBJECTIVE: " + computeObjectiveL1R(trainExs, computer, weights, lambda));
    }
    weights
  }

  def takeAdagradStepL1R(exs: Seq[T],
                         computer: LikelihoodAndGradientComputer[T],
                         weights: Array[Double],
                         reusableGradientArray: Array[Double],
                         diagGt: Array[Double],
                         eta: Double,
                         lambda: Double): Double = {
    Arrays.fill(reusableGradientArray, 0.0);
    val objective = getMinibatchObjectiveAndGradient(exs, computer, weights, reusableGradientArray)
    val nanoTime = System.nanoTime();
    // Precompute this so dividing by batch size is a multiply and not a divide
    val batchSizeMultiplier = 1.0/exs.size;
    var i = 0;
    while (i < reusableGradientArray.size) {
      val xti = weights(i);
      // N.B. We negate the gradient here because the Adagrad formulas are all for minimizing
      // and we're trying to maximize, so think of it as minimizing the negative of the objective
      // which has the opposite gradient
      // Equation (25) in http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf
      // eta is the step size, lambda is the regularization
      val gti = -reusableGradientArray(i) * batchSizeMultiplier;
      // Update diagGt
      diagGt(i) += gti * gti;
      val Htii = 1 + Math.sqrt(diagGt(i)).toDouble;
      // Avoid divisions at all costs...
      val etaOverHtii = eta / Htii;
      val newXti = xti - etaOverHtii * gti;
      weights(i) = Math.signum(newXti) * Math.max(0, Math.abs(newXti) - lambda * etaOverHtii);
      i += 1;
    }
    adagradNanos += (System.nanoTime() - nanoTime);
    objective
  }
  
  
  ////////////////////
  // SPARSE ADAGRAD //
  ////////////////////

  
  def trainAdagradSparse(trainExs: Seq[T],
                         computer: LikelihoodAndGradientComputerSparse[T],
                         eta: Double,
                         lambda: Double,
                         batchSize: Int,
                         numItrs: Int,
                         initialWeights: Array[Double],
                         verbose: Boolean = true): Array[Double] = {
    val weights = new AdagradWeightVector(initialWeights, lambda, eta);
    for (i <- 0 until numItrs) {
      Logger.logss("ITERATION " + i);
      val startTime = System.nanoTime();
      inferenceNanos = 0;
      adagradNanos = 0;
      if (verbose) Logger.startTrack("Computing gradient");
      var cumulativeObjective = 0.0
      var currIdx = 0;
      var currBatchIdx = 0;
      val printFreq = (trainExs.size / batchSize) / 10 // Print progress 10 times per pass through the data
      while (currIdx < trainExs.size) {
        if (verbose && (printFreq == 0 || currBatchIdx % printFreq == 0)) {
          Logger.logs("Computing gradient on " + currIdx + " (batch " + currBatchIdx + " / " + (trainExs.size / batchSize) + ")");
        }
        cumulativeObjective += takeAdagradStepL1RSparse(trainExs.slice(currIdx, Math.min(trainExs.size, currIdx + batchSize)), computer, weights);
        computer.weightsUpdateCallback(weights)
        currIdx += batchSize;
        currBatchIdx += 1;
      }
      cumulativeObjective += computeRegularizationTermL1R(weights.weights, lambda)
      Logger.logss("APPROXIMATE OBJECTIVE: " + cumulativeObjective + " (avg = " + cumulativeObjective/trainExs.size + ")")
      if (verbose) {
        Logger.endTrack();
        Logger.logss("Not displaying weights since they will be inaccurate")
        displayTime(i, startTime, inferenceNanos, adagradNanos)
      }
      computer.iterationEndCallback(weights)
    }
    if (verbose) Logger.logss("FINAL TRAIN OBJECTIVE: " + computeObjectiveL1RSparse(trainExs, computer, weights, lambda));
    val finalWeights = weights.finalizeWeights
    displayWeights(finalWeights)
    finalWeights
  }
  
  def takeAdagradStepL1RSparse(exs: Seq[T],
                               computer: LikelihoodAndGradientComputerSparse[T],
                               weights: AdagradWeightVector): Double = {
    val gradientCounter = new IntCounter
    val objective = getMinibatchObjectiveAndGradientSparse(exs, computer, weights, gradientCounter)
    val nanoTime = System.nanoTime();
    // Precompute this so dividing by batch size is a multiply and not a divide
    weights.applyGradientUpdate(gradientCounter, exs.size)
    adagradNanos += (System.nanoTime() - nanoTime);
    objective
  }
  
  //////////////
  // ADADELTA //
  //////////////
  
  def trainAdadelta(trainExs: Seq[T],
                    computer: LikelihoodAndGradientComputer[T],
                    rho: Double,
                    batchSize: Int,
                    numItrs: Int,
                    initialWeights: Array[Double],
                    verbose: Boolean = true): Array[Double] = {
//    val weights = Array.fill(pairwiseIndexingFeaturizer.featureIndexer.size)(0.0);
    val weights = initialWeights;
    val reusableGradientArray = Array.fill(initialWeights.size)(0.0);
    val gradsSquared = Array.fill(initialWeights.size)(0.0);
    val updatesSquared = Array.fill(initialWeights.size)(0.0);
    for (i <- 0 until numItrs) {
      Logger.logss("ITERATION " + i);
      val startTime = System.nanoTime();
      inferenceNanos = 0;
      adagradNanos = 0;
      if (verbose) Logger.startTrack("Computing gradient");
      var cumulativeObjective = 0.0
      var currIdx = 0;
      var currBatchIdx = 0;
      val printFreq = (trainExs.size / batchSize) / 10 // Print progress 10 times per pass through the data
      while (currIdx < trainExs.size) {
        if (verbose && (printFreq == 0 || currBatchIdx % printFreq == 0)) {
          Logger.logs("Computing gradient on " + currIdx + " (batch " + currBatchIdx + " / " + (trainExs.size / batchSize) + ")");
        }
        cumulativeObjective += takeAdadeltaStep(trainExs.slice(currIdx, Math.min(trainExs.size, currIdx + batchSize)),
                                                computer,
                                                weights,
                                                reusableGradientArray,
                                                gradsSquared,
                                                updatesSquared,
                                                rho);
        computer.weightsUpdateCallback(weights)
        currIdx += batchSize;
        currBatchIdx += 1;
      }
      Logger.logss("APPROXIMATE OBJECTIVE: " + cumulativeObjective + " (avg = " + cumulativeObjective/trainExs.size + ")")
      if (verbose) {
        Logger.endTrack();
        displayWeights(weights)
        displayTime(i, startTime, inferenceNanos, adagradNanos)
      }
      computer.iterationEndCallback(weights)
    }
    if (verbose) {
      Logger.logss("FINAL TRAIN OBJECTIVE: " + computeObjectiveL1R(trainExs, computer, weights, 0.0));
    }
    weights
  }
  
  
  def takeAdadeltaStep(exs: Seq[T],
                       computer: LikelihoodAndGradientComputer[T],
                       weights: Array[Double],
                       reusableGradientArray: Array[Double],
                       gradsSquared: Array[Double],
                       updatesSquared: Array[Double],
                       rho: Double): Double = {
    Arrays.fill(reusableGradientArray, 0.0);
    val objective = getMinibatchObjectiveAndGradient(exs, computer, weights, reusableGradientArray)
    val nanoTime = System.nanoTime();
    val epsilon = 1e-6
    val inverseBatchSize = 1.0/exs.size
    var i = 0
    while (i < reusableGradientArray.size) {
      // Rescale the gradient by the batch size
      reusableGradientArray(i) *= inverseBatchSize
      gradsSquared(i) = rho * gradsSquared(i) + (1 - rho) * reusableGradientArray(i) * reusableGradientArray(i)
      val step = Math.sqrt(updatesSquared(i) + epsilon)/Math.sqrt(gradsSquared(i) + epsilon) * reusableGradientArray(i)
      updatesSquared(i) = rho * updatesSquared(i) + (1 - rho) * step * step
      weights(i) += step
      i += 1
    }
    adagradNanos += (System.nanoTime() - nanoTime);
    objective
  }
  
  
  ///////////////////////
  // SGD WITH MOMENTUM //
  ///////////////////////
  
  def trainSGDMomentum(trainExs: Seq[T],
                       computer: LikelihoodAndGradientComputer[T],
                       stepSize: Double,
                       momentum: Double,
                       lambda: Double,
                       batchSize: Int,
                       numItrs: Int,
                       initialWeights: Array[Double],
                       verbose: Boolean = true): Array[Double] = {
//    val weights = Array.fill(pairwiseIndexingFeaturizer.featureIndexer.size)(0.0);
    val weights = initialWeights;
    val reusableGradientArray = Array.fill(initialWeights.size)(0.0);
    val pastStep = Array.fill(initialWeights.size)(0.0);
    for (i <- 0 until numItrs) {
      Logger.logss("ITERATION " + i);
      val startTime = System.nanoTime();
      inferenceNanos = 0;
      adagradNanos = 0;
      if (verbose) Logger.startTrack("Computing gradient");
      var cumulativeObjective = 0.0
      var currIdx = 0;
      var currBatchIdx = 0;
      val printFreq = (trainExs.size / batchSize) / 10 // Print progress 10 times per pass through the data
      while (currIdx < trainExs.size) {
        if (verbose && (printFreq == 0 || currBatchIdx % printFreq == 0)) {
          Logger.logs("Computing gradient on " + currIdx + " (batch " + currBatchIdx + " / " + (trainExs.size / batchSize) + ")");
        }
        cumulativeObjective += takeSGDMomentumStep(trainExs.slice(currIdx, Math.min(trainExs.size, currIdx + batchSize)),
                                                   computer,
                                                   weights,
                                                   reusableGradientArray,
                                                   pastStep,
                                                   stepSize,
                                                   momentum,
                                                   lambda);
        computer.weightsUpdateCallback(weights)
        currIdx += batchSize;
        currBatchIdx += 1;
      }
      Logger.logss("APPROXIMATE OBJECTIVE: " + cumulativeObjective + " (avg = " + cumulativeObjective/trainExs.size + ")")
      if (verbose) {
        Logger.endTrack();
        displayWeights(weights)
        displayTime(i, startTime, inferenceNanos, adagradNanos)
      }
      computer.iterationEndCallback(weights)
    }
    if (verbose) {
      Logger.logss("FINAL TRAIN OBJECTIVE: " + computeObjectiveL1R(trainExs, computer, weights, 0.0));
    }
    weights
  }
  
  def takeSGDMomentumStep(exs: Seq[T],
                          computer: LikelihoodAndGradientComputer[T],
                          weights: Array[Double],
                          reusableGradientArray: Array[Double],
                          pastStep: Array[Double],
                          stepSize: Double,
                          momentum: Double,
                          lambda: Double): Double = {
    Arrays.fill(reusableGradientArray, 0.0);
    var nanoTime = System.nanoTime();
    var objective = 0.0
    for (ex <- exs) {
      objective += computer.accumulateGradientAndComputeObjective(ex, weights, reusableGradientArray);
    }
    inferenceNanos += (System.nanoTime() - nanoTime);
    nanoTime = System.nanoTime();
    // Precompute this so dividing by batch size is a multiply and not a divide
    var i = 0;
    while (i < reusableGradientArray.size) {
      pastStep(i) = pastStep(i) * momentum + stepSize * (reusableGradientArray(i) - 2 * weights(i) * lambda)
      weights(i) += pastStep(i)
      i += 1
    }
    adagradNanos += (System.nanoTime() - nanoTime);
    objective
  }
  
  ///////////
  // LBFGS //
  ///////////
  
  def trainLBFGS(trainExs: Seq[T],
                 computer: LikelihoodAndGradientComputer[T],
                 lambda: Double,
                 epsilon: Double,
                 numItrs: Int,
                 initialWeights: Array[Double],
                 verbose: Boolean = true): Array[Double] = {
    val diffFunc = new CachingDifferentiableFunction() {
      val reusableGradientArr = Array.tabulate(initialWeights.size)(i => 0.0)
      
      def dimension(): Int = initialWeights.size
      
//      protected[edu.berkeley.nlp.futile.math] calculate(currWeights: Array[Double]): edu.berkeley.nlp.futile.fig.basic.Pair[Double,Array[Double]] = {
      import java.lang.{Double => JDouble}
      def calculate(currWeights: Array[Double]): edu.berkeley.nlp.futile.fig.basic.Pair[JDouble,Array[Double]] = {
//        val sweights = Array.tabulate(currWeights.size)(i => currWeights(i).asInstanceOf[Double])
        val nanoTime = System.nanoTime();
        var objective = 0.0;
        Arrays.fill(reusableGradientArr, 0.0);
        for (ex <- trainExs) {
          objective += computer.accumulateGradientAndComputeObjective(ex, currWeights, reusableGradientArr)
//          objective += computer.accumulateGradientAndComputeObjective(ex, sweights, reusableGradientArr)
        }
        for (i <- 0 until reusableGradientArr.size) {
          objective -= lambda * currWeights(i) * currWeights(i);
          reusableGradientArr(i) -= 2 * lambda * currWeights(i)
          reusableGradientArr(i) = -reusableGradientArr(i)
        }
        val negObjective = -objective;
        var norm = 0.0
        for (j <- 0 until currWeights.size) {
          norm += currWeights(j) * currWeights(j);
        }
        Logger.logss("NORM OF WEIGHTS: " + norm);
        Logger.logss("OBJECTIVE: " + objective + " (avg = " + objective/trainExs.size + ")")
        Logger.logss("TRAIN MILLIS: "+  (System.nanoTime() - nanoTime)/1000000);
        new edu.berkeley.nlp.futile.fig.basic.Pair[JDouble,Array[Double]](new JDouble(negObjective), reusableGradientArr);
      }
    }
    new LBFGSMinimizer(numItrs).minimize(diffFunc, initialWeights, epsilon, true);
  }
}

object GeneralTrainer {
  
  def addToGradient(arr: Array[Int], scale: Double, gradient: IntCounter) {
    var i = 0
    while (i < arr.size) {
      gradient.incrementCount(arr(i), scale)
      i += 1
    }
  }
  
  def checkGradient[T](trainExs: Seq[T],
                       computer: LikelihoodAndGradientComputer[T],
                       numFeats: Int,
                       indexSet: Set[Int] = Set(),
                       verbose: Boolean = false) {
    val rng = new Random(0);
    val randWeights = Array.tabulate(numFeats)(i => (rng.nextDouble * 0.1).toDouble)
    if (numFeats < 100) {
      Logger.logss("Weights: " + randWeights.toSeq);
    }
    checkGradientFromPoint(trainExs, computer, randWeights, Array.tabulate(numFeats)(i => 0.0), indexSet, verbose);
  }
  
  def checkGradientFromPoint[T](trainExs: Seq[T],
                                computer: LikelihoodAndGradientComputer[T],
                                weights: Array[Double],
                                gradient: Array[Double],
                                indexSet: Set[Int] = Set(),
                                verbose: Boolean = false) {
    var currLL = trainExs.map(ex => computer.accumulateGradientAndComputeObjective(ex, weights, gradient)).reduce(_ + _);
    Logger.logss("Base LL: " + currLL)
    val stepSize = 1e-3F;
    // If we've restricted to an index set, only check those, otherwise check all indices
    val indicesToCheck = if (indexSet.isEmpty) 0 until gradient.size else indexSet.toSeq.sorted;
    for (i <- indicesToCheck) {
      if (i % 1000 == 0) {
        Logger.logss("Checking empirical gradient on weight " + i);
      }
      weights(i) += stepSize;
      var newLL: Double = trainExs.map(computer.computeObjective(_, weights)).reduce(_ + _)
      val empGradient = (newLL - currLL)/stepSize;
      if (Math.abs(empGradient - gradient(i)) > 1e-3) {
        Logger.logss("Difference on feature " + i + ": gradient: " + gradient(i) + ", emp gradient: " + empGradient);
      } else if (verbose) {
        Logger.logss("On feature " + i + ": gradient: " + gradient(i) + ", emp gradient: " + empGradient);
      }
      weights(i) -= stepSize;
    }
  }
}

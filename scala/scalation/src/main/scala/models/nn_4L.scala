

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Neural Network with 3 Layers (input, hidden and output layers)
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation
package modeling
package neuralnet

import scala.math.max
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import ActivationFun._
import Initializer._
import Optimizer._

import java.io.PrintStream
import java.io.FileOutputStream
import java.io.FileDescriptor

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XL` class supports multi-output, X-layer (input, hidden* and output)
 *  Neural-Networks.  It can be used for both classification and prediction,
 *  depending on the activation functions used.  Given several input vectors and output
 *  vectors (training data), fit the parameters [b] connecting the layers,
 *  so that for a new input vector v, the net can predict the output value, e.g.,
 *      yp = f3 (c * f2 (b * f (a * v)))
 *  where f, f2 and f3 are the activation functions and the parameter a, b and b
 *  are the parameters between input-hidden1, hidden1-hidden2 and hidden2-output layers.
 *  Unlike `NeuralNet_XL` which adds input x0 = 1 to account for the intercept/bias,
 *  `NeuralNet_XL` explicitly adds bias.
 *  Defaults to two hidden layers.
 *  This implementation is partially adapted from Michael Nielsen's Python implementation found in
 *  @see  github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
 *  @see  github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/network2.py
 *------------------------------------------------------------------------------
 *  @param x       the m-by-n input/data matrix (training data consisting of m input vectors)
 *  @param y       the m-by-ny output/response matrix (training data consisting of m output vectors)
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param nz      the number of nodes in each hidden layer, e.g., Array (9, 8) => 2 hidden of sizes 9 and 8
 *                 (null => use default formula)
 *  @param hparam  the hyper-parameters for the model/network
 *  @param f       the array of activation function families between every pair of layers
 *  @param itran   the inverse transformation function returns response matrix to original scale
 */
class NeuralNet_XL (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                    private var nz: Array [Int] = null, hparam: HyperParameter = Optimizer.hp,
                    f: Array [AFF] = Array (f_sigmoid, f_sigmoid, f_id),
                    val itran: FunctionM2M = null)
      extends PredictorMV (x, y, fname_, hparam)
         with Fit (dfm = x.dim2, df = x.dim - x.dim2):                    // under-estimate of degrees of freedom

    private val debug     = debugf ("NeuralNet_XL", false)                // debug function
    private val flaw      = flawf ("NeuralNet_XL")                        // flaw function
//  private val eta       = hp("eta").toDouble                            // learning rate
    private val eta       = 0.001                                         // learning rate
    private val bSize     = hp("bSize").toInt                             // batch size
    private val maxEpochs = hp("maxEpochs").toInt                         // maximum number of training epochs/iterations
    private val lambda    = hp ("lambda").toDouble                        // regularization hyper-parameter
    private val nl        = f.length                                      // number of active layers (i.e., with activation function)
    private val layers    = 0 until nl                                    // range for active layers
//          val opti      = new Optimizer_SGD ()                          // parameter optimizer SGD
            val opti      = new Optimizer_SGDM ()                         // parameter optimizer SGDM

    // Guidelines for setting the number of nodes in hidden layer:
    if nz == null then nz = compute_nz                                    // default number of nodes for each hidden layers

    if nz.length + 1 != nl then
        flaw ("init", "count mismatch among number of layers and activation functions")
    end if

    private val sizes = x.dim2 +: nz :+ y.dim2                            // sizes (# nodes) of all layers
                bb = Array.ofDim [NetParam] (nl)                          // parameters for each active layer

    for l <- layers do
        bb(l) = new NetParam (weightMat (sizes(l), sizes(l+1)),           // parameters weights &
                              weightVec (sizes(l+1)))                     // biases per active layer
    end for

    modelName = s"NeuralNet_XL_${f(0).name}_${f(1).name}"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute default values for the number nodes in each hidden layer, based on
     *  the number of nodes in the input layer.
     *  Rule: e.g., n = 15 => [ 31, 15, 10, 7 ] 
     */
    def compute_nz: Array [Int] =
        val nz1 = 2 * x.dim2 + 1
        (for l <- 1 until f.length yield max (1, nz1 / l)).toArray
    end compute_nz

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameters bb.
     *  Minimize the error in the prediction by adjusting the parameters bb.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val epochs = opti.optimize (x_, y_, bb, eta, f)                   // optimize parameters bb
        println (s"ending epoch = $epochs")
        estat.tally (epochs._2)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameters bb.
     *  Minimize the error in the prediction by adjusting the parameters bb.
     *  This version preforms an interval search for the best eta value.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    override def train2 (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val etaI = (0.25 * eta, 4.0 * eta)                                     // quarter to four times eta
        val epochs = opti.auto_optimize (x_, y_, bb, etaI, f, opti.optimize)   // optimize parameters bb
        println (s"ending epoch = $epochs")
        estat.tally (epochs._2)
    end train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output matrix (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : MatrixD = y): (MatrixD, MatrixD) =
        val yp = predict (x_)                                            // make predictions
        e = y_ - yp                                                      // RECORD the residuals/errors (@see `Predictor`)
        val qof = MatrixD (for k <- y_.indices2 yield diagnose (y_(?, k), yp(?, k))).transpose
        (yp, qof)                                                        // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector v, predict the output/response vector f(v).
     *  @param v  the new input vector
     */
    def predict (v: VectorD): VectorD =
        var u = v
        for l <- layers do u = f(l).f_ (bb(l) dot u)
        u
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an input matrix v, predict the output/response matrix f(v).
     *  @param v  the input matrix
     */
    override def predict (v: MatrixD = x): MatrixD =
        var u = v
        for l <- layers do u = f(l).fM (bb(l) * u)
        u
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatrixD): NeuralNet_XL =
        new NeuralNet_XL (x_cols, y, null, null, hparam, f, itran)
    end buildModel

end NeuralNet_XL


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XL` companion object provides factory functions for buidling multi-layer
 *  neural nets.
 */
object NeuralNet_XL extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_XL` with automatic resclaing from a combined data matrix.
     *  @param xy      the combined input and output matrix
     *  @param fname   the feature/variable names
     *  @param nz      the number of nodes in each hidden layer, e.g., Array (5, 10) means 2 hidden with sizes 5 and 10
     *  @param hparam  the hyper-parameters
     *  @param f       the array of activation function families between every pair of layers
     *  @param col     the first designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               nz: Array [Int] = null, hparam: HyperParameter = Optimizer.hp,
               f: Array [AFF] = Array (f_sigmoid, f_sigmoid, f_id))
               (col: Int = xy.dim2 - 1): NeuralNet_XL =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale
        val (x, y) = (xy(?, 0 until col), xy(?, col until xy.dim2)) 

        val x_s = if scale then rescaleX (x, f(0))
                  else x
        val y_s = if f.last.bounds != null then { val y_i = rescaleY (y, f.last); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_XL (x_s, y_s, fname, nz, hparam, f, itran)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_XL` with automatic rescaling from a data matrix and response vector.
     *  @param x       the input/data matrix
     *  @param y       the output/response matrix
     *  @param fname   the feature/variable names
     *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam  the hyper-parameters
     *  @param f       the array of activation function families between every pair of layers
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                 nz: Array [Int] = null, hparam: HyperParameter = Optimizer.hp,
                 f: Array [AFF] = Array (f_sigmoid, f_sigmoid, f_id)): NeuralNet_XL =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale

        val x_s = if scale then rescaleX (x, f(0))
                  else x
        val y_s = if f.last.bounds != null then { val y_i = rescaleY (y, f.last); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_XL (x_s, y_s, fname, nz, hparam, f, itran)
    end rescale

end NeuralNet_XL



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AutoMPG TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_sigmoid_AutoMPG` main function tests the `NeuralNet_XL` class using the
 *  AutoMPG dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_sigmoid_AutoMPG
 */
@main def nn_4L_sigmoid_AutoMPG (args: String*): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_sigmoid_AutoMPG.txt")))
 
    import AutoMPG_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println(s"ox_fname = ${stringOf (ox_fname)}")
   
    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_XL for AutoMPG with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_sigmoid_AutoMPG


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_tanh_AutoMPG` main function tests the `NeuralNet_XL` class using the
 *  AutoMPG dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_tanh_AutoMPG
 */
@main def nn_4L_tanh_AutoMPG (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_tanh_AutoMPG.txt")))

    import AutoMPG_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    s"ox_fname = ${stringOf (ox_fname)}"

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_XL for AutoMPG with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_tanh_AutoMPG


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_reLU_AutoMPG` main function tests the `NeuralNet_XL` class using the
 *  AutoMPG dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_reLU_AutoMPG
 */
@main def nn_4L_reLU_AutoMPG (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_reLU_AutoMPG.txt")))

    import AutoMPG_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_XL for AutoMPG with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_reLU_AutoMPG


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ForestFires TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_sigmoid_ForestFires` main function tests the `NeuralNet_XL` class using the
 *  ForestFires dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_sigmoid_ForestFires
 */
@main def nn_4L_sigmoid_ForestFires (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_sigmoid_ForestFires.txt")))

    import ForestFires_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_XL for ForestFires with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_sigmoid_ForestFires


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_tanh_ForestFires` main function tests the `NeuralNet_XL` class using the
 *  ForestFires dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_tanh_ForestFires
 */
@main def nn_4L_tanh_ForestFires (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_tanh_ForestFires.txt")))

    import ForestFires_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_XL for ForestFires with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_tanh_ForestFires


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_reLU_ForestFires` main function tests the `NeuralNet_XL` class using the
 *  ForestFires dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_reLU_ForestFires
 */
@main def nn_4L_reLU_ForestFires (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_reLU_ForestFires.txt")))

    import ForestFires_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_XL for ForestFires with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_reLU_ForestFires


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CCPP TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_sigmoid_CCPP` main function tests the `NeuralNet_XL` class using the
 *  CCPP dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_sigmoid_CCPP
 */
@main def nn_4L_sigmoid_CCPP (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_sigmoid_CCPP.txt")))

    import CCPP_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_XL for CCPP with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_sigmoid_CCPP


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_tanh_CCPP` main function tests the `NeuralNet_XL` class using the
 *  CCPP dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_tanh_CCPP
 */
@main def nn_4L_tanh_CCPP (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_tanh_CCPP.txt")))

    import CCPP_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_XL for CCPP with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_tanh_CCPP


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_reLU_CCPP` main function tests the `NeuralNet_XL` class using the
 *  CCPP dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_reLU_CCPP
 */
@main def nn_4L_reLU_CCPP (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_reLU_CCPP.txt")))

    import CCPP_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_XL for CCPP with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_reLU_CCPP


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WineQuality TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_sigmoid_WineQuality` main function tests the `NeuralNet_XL` class using the
 *  WineQuality dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_sigmoid_WineQuality
 */
@main def nn_4L_sigmoid_WineQuality (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_sigmoid_WineQuality.txt")))

    import WineQuality_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_XL for WineQuality with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_sigmoid_WineQuality


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_tanh_WineQuality` main function tests the `NeuralNet_XL` class using the
 *  WineQuality dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_tanh_WineQuality
 */
@main def nn_4L_tanh_WineQuality (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_tanh_WineQuality.txt")))

    import WineQuality_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_XL for WineQuality with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_tanh_WineQuality


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_reLU_WineQuality` main function tests the `NeuralNet_XL` class using the
 *  WineQuality dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_reLU_WineQuality
 */
@main def nn_4L_reLU_WineQuality (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_reLU_WineQuality.txt")))

    import WineQuality_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_XL for WineQuality with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_reLU_WineQuality


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BikeSharing TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_sigmoid_BikeSharing` main function tests the `NeuralNet_XL` class using the
 *  BikeSharing dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_sigmoid_BikeSharing
 */
@main def nn_4L_sigmoid_BikeSharing (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_sigmoid_BikeSharing.txt")))

    import BikeSharing_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_XL for BikeSharing with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_sigmoid_BikeSharing


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_tanh_BikeSharing` main function tests the `NeuralNet_XL` class using the
 *  BikeSharing dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_tanh_BikeSharing
 */
@main def nn_4L_tanh_BikeSharing (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_tanh_BikeSharing.txt")))

    import BikeSharing_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_XL for BikeSharing with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_tanh_BikeSharing


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_XL_reLU_BikeSharing` main function tests the `NeuralNet_XL` class using the
 *  BikeSharing dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_XL_reLU_BikeSharing
 */
@main def nn_4L_reLU_BikeSharing (): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("log/output/nn_4L_reLU_BikeSharing.txt")))

    import BikeSharing_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_XL for BikeSharing with ${af.name}")
//  val mod = new NeuralNet_XL (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (ox, yy, ox_fname,            // create model with intercept (else pass x) - rescales
                                    f = Array(af, af, f_id)) 
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
// println (mod.summary ())                                      // parameter/coefficient statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end nn_4L_reLU_BikeSharing
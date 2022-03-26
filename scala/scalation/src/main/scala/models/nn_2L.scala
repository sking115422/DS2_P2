

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Neural Network with 2 Layers (input and output layers)
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation
package modeling
package neuralnet

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import ActivationFun._
import Initializer._
import Optimizer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_2L` class supports multi-output, 2-layer (input and output)
 *  Neural-Networks.  It can be used for both classification and prediction,
 *  depending on the activation functions used.  Given several input vectors and output
 *  vectors (training data), fit the weights/parameters b connecting the layers,
 *  so that for a new input vector z, the net can predict the output value, i.e.,
 *      yp_j = f (b dot z)
 *  where f is the activation function and the parameters b gives the weights
 *  between input and output layers.
 *  NOTE, b0 is treated as the bias, so x0 must be 1.0.
 *  @param x       the m-by-n input/data matrix (training data consisting of m input vectors)
 *  @param y       the m-by-ny output/response matrix (training data consisting of m output vectors)
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model/network
 *  @param f       the activation function family for layers 1->2 (input to output)
 *  @param itran   the inverse transformation function returns response matrix to original scale
 */
class NeuralNet_2L (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                    hparam: HyperParameter = Optimizer.hp,
                    f: AFF = f_sigmoid, val itran: FunctionM2M = null)
      extends PredictorMV (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):

    private val debug     = debugf ("NeuralNet_2L", false)                // debug function
    private val eta       = hp("eta").toDouble                            // learning rate
    private val bSize     = hp("bSize").toInt                             // batch size
    private val maxEpochs = hp("maxEpochs").toInt                         // maximum number of training epochs/iterations
//          val opti      = new Optimizer_SGD ()                          // parameter optimizer SGD
            val opti      = new Optimizer_SGDM ()                         // parameter optimizer SGDM

    b  = new NetParam (weightMat (x.dim2, y.dim2))                        // initialize parameters b
    bb = Array (b.asInstanceOf [NetParam])                                // inside array 

    modelName = "NeuralNet_2L_" + f.name

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameters bb.
     *  Minimize the error in the prediction by adjusting the parameters bb.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val epochs = opti.optimize2 (x_, y_, bb, eta, Array (f))          // optimize parameters bb
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
        val etaI = (0.25 * eta, 4.0 * eta)                                              // quarter to four times eta
        val epochs = opti.auto_optimize (x_, y_, bb, etaI, Array (f), opti.optimize2)   // optimize parameters bb
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
    def predict (v: VectorD): VectorD = f.f_ (bb(0) dot v)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an input matrix v, predict the output/response matrix f(v).
     *  @param v  the input matrix
     */
    override def predict (v: MatrixD = x): MatrixD = f.fM (bb(0) * v)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatrixD): NeuralNet_2L =
        new NeuralNet_2L (x_cols, y, null, hparam, f, itran)
    end buildModel

end NeuralNet_2L


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_2L` companion object provides factory functions for buidling two-layer
 *  neural nets.
 */
object NeuralNet_2L extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_2L` with automatic resclaing from a combined data matrix.
     *  @param xy      the combined input and output matrix
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to output)
     *  @param col     the first designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = Optimizer.hp, f: AFF = f_sigmoid)
               (col: Int = xy.dim2 - 1): NeuralNet_2L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale
        val (x, y) = (xy(?, 0 until col), xy(?, col until xy.dim2)) 

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_2L (x_s, y_s, fname, hparam, f, itran)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_2L` with automatic rescaling from a data matrix and response vector.
     *  @param x       the input/data matrix
     *  @param y       the output/response matrix
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to output)
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                 hparam: HyperParameter = Optimizer.hp, f: AFF = f_sigmoid): NeuralNet_2L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_2L (x_s, y_s, fname, hparam, f, itran)
    end rescale

end NeuralNet_2L


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AutoMPG TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_sigmoid_AutoMPG` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_sigmoid_AutoMPG
 */
@main def nn_2L_sigmoid_AutoMPG (): Unit = 

    import AutoMPG_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_2L for AutoMPG with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_sigmoid_AutoMPG


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_tanh_AutoMPG` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_tanh_AutoMPG
 */
@main def nn_2L_tanh_AutoMPG (): Unit = 

    import AutoMPG_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_2L for AutoMPG with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_tanh_AutoMPG


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_reLU_AutoMPG` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_reLU_AutoMPG
 */
@main def nn_2L_reLU_AutoMPG (): Unit = 

    import AutoMPG_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_2L for AutoMPG with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_reLU_AutoMPG


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ForestFires TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_sigmoid_ForestFires` main function tests the `NeuralNet_2L` class using the
 *  ForestFires dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_sigmoid_ForestFires
 */
@main def nn_2L_sigmoid_ForestFires (): Unit = 

    import ForestFires_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_2L for ForestFires with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_sigmoid_ForestFires


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_tanh_ForestFires` main function tests the `NeuralNet_2L` class using the
 *  ForestFires dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_tanh_ForestFires
 */
@main def nn_2L_tanh_ForestFires (): Unit = 

    import ForestFires_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_2L for ForestFires with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_tanh_ForestFires


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_reLU_ForestFires` main function tests the `NeuralNet_2L` class using the
 *  ForestFires dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_reLU_ForestFires
 */
@main def nn_2L_reLU_ForestFires (): Unit = 

    import ForestFires_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_2L for ForestFires with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_reLU_ForestFires


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CCPP TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_sigmoid_CCPP` main function tests the `NeuralNet_2L` class using the
 *  CCPP dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_sigmoid_CCPP
 */
@main def nn_2L_sigmoid_CCPP (): Unit = 

    import CCPP_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_2L for CCPP with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_sigmoid_CCPP


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_tanh_CCPP` main function tests the `NeuralNet_2L` class using the
 *  CCPP dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_tanh_CCPP
 */
@main def nn_2L_tanh_CCPP (): Unit = 

    import CCPP_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_2L for CCPP with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_tanh_CCPP


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_reLU_CCPP` main function tests the `NeuralNet_2L` class using the
 *  CCPP dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_reLU_CCPP
 */
@main def nn_2L_reLU_CCPP (): Unit = 

    import CCPP_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_2L for CCPP with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_reLU_CCPP


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WineQuality TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_sigmoid_WineQuality` main function tests the `NeuralNet_2L` class using the
 *  WineQuality dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_sigmoid_WineQuality
 */
@main def nn_2L_sigmoid_WineQuality (): Unit = 

    import WineQuality_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_2L for WineQuality with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_sigmoid_WineQuality


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_tanh_WineQuality` main function tests the `NeuralNet_2L` class using the
 *  WineQuality dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_tanh_WineQuality
 */
@main def nn_2L_tanh_WineQuality (): Unit = 

    import WineQuality_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_2L for WineQuality with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_tanh_WineQuality


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_reLU_WineQuality` main function tests the `NeuralNet_2L` class using the
 *  WineQuality dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_reLU_WineQuality
 */
@main def nn_2L_reLU_WineQuality (): Unit = 

    import WineQuality_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_2L for WineQuality with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_reLU_WineQuality


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BikeSharing TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_sigmoid_BikeSharing` main function tests the `NeuralNet_2L` class using the
 *  BikeSharing dataset.  It tries the sigmoid acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_sigmoid_BikeSharing
 */
@main def nn_2L_sigmoid_BikeSharing (): Unit = 

    import BikeSharing_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_sigmoid

    banner (s"NeuralNet_2L for BikeSharing with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_sigmoid_BikeSharing


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_tanh_BikeSharing` main function tests the `NeuralNet_2L` class using the
 *  BikeSharing dataset.  It tries the tanh acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_tanh_BikeSharing
 */
@main def nn_2L_tanh_BikeSharing (): Unit = 

    import BikeSharing_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_tanh

    banner (s"NeuralNet_2L for BikeSharing with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_tanh_BikeSharing


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nn_2L_reLU_BikeSharing` main function tests the `NeuralNet_2L` class using the
 *  BikeSharing dataset.  It tries the reLU acitvation function. It then cross-validates
 *  and tests forward selection. 
 *  > runMain scalation.modeling.neuralnet.nn_2L_reLU_BikeSharing
 */
@main def nn_2L_reLU_BikeSharing (): Unit = 

    import BikeSharing_Data._
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    val af = f_reLU

    banner (s"NeuralNet_2L for BikeSharing with ${af.name}")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
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

end nn_2L_reLU_BikeSharing
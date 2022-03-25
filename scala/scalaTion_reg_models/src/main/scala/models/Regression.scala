

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Multiple Linear Regression (linear terms, no cross-terms)
 */

package scalation
package modeling

import scalation.mathstat._
import scalation.mathstat.MatrixD

import scala.runtime.ScalaRunTime.stringOf

import scala.language.postfixOps
import java.io._



//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Regression` class supports multiple linear regression.  In this case,
 *  x is multi-dimensional [1, x_1, ... x_k].  Fit the parameter vector b in
 *  the regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve the parameter vector b
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y 
 *      b  =  fac.solve (.)
 *  Five factorization algorithms are provided:
 *      `Fac_QR`         QR Factorization: slower, more stable (default)
 *      `Fac_SVD`        Singular Value Decomposition: slowest, most robust
 *      `Fac_Cholesky`   Cholesky Factorization: faster, less stable (reasonable choice)
 *      `Fac_LU'         LU Factorization: better than Inverse
 *      `Fac_Inverse`    Inverse Factorization: textbook approach
$\RN{y}$ *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  Note, not intended for use when the number of degrees of freedom 'df' is negative.
 *  @see en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)
 *------------------------------------------------------------------------------
 *  @param x       the data/input m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-vector
 *  @param fname_  the feature/variable names
 *  @param hparam  the hyper-parameters
 */
class Regression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                  hparam: HyperParameter = Regression.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug     = debugf ("Regression", false)                 // debug function
    private val flaw      = flawf ("Regression")                         // flaw function
    private val algorithm = hparam("factorization")                      // factorization algorithm
    private val n         = x.dim2                                       // number of columns

    modelName = "Regression"

    if n < 1 then flaw ("init", s"dim2 = $n of the 'x' matrix must be at least 1")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a solver for the Normal Equations using the selected factorization algorithm.
     *  @param x_  the matrix to be used by the solver
     */
    private def solver (x_ : MatrixD): Factorization =
        algorithm match                                                  // select factorization algorithm
        case "Fac_Cholesky" => new Fac_Cholesky (x_.transpose * x_)      // Cholesky Factorization
        case "Fac_LU"       => new Fac_LU (x_.transpose * x_)            // LU Factorization
        case "Fac_Inverse"  => new Fac_Inverse (x_.transpose * x_)       // Inverse Factorization
        case "Fac_SVD"      => new Fac_SVD (x_)                          // Singular Value Decomposition
        case _              => new Fac_QR (x_)                           // QR Factorization (default)
        end match
    end solver

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  using the ordinary least squares 'OLS' method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val fac = solver (x_)
        fac.factor ()                                                    // factor the matrix, either X or X.t * X

        b = fac match                                                    // RECORD the parameters/coefficients (@see `Predictor`)
            case fac: Fac_QR  => fac.solve (y_)
            case fac: Fac_SVD => fac.solve (y_)
            case _            => fac.solve (x_.transpose * y_)

        if b(0).isNaN then flaw ("train", s"parameter b = $b")
        debug ("train", s"$fac estimates parameter b = $b")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                            // make predictions
        e = y_ - yp                                                      // RECORD the residuals/errors (@see `Predictor`)
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_     the testing/full data/input matrix
     *  @param fname  the array of feature/variable names
     *  @param b      the parameters/coefficients for the model
     *  @param vifs   the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = b,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b).  It is overridden for speed.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): VectorD = x_ * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): Regression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new Regression (x_cols, y, null, hparam)
    end buildModel

end Regression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Regression` companion object provides factory apply functions and a testing method.
 */
object Regression:

    /** Base hyper-parameter specification for `Regression`
     */
    val hp = new HyperParameter; hp += ("factorization", "Fac_QR", "Fac_QR")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): Regression = 
        new Regression (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (use Regression.hp for default)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): Regression = 
        val xn = normalize (x, (x.mean, x.stdev))
        new Regression (xn, y, fname, hparam)
    end rescale

end Regression


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest` main function tests `Regression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.regressionTest
 */
@main def RegressionAutoMPG (): Unit =

    import AutoMPG_Data._

    banner ("Regression for Auto MPG")
    val mod = new Regression (x, y)                           // create a  regression model
    mod.trainNtest ()()                                            // train and test the model
    mod.summary ()                                                 // parameter/coefficient statistics

    // val writer = new FileWriter(new File("/Users/sdk/Desktop/UGA_proj/DS2_P1/scala/scalaTion_reg_models/src/main/scala/models/out/rep_sum.txt"), true)
    // writer.write(s"$summ")
    // writer.close()

    // banner ("Cross-Validation")
    // Fit.showQofStatTable (mod.crossValidate ())

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for  Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for


end RegressionAutoMPG



@main def Regression_ForestFires (): Unit =

    import ForestFires_Data._

    banner ("Regression for Forest Fires")
    val mod = new Regression (x, y)                           // create a  regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    // println (s"best (lambda, sse) = ${mod.findLambda}")

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for  Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
    
end Regression_ForestFires



@main def Regression_BikeSharing (): Unit =

    import BikeSharing_Data._

    banner ("Regression for Bike Sharing")
    val mod = new Regression (x, y)                           // create a  regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    // println (s"best (lambda, sse) = ${mod.findLambda}")

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for  Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end Regression_BikeSharing



@main def Regression_CCPP (): Unit =

    import CCPP_Data._

    banner ("Regression for CCPP")
    val mod = new Regression (x, y)                           // create a  regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    // println (s"best (lambda, sse) = ${mod.findLambda}")

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for  Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end Regression_CCPP



@main def Regression_WineQuality (): Unit =

    import WineQuality_Data._

    banner ("Regression for Wine Quality")
    val mod = new Regression (x, y)                           // create a  regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    // println (s"best (lambda, sse) = ${mod.findLambda}")

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for  Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end Regression_WineQuality

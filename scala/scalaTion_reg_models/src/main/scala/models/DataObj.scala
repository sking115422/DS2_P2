

package scalation

import scalation.mathstat._



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    DATA OBJECTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


object AutoMPG_Data:

    /** the names of the predictor variables; the name of response variable is mpg
     */
    val xr_fname = Array ("cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "modelyear", "origin")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("auto_mpg_fixed_cleaned.csv")

    /** the origin column (6) is categorical
     */
    val oxr = xyr.not(?, 7)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    /** the combined data matrix xy with the origin column (6) removed
     */
    val xy = xyr.not(?, 6)                                             // remove the origin column
//  val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (6)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end AutoMPG_Data



object ForestFires_Data:

    /** the names of the predictor variables; the name of response variable is area
     */
    val xr_fname = Array ("X", "Y", "month", "day",
                          "FFMC", "DMC", "DC", "ISI",
                          "temp", "RH", "wind", "rain")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("forestfires_cleaned.csv")

    val oxr = xyr.not(?, 12)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (12)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end ForestFires_Data



object CCPP_Data:

    /** the names of the predictor variables; the name of response variable is PE
     */
    val xr_fname = Array ("AT", "V", "AP", "RH")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("CCPP.csv")

    val oxr = xyr.not(?, 4)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (4)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end CCPP_Data



object WineQuality_Data:

    /** the names of the predictor variables; the name of response variable is quality
     */
    val xr_fname = Array ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
                          "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("winequality-white_fixed.csv")

    val oxr = xyr.not(?, 11)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (11)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end WineQuality_Data



object BikeSharing_Data:

    /** the names of the predictor variables; the name of response variable is cnt
     */
    val xr_fname = Array ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("winequality-white_fixed.csv")

    val oxr = xyr.not(?, 12)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (12)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end BikeSharing_Data

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Spencer King, John Miller
 *  @version 2.0
 *  @date    Wed Jun  8 13:16:15 EDT 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   RedirIO Allows Switching between Standard Output and Printing to Files
 */

package scalation

import java.io.PrintStream
import java.io.FileOutputStream
import java.io.FileDescriptor

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RedirIO` class makes it easy to switch between writing to standard output
 *  and a (log) file.
 *  @param project   the project or directory involved
 *  @param filename  the name of the file to be written
 *  @param toFile    flag indicating whether to write to a file
 */
class RedirIO (project: String):

    private val debug = debugf ("RedirIO", true)                    // debug function

    /** The file path for the (log) file
     */
    private val LOG_PATH = LOG_DIR + project + ⁄
    debug ("contructor", s"log file path = $LOG_PATH")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Redirect all IO output to logfile
     */
    def redirIOtoFile (name: String = "out.txt"): Unit =
        println("Output will now be logged to "+ LOG_PATH + name)
        return System.setOut(new PrintStream(new FileOutputStream(LOG_PATH + name)))
        
    end redirIOtoFile

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Redirect all IO output to console
     */
    def redirIOtoConsole (): Unit =
        println("Output will now be logged to console")
        return System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)))
    end redirIOtoConsole

end RedirIO


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RedirIOTest1` main function is used to test the `RedirIO` class.
 *  It will redirect all out put to a file then back to the console
 *  > runMain scalation.RedirIOTest1
 */
@main def RedirIOTest1 (args: String*): Unit =   

    // This below code statement works if it is not called from the class above...
    // I am not sure why it matters where it is called from, but apparently it does.
     
    // System.setOut(new PrintStream(new FileOutputStream("log/output/out.txt"))) 

    val redir = new RedirIO ("output")
    
    val s = "Hello World!"

    println(s + "   1")    

    redir.redirIOtoFile ("RedirIOTest1.txt")         // switch to output to logfile

    println(s + "   2")

    redir.redirIOtoConsole ()         // switch to output to console

    println(s + "   3")

end RedirIOTest1





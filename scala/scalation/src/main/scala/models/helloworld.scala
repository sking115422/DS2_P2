import java.io.{PrintStream, FileOutputStream}


@main def helloworld (args: String*): Unit = 

    System.setOut(new PrintStream(new FileOutputStream("/Users/sdk/Desktop/DS2_P2/scala/scalation/log/output/out.txt")))

    println("Hello World")

end helloworld
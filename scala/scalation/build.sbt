
lazy val models = project.in(file("."))
  .settings(
    scalaVersion  := "3.1.1",
//  scalacOptions += "-deprecation",
//  javacOptions  += "--add-modules jdk.incubator.vector"
  )

fork := true


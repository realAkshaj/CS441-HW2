ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "CS441-HW2"
  )



libraryDependencies ++= Seq(
  "com.typesafe" % "config" % "1.4.3",                 // For configuration management
  "com.knuddels" % "jtokkit" % "1.1.0",
  "org.slf4j" % "slf4j-log4j12" % "2.0.13",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-datasets" % "1.0.0-M2.1",
  "org.datavec" % "datavec-api" % "1.0.0-M2.1",
  "org.datavec" % "datavec-data-codec" % "1.0.0-M1.1",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
  "org.yaml" % "snakeyaml" % "2.0",
  "com.opencsv" % "opencsv" % "5.9",
  "org.apache.spark" %% "spark-core" % "3.5.0",
  "org.apache.spark" %% "spark-sql" % "3.5.0",
  "org.apache.spark" %% "spark-mllib" % "3.5.0",
  "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-M2.1"


  //"org.scalanlp" % "breeze" % "2.1.0"
)


//  "org.nd4j" % "nd4j-api" % "1.0.0-beta4",
//  "org.nd4j" % "nd4j-native" % "1.0.0-beta4",
//  "org.nd4j" % "nd4j-common" % "1.0.0-beta4",
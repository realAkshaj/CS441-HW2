ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "CS441-HW2"
  )



libraryDependencies ++= Seq(
  "com.typesafe" % "config" % "1.4.3",                 // For configuration management
  "com.knuddels" % "jtokkit" % "1.1.0",
  "org.slf4j" % "slf4j-log4j12" % "2.0.2",
  "org.apache.hadoop" % "hadoop-client" % "3.3.6",    // Hadoop client
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",// Deeplearning4j
  "org.apache.hadoop" % "hadoop-client" % "3.3.3",
  "org.apache.hadoop" % "hadoop-common" % "3.3.3",
  "org.apache.hadoop" % "hadoop-hdfs" % "3.3.3",
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.3",
  "org.apache.hadoop" % "hadoop-yarn-client" % "3.3.3",
  "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.3",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta6",
  "org.nd4j" % "nd4j-api" % "1.0.0-beta6",
  "org.nd4j" % "nd4j-native" % "1.0.0-beta6",
  "org.nd4j" % "nd4j-common" % "1.0.0-beta6",
  "org.yaml" % "snakeyaml" % "2.0",
  "com.opencsv" % "opencsv" % "5.9",
  "org.apache.spark" %% "spark-core" % "3.5.0",
  "org.apache.spark" %% "spark-sql" % "3.5.0",
  "org.apache.spark" %% "spark-mllib" % "3.5.0",
  "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-M2.1"


  //"org.scalanlp" % "breeze" % "2.1.0"
)
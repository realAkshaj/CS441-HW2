package mainapp

import org.apache.spark.sql.SparkSession
import datahandler.SlidingWindowWithPositionalEmbedding
import org.apache.hadoop.fs.{FileSystem, Path}
import modelgenerator.SlidingWindowModelTraining
import org.apache.spark.SparkConf

object MainApp {

  def main(args: Array[String]): Unit = {

    // Determine the environment: "local", "spark", or "aws" (default to "local" if no argument is provided)
    val environment = if (args.isEmpty) "local" else args(0).toLowerCase
    val (masterUrl, appName) = environment match {
      case "spark" => ("spark://localhost:7077", "CS441-HW2-Spark")
      case "aws" => ("yarn", "CS441-HW2-AWS") // AWS EMR uses YARN by default
      case _ => ("local[*]", "CS441-HW2-Local")
    }

    // Initialize Spark configuration and session
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    val spark = SparkSession.builder.config(conf).getOrCreate()
    println(s"Running on $environment environment")

    // Define input and output paths based on environment
    val (inputPath, outputPath, tempOutputPath) = environment match {
      case "aws" =>
        (
          "hdfs:///user/akshaj/input/final_input.csv",  // HDFS for input
          "hdfs:///user/akshaj/output/sliding_window_dataset",   // HDFS for output
          "hdfs:///user/akshaj/output/temp_sliding_window_dataset" // HDFS for temporary output
        )
      case "spark" =>
        (
          "hdfs://localhost:9000/user/aksha/input/final_input.csv", // HDFS path for standalone Spark
          "hdfs://localhost:9000/output/user/aksha/sliding_window_dataset",
          "hdfs://localhost:9000/output/user/aksha/temp_sliding_window_dataset"
        )
      case _ =>
        (
          "src/main/correctedInput/final_input.csv",
          "src/main/correctedInput/sliding_window_dataset",
          "src/main/correctedInput/temp_sliding_window_dataset"
        )
    }

    // Set standardized output file name
    val standardizedFileName = "sliding_window_dataset.csv"
    val windowSize = 5

    // Load data
    val dataRDD = SlidingWindowWithPositionalEmbedding.loadData(spark, inputPath)
    val slidingWindowsRDD = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(dataRDD, windowSize)

    // Convert to DataFrame and write output
    import spark.implicits._
    val slidingWindowDF = slidingWindowsRDD.toDF("inputWindow", "target")

    slidingWindowDF.coalesce(1)
      .write
      .option("header", "true")
      .csv(tempOutputPath)

    // Rename the output file to a standardized name
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val tempDir = new Path(tempOutputPath)
    val finalDir = new Path(outputPath)
    val targetFile = new Path(s"$outputPath/$standardizedFileName")

    // Clean up and rename
    if (fs.exists(finalDir)) fs.delete(finalDir, true)
    fs.mkdirs(finalDir)
    val tempFile = fs.globStatus(new Path(s"$tempOutputPath/part-*.csv"))(0).getPath
    fs.rename(tempFile, targetFile)
    fs.delete(tempDir, true)

    println(s"Sliding window dataset saved to $targetFile")

    // Stop the Spark session
    spark.stop()

    // Run the model training
//    val datapath = ""
    SlidingWindowModelTraining.run(Array())  // This calls the model training class
  }
}
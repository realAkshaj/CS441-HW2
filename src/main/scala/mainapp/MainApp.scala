package mainapp

import org.apache.spark.sql.SparkSession
import datahandler.SlidingWindowWithPositionalEmbedding
import org.apache.hadoop.fs.{FileSystem, Path}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.util.ModelSerializer
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import modelgenerator.SlidingWindowModelTraining

object MainApp {

  def main(args: Array[String]): Unit = {

    // Initialize Spark session
    val spark = SparkSession.builder
      .appName("SlidingWindowDataset")
      .master("local[*]") // Use local mode with all available cores
      .getOrCreate()

    // Paths and parameters
    val inputPath = "src/main/correctedInput/final_input.csv"
    val outputPath = "src/main/correctedInput/sliding_window_dataset"
    val tempOutputPath = "src/main/correctedInput/temp_sliding_window_dataset"
    val standardizedFileName = "sliding_window_dataset.csv"
    val windowSize = 4

    // Load data
    val dataRDD = SlidingWindowWithPositionalEmbedding.loadData(spark, inputPath)
    val slidingWindowsRDD = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(dataRDD, windowSize)

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

    if (fs.exists(finalDir)) {
      fs.delete(finalDir, true)
    }
    fs.mkdirs(finalDir)
    val tempFile = fs.globStatus(new Path(s"$tempOutputPath/part-*.csv"))(0).getPath
    fs.rename(tempFile, targetFile)
    fs.delete(tempDir, true)

    println(s"Sliding window dataset saved to $targetFile")

    // Stop the Spark session
    spark.stop()

    SlidingWindowModelTraining.run(Array())  //code to run the SlidingWindow Model Generator
  }
}

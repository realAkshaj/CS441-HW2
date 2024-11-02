package mainapp

import org.apache.spark.sql.SparkSession
import datahandler.SlidingWindowWithPositionalEmbedding
import org.apache.hadoop.fs.{FileSystem, Path}


object MainApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("SlidingWindowDataset")
      .master("local[*]") // Use local mode with all available cores
      .getOrCreate()

    val inputPath = "src/main/correctedInput/final_input.csv"
    val outputPath = "src/main/correctedInput/sliding_window_dataset"
    val tempOutputPath = "src/main/correctedInput/temp_sliding_window_dataset"
    val standardizedFileName = "sliding_window_dataset.csv"
    val windowSize = 4

    // Load data
    val dataRDD = SlidingWindowWithPositionalEmbedding.loadData(spark, inputPath)

    // Create sliding windows with positional embeddings
    val slidingWindowsRDD = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(dataRDD, windowSize)

    // Convert to DataFrame and save as a single file
    import spark.implicits._
    val slidingWindowDF = slidingWindowsRDD.toDF("inputWindow", "target")

    // Coalesce to a single partition to output a single file
    slidingWindowDF.coalesce(1)
      .write
      .option("header", "true")
      .csv(tempOutputPath)

    // Rename the output file to a standardized name
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val tempDir = new Path(tempOutputPath)
    val finalDir = new Path(outputPath)
    val targetFile = new Path(s"$outputPath/$standardizedFileName")

    // Delete the final directory if it already exists
    if (fs.exists(finalDir)) {
      fs.delete(finalDir, true)
    }

    // Create the final directory
    fs.mkdirs(finalDir)

    // Move the CSV file from temporary directory to final directory with standardized name
    val tempFile = fs.globStatus(new Path(s"$tempOutputPath/part-*.csv"))(0).getPath
    fs.rename(tempFile, targetFile)

    // Clean up temporary directory
    fs.delete(tempDir, true)

    println(s"Sliding window dataset saved to $targetFile")

    // Stop the Spark session
    spark.stop()
  }
}

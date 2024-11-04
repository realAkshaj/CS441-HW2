package mainapp

import datahandler.SlidingWindowWithPositionalEmbedding
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

class DataLoadingTest extends AnyFlatSpec with Matchers {

  "Data loading" should "load data correctly from CSV" in {
    val spark = SparkSession.builder.master("local[*]").appName("DataLoadingTest").getOrCreate()

    val dataPath = "src/main/correctedInput/final_input.csv" // Path to a small test dataset
    val rawDataRDD: RDD[(String, Int, Array[Double])] = SlidingWindowWithPositionalEmbedding.loadData(spark, dataPath)

    // Map to RDD[DataSet]
    val dataRDD: RDD[DataSet] = rawDataRDD.map { case (_, label, features) =>
      val featureArray = Nd4j.create(features)
      val labelArray = Nd4j.create(Array(label.toDouble)) // Use label array as needed, or adjust for multiple labels
      new DataSet(featureArray, labelArray)
    }

    // Assertions to verify data loading
    dataRDD should not be empty
    dataRDD.count() should be > 0L

    val firstData = dataRDD.first()

    // Print feature and label shapes for debugging
    println(s"Features shape: ${firstData.getFeatures.shape().mkString(", ")}")
    println(s"Labels shape: ${firstData.getLabels.shape().mkString(", ")}")

    // Check the number of elements in the feature array
    firstData.getFeatures.length() shouldEqual 50 // Replace 50 with the expected number of features

    // Check that the label array is non-empty
    firstData.getLabels.length() should be > 0L  // Adjusted to `0L` to match the `Long` type

    spark.stop()
  }
}

package datahandler

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object SlidingWindowWithPositionalEmbedding {

  def loadData(spark: SparkSession, path: String): RDD[(String, Int, Array[Double])] = {
    spark.read
      .option("header", "true")
      .csv(path)
      .rdd.map(row => (row.getString(0), row.getString(1).toInt, row.getString(2).split(" ").map(_.toDouble)))
  }

  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): Array[Array[Double]] = {
    val positionalEncoding = Array.ofDim[Double](windowSize, embeddingDim)
    for (pos <- 0 until windowSize; i <- 0 until embeddingDim by 2) {
      val angle = pos / math.pow(10000, 2.0 * i / embeddingDim)
      positionalEncoding(pos)(i) = math.sin(angle)
      positionalEncoding(pos)(i + 1) = math.cos(angle)
    }
    positionalEncoding
  }

  def createSlidingWindowsWithPositionalEmbedding(
                                                   data: RDD[(String, Int, Array[Double])],
                                                   windowSize: Int
                                                 ): RDD[(String, String)] = {
    val positionalEmbedding = computePositionalEmbedding(windowSize, 128)

    data.mapPartitions { iter =>
      val tokens = iter.toArray
      val dataSet = for (i <- 0 until tokens.length - windowSize if i + windowSize < tokens.length) yield {
        val inputWindow = tokens.slice(i, i + windowSize).map(_._3) // Extract embeddings for window
        val positionAwareEmbedding = inputWindow.zip(positionalEmbedding).map { case (embed, posEmbed) =>
          embed.zip(posEmbed).map { case (e, p) => e + p } // Add positional embedding
        }

        // Flatten input embeddings into a single string
        val inputString = positionAwareEmbedding.map(_.mkString(" ")).mkString("|")

        // Ensure we do not access beyond the bounds for the target token
        if (i + windowSize < tokens.length) {
          val targetEmbedding = tokens(i + windowSize)._3.mkString(" ")
          Some((inputString, targetEmbedding))
        } else {
          None // Skip if we can't get a target token
        }
      }
      dataSet.flatten.iterator
    }
  }
}

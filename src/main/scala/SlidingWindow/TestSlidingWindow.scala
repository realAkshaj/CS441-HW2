package SlidingWindow

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object TestSlidingWindow {

  def main(args: Array[String]): Unit = {
    // Sample tokenized sentence (using token IDs)
    val tokens = Seq(101, 102, 103, 104, 105, 106)

    // Sample embeddings for each token (normally pre-trained in Homework 1)
    val embeddings = Map(
      101 -> Nd4j.create(Array(0.1, 0.2, 0.3)),  // Example embedding with 3 dimensions for simplicity
      102 -> Nd4j.create(Array(0.2, 0.3, 0.4)),
      103 -> Nd4j.create(Array(0.3, 0.4, 0.5)),
      104 -> Nd4j.create(Array(0.4, 0.5, 0.6)),
      105 -> Nd4j.create(Array(0.5, 0.6, 0.7)),
      106 -> Nd4j.create(Array(0.6, 0.7, 0.8))
    )

    // Set window size
    val windowSize = 3

    // Call the sliding window creation function
    val slidingWindows = SlidingWindowDataset.createSlidingWindows(tokens, embeddings, windowSize)

    // Output the results for testing
    println(s"Number of sliding windows: ${slidingWindows.size}")

    // Print out each window and target for visual inspection
    slidingWindows.foreach { case DataSet(input, target) =>
      println(s"Input Window Embeddings:\n$input")
      println(s"Target Embedding:\n$target")
      println("-------------------")
    }
  }
}

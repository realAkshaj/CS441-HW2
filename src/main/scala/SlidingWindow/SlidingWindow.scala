package SlidingWindow

import utils.EmbeddingLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object SlidingWindow {

  // Method to create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindows(embeddings: Seq[EmbeddingLoader.TokenEmbedding], windowSize: Int): Seq[(INDArray, INDArray)] = {
    embeddings.sliding(windowSize + 1).collect {
      case window if window.length == windowSize + 1 =>
        val inputWindow = window.take(windowSize).map(_.embedding) // Extract embeddings for input
        val targetEmbedding = window.last.embedding               // Extract target embedding

        // Stack embeddings in inputWindow and return with target
        (Nd4j.vstack(inputWindow: _*), targetEmbedding)
    }.toSeq
  }

  // Method to test and print sliding windows
  def testSlidingWindow(slidingWindows: Seq[(INDArray, INDArray)]): Unit = {
    slidingWindows.foreach { case (input, target) =>
      println(s"Input Window Embeddings:\n$input")
      println(s"Target Embedding:\n$target")
      println("-----")
    }
    println(s"Total sliding windows created: ${slidingWindows.size}")
  }

  // Main method for running the sliding window creation and testing
  def main(args: Array[String]): Unit = {
    val filePath = "src/main/output/CombinedOutput.csv" // Replace with actual file path
    val windowSize = 4 // Set desired window size

    // Load embeddings from CSV and create sliding windows
    val embeddings = EmbeddingLoader.loadEmbeddings(filePath)
    val slidingWindows = createSlidingWindows(embeddings, windowSize)

    // Test sliding windows output
    testSlidingWindow(slidingWindows)
  }
}

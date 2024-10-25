package SlidingWindow

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import scala.collection.mutable.ListBuffer

// Case class to hold input and target data
case class DataSet(input: INDArray, target: INDArray)

object SlidingWindowDataset {

  // Function to create sliding windows with positional embeddings
  def createSlidingWindows(tokens: Seq[Int], embeddings: Map[Int, INDArray], windowSize: Int): Seq[DataSet] = {
    val dataSetList = ListBuffer[DataSet]()

    for (i <- 0 until tokens.length - windowSize) {
      // Extract the input window
      val inputTokens = tokens.slice(i, i + windowSize)
      val inputEmbeddings = inputTokens.map(token => embeddings(token)).toArray

      // Combine the embeddings into a single INDArray (for the input window)
      val inputWindow = Nd4j.vstack(inputEmbeddings: _*)

      // Add positional embeddings (this could be sinusoidal or learnable)
      val positionalEmbedding = computePositionalEmbedding(windowSize, inputEmbeddings(0).columns())
      val positionAwareEmbedding = inputWindow.add(positionalEmbedding)

      // Extract the target token (the token after the input window)
      val targetToken = tokens(i + windowSize)
      val targetEmbedding = embeddings(targetToken)

      // Create DataSet object and add to the list
      dataSetList += DataSet(positionAwareEmbedding, targetEmbedding)
    }

    dataSetList.toSeq
  }

  // Function to compute positional embeddings (sinusoidal example)
  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): INDArray = {
    // Ensure the embeddingDim is large enough to handle both i and i + 1
    val positionalEncoding = Nd4j.zeros(windowSize.toLong, embeddingDim.toLong)  // Use Long dimensions

    for (pos <- 0 until windowSize) {
      for (i <- 0 until embeddingDim by 2) {
        if (i + 1 < embeddingDim) {  // Ensure i+1 is within bounds
          val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
          positionalEncoding.putScalar(Array(pos, i), math.sin(angle))
          positionalEncoding.putScalar(Array(pos, i + 1), math.cos(angle))
        }
      }
    }

    positionalEncoding
  }
}

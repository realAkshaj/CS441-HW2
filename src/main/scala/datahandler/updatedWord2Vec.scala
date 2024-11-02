package datahandler

import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import java.io.File


object updatedWord2Vec {


  def trainModel(sentences: List[List[Int]]): Word2Vec = {
    // Create a custom sentence iterator
    val iterator = new CustomSentenceIterator(sentences)

    val word2Vec = new Word2Vec.Builder()
      .minWordFrequency(2) // Minimum frequency of words to be included - Lower threshold to include more words
      .iterations(10) // Number of training iterations - Fewer iterations for quicker training
      .layerSize(50) // Size of the word vectors - Moderate embedding size
      .seed(42) // Random seed for reproducibility
      .windowSize(5) // Context window size for embeddings - looks 5 words before and after
      .workers(4) // Parallelism - make use of the multicore cpu
      .iterate(iterator) // Use the custom iterator
      .build()

    // Train the model
    word2Vec.fit()
    word2Vec
  }

  def saveModel(word2Vec: Word2Vec, filePath: String): Unit = {
    WordVectorSerializer.writeWord2VecModel(word2Vec, new File(filePath))
  }

  def loadSentences(lines: List[String]): List[List[Int]] = {
    lines.map { line =>
      // Remove square brackets and trim
      val cleanedLine = line.replace("[", "").replace("]", "").trim
      // Check if cleanedLine is not empty
      if (cleanedLine.nonEmpty) {
        // If it contains a single value, directly convert to Int, else split by comma
        if (cleanedLine.contains(",")) {
          cleanedLine.split(",").map(_.trim.toInt).toList
        } else {
          List(cleanedLine.toInt) // Directly convert single value to List[Int]
        }
      } else {
        List.empty[Int] // Handle the case for empty lines
      }
    }
  }


}

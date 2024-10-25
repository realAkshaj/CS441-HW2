package datacombiner

import scala.io.Source
import java.io.{BufferedWriter, FileWriter, File}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray

// Case classes to hold the data
case class TokenData(word: String, token: Int, frequency: Int)
case class EmbeddingData(token: Int, embedding: INDArray)
case class CombinedData(word: String, token: Int, frequency: Int, embedding: INDArray)

object DataCombiner {

  // Function to load the TokenizationOutput file
  def loadTokenizationOutput(filePath: String): Seq[TokenData] = {
    val tokenizationData = Source.fromFile(filePath).getLines().toSeq
    tokenizationData.map { line =>
      val cols = line.split("\\s*-\\s*").map(_.trim)
      val word = cols(0)
      val token = cols(1).toInt
      val frequency = cols(2).toInt
      TokenData(word, token, frequency)
    }
  }

  // Function to load the EmbeddingOutput file
  def loadEmbeddingOutput(filePath: String): Map[Int, INDArray] = {
    val embeddingData = Source.fromFile(filePath).getLines().toSeq
    embeddingData.map { line =>
      val cols = line.split("\\s+")
      val token = cols(0).toInt
      val embedding = Nd4j.create(cols(1).split(",").map(_.toDouble))
      token -> embedding
    }.toMap
  }

  // Function to combine the tokenization and embedding datasets
  def combineDatasets(tokenData: Seq[TokenData], embeddingData: Map[Int, INDArray]): Seq[CombinedData] = {
    tokenData.flatMap { tokenInfo =>
      embeddingData.get(tokenInfo.token).map { embedding =>
        CombinedData(tokenInfo.word, tokenInfo.token, tokenInfo.frequency, embedding)
      }
    }
  }

  // Function to save the combined data to a file
  def saveCombinedData(outputFilePath: String, combinedData: Seq[CombinedData]): Unit = {
    // Extract the directory path and ensure it exists
    val outputDir = new File(outputFilePath).getParentFile
    if (!outputDir.exists()) {
      outputDir.mkdirs() // Create the output directory if it doesn't exist
    }

    // Now proceed to save the file
    val writer = new BufferedWriter(new FileWriter(outputFilePath))

    // Write header
    writer.write("Word,Token,Frequency,Embedding\n")

    // Write each entry
    combinedData.foreach { data =>
      val embeddingString = data.embedding.data().asDouble().mkString(",")
      writer.write(s"${data.word},${data.token},${data.frequency},$embeddingString\n")
    }

    writer.close()
  }
}

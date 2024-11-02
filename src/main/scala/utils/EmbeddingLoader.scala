package utils

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import scala.io.Source
import scala.util.Try

object EmbeddingLoader {

  // Case class to store word embeddings
  case class TokenEmbedding(word: String, token: Int, frequency: Int, embedding: INDArray)

  // Method to load embeddings from a CSV file
  def loadEmbeddings(filePath: String): Seq[TokenEmbedding] = {
    val lines = Source.fromFile(filePath).getLines().toSeq
    val embeddings = lines.drop(1).flatMap { line =>
      val tokens = line.split(",")
      if (tokens.length < 5) None // Ensure there's enough data
      else {
        val word = tokens(0)
        val token = Try(tokens(1).toInt).getOrElse(0)
        val frequency = Try(tokens(2).toInt).getOrElse(0)
        val embeddingValues = tokens.drop(3).flatMap(value => Try(value.toDouble).toOption)
        val embedding = Nd4j.create(embeddingValues.toArray)
        Some(TokenEmbedding(word, token, frequency, embedding))
      }
    }
    embeddings
  }

  // Method to test loaded embeddings
  def testEmbeddings(embeddings: Seq[TokenEmbedding]): Unit = {
    embeddings.foreach { emb =>
      println(s"Word: ${emb.word}, Token: ${emb.token}, Frequency: ${emb.frequency}, Embedding: ${emb.embedding}")
    }
    println(s"Total embeddings loaded: ${embeddings.size}")
  }
}


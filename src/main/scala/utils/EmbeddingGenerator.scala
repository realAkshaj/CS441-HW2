package utils

import datahandler.{CustomSentenceIterator, updatedTokenizer, updatedWord2Vec}
import org.deeplearning4j.models.word2vec.Word2Vec

import scala.collection.JavaConverters._
import scala.io.Source
import java.io.{File, PrintWriter}
import org.nd4j.linalg.api.ndarray.INDArray


object EmbeddingGenerator {

  // Method to read and tokenize words from a file
  def readAndTokenizeWords(filePath: String): List[(String, Array[Int])] = {
    val source = Source.fromFile(filePath)
    try {
      source.getLines().flatMap(_.split("\\s+")).toList.map { word =>
        val tokens = updatedTokenizer.tokenize(word) // Use tokenizer to get token IDs
        (word, tokens)
      }
    } finally {
      source.close()
    }
  }

  // Method to generate and save embeddings for each token in each word
  def generateAndSaveWordEmbeddings(model: Word2Vec, wordsWithTokens: List[(String, Array[Int])], outputPath: String): Unit = {
    val outputDir = new File(outputPath).getParentFile
    if (!outputDir.exists()) outputDir.mkdirs() // Create correctedInput directory if it doesn't exist

    val writer = new PrintWriter(outputPath)
    try {
      writer.println("Word,Token,Embedding") // Header for CSV file
      wordsWithTokens.foreach { case (word, tokens) =>
        tokens.foreach { tokenId =>
          val vector: INDArray = model.getWordVectorMatrix(tokenId.toString)
          val embedding = if (vector != null) vector.toDoubleVector.mkString(" ") else Array.fill(50)(0.0).mkString(" ")
          writer.println(s"$word,$tokenId,$embedding")
        }
      }
    } finally {
      writer.close()
    }
  }

  // Run the entire process of reading, tokenizing, generating, and saving embeddings
  def runOnceAndSaveWords(filePath: String): Unit = {
    val outputPath = "src/main/correctedInput/final_input.csv" // Set the specified output path
    val wordsWithTokens = readAndTokenizeWords(filePath) // Read and tokenize words
    val tokenSequences = wordsWithTokens.map(_._2.toList) // Extract token lists for model training
    val model = updatedWord2Vec.trainModel(tokenSequences) // Train model on token sequences
    generateAndSaveWordEmbeddings(model, wordsWithTokens, outputPath) // Save embeddings in format
  }
}






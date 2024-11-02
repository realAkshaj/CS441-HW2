package datahandler

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, IntArrayList, ModelType}
import org.slf4j.LoggerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.io.PrintWriter
import scala.collection.mutable

object updatedTokenizer {

  private val logger = LoggerFactory.getLogger(getClass)

  // Initialize JTokkit's encoding registry
  private val encodingRegistry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = encodingRegistry.getEncodingForModel(ModelType.GPT_4O)

  // Method to generate tokens for a given word
  def tokenize(word: String): Array[Int] = {
    val tokens: IntArrayList = encoding.encode(word) // Tokenize the word using JTokkit
    (0 until tokens.size()).map(tokens.get).toArray // Convert to Array[Int]
  }

  // Method to process input data, compute token frequencies, and save results
  def processAndSave(inputData: List[String], outputPath: String): Unit = {
    val wordFrequencyMap = mutable.LinkedHashMap.empty[String, (Array[Int], Int)]

    // Process each word in the input data
    inputData.foreach { word =>
      val tokens = tokenize(word)
      val (existingTokens, frequency) = wordFrequencyMap.getOrElse(word, (tokens, 0))
      wordFrequencyMap(word) = (existingTokens, frequency + 1)
    }

    // Save the results to the specified output file
    val writer = new PrintWriter(outputPath)
    try {
      writer.println("Word,Token,Frequency") // Header
      wordFrequencyMap.foreach { case (word, (tokens, frequency)) =>
        val tokensString = tokens.mkString(" ")
        writer.println(s"$word,$tokensString,$frequency")
      }
    } finally {
      writer.close()
    }
  }
}

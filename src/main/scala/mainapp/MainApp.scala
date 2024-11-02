package mainapp

import datahandler.updatedWord2Vec._
import org.deeplearning4j.models.word2vec.Word2Vec

import java.io.File
import scala.io.Source

object MainApp { // Ensure this is defined as 'object'

  def main(args: Array[String]): Unit = {

    val inputPath = "src/main/input/tokens"
    def readAllFilesInDirectory(directoryPath: String): String = {
      val directory = new File(directoryPath)

      // Filter for files only and read each file's content
      val allContent = directory.listFiles
        .filter(_.isFile)
        .map { file =>
          val source = Source.fromFile(file)
          try source.getLines().mkString("\n") // Join lines with newline
          finally source.close()
        }
        .mkString("\n") // Join content from all files with newline

      allContent
    }

    val contents = readAllFilesInDirectory(inputPath).split("\n").toList
    val allSentences = loadSentences(contents)
    //    println(allSentences)
    val model: Word2Vec = trainModel(allSentences)
    saveModel(model,"src/main/output/word2vecModel.bin")

//val directory = new File(inputPath)
//println(directory)
  }
  // Paths to input and output files
  //    val inputTokenization = "src/main/input/sharded_text.txt"
  //    val outputTokenization = "src/main/output/updatedTokenization.txt"
  //    val tokenizationFilePath = "src/main/input/TokenizationOutput"
  //    val embeddingFilePath = "src/main/input/EmbeddingOutput"
  //    val outputFilePath = "src/main/output/CombinedOutput.csv"
  //
  //    // Step 1: Load the data using DataCombiner
  //    val tokenizationData = DataCombiner.loadTokenizationOutput(tokenizationFilePath)
  //    val embeddingData = DataCombiner.loadEmbeddingOutput(embeddingFilePath)
  //    // Step 2: Combine the data
  //    val combinedData = DataCombiner.combineDatasets(tokenizationData, embeddingData)
  //    // Step 3: Save the combined data
  //    DataCombiner.saveCombinedData(outputFilePath, combinedData)
  //    println(s"Data combined and saved to $outputFilePath")
  //
  //    val words = readWordsFromFile(inputTokenization)
  //
  //    // Process and save the output using updatedTokenizer
  //    updatedTokenizer.processAndSave(words, outputTokenization)
  //    println(s"Tokenization and frequency saved to $outputTokenization")
  //
  //  }
  //
  //  // Method to read words from the input file
  //  def readWordsFromFile(filePath: String): List[String] = {
  //    val bufferedSource = Source.fromFile(filePath)
  //    try {
  //      bufferedSource.getLines().toList.flatMap(_.split("\\s+")) // Split lines into words
  //    } finally {
  //      bufferedSource.close()
  //    }
  //  }

}

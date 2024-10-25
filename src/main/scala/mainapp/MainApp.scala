package mainapp

import datacombiner.DataCombiner

object MainApp {  // Ensure this is defined as 'object'

  def main(args: Array[String]): Unit = {
    // Paths to input and output files
    val tokenizationFilePath = "src/main/input/TokenizationOutput"
    val embeddingFilePath = "src/main/input/EmbeddingOutput"
    val outputFilePath = "src/main/output/CombinedOutput.csv"

    // Step 1: Load the data using DataCombiner
    val tokenizationData = DataCombiner.loadTokenizationOutput(tokenizationFilePath)
    val embeddingData = DataCombiner.loadEmbeddingOutput(embeddingFilePath)

    // Step 2: Combine the data
    val combinedData = DataCombiner.combineDatasets(tokenizationData, embeddingData)

    // Step 3: Save the combined data
    DataCombiner.saveCombinedData(outputFilePath, combinedData)

    println(s"Data combined and saved to $outputFilePath")
  }
}

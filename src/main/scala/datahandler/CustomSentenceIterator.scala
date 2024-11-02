package datahandler

import org.deeplearning4j.text.sentenceiterator.{SentenceIterator, SentencePreProcessor}

class CustomSentenceIterator(corpus: List[List[Int]]) extends SentenceIterator {
    private var currentIndex = 0

    override def hasNext: Boolean = currentIndex < corpus.size

    override def nextSentence(): String = {
      if (!hasNext) throw new NoSuchElementException("No more sentences available.")
      val sentence = corpus(currentIndex).map(_.toString).mkString(" ")  // Convert Int tokens to strings
      currentIndex += 1
      sentence
    }

    override def reset(): Unit = currentIndex = 0

    override def setPreProcessor(preProcessor: SentencePreProcessor): Unit = {}
    override def getPreProcessor: SentencePreProcessor = null
    override def finish(): Unit = {}
  }

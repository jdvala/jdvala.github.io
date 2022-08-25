+++
title =  "NLP and ML notes"
description =  "Handy dandy notes"
date = "2018-11-02"
author = "Jay Vala"
tags = ["python", "text", "nlp", "analysis", "word2vec", "POS", "TF-IDF", "probability"]
+++

# Notes: Natural Language Processing and Machine Learning


**Question:** What are *word embedding* or what are *word vectors*?

**Answer:** Word embedding are learned representation for text where words that have the same meaning have a similar representation. Word embeddings are in fact a class of techniques where individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one vector and the vector values are learned in a way that resembles a neural network, and hence the technique is often lumped into the field of deep learning.

**Question** What is POS tagging?

**Answer:** Parts of Speech(POS) tagging is a method of assigning Part of Speech such as *Noun*, *Verb*, *adjectivte* etc to each word of the text. 

**Question:** What is TF-IDF matrix?

**Answer:** *Term Frequency Inverse Document Frequency(TF-IDF)* is a numerical statistical method that shows how important a word is to a document in a corpus. It contains two terms,
+ **Term Frequecy**: It is the measure of how frequently a term appears in a document. Every document is of different length and hence it may happen that a term might appear in several times in a document, so as a measure of normalization we divide the term appearence in document by the length of document.

+ **Inverse Document Frequency**: It is the meausre of how important a term is. While calculating Term Frequency every term is given same importance, but it is evedent that terms like "is", "the" don't contribute  and hence they need to be scaled down and the rare once have to scaledup

  * TF = (Number of times term appear in document/Number of document with term in it)
  * IDF = log_e(Total Number of documents/ Number of documents with the term in it)


**Question:** What is Zip's Law?

**Answer:** Zip's Law states that *"If we count the frequency of words and rank them according to the frequency of occurance we can explore relationship between the frequency and its frequency in the list known as rank "r", then Zip's law says that"*
$$ f \propto \frac{1}{r}$$

or in other words, there is a constant $$ k $$ such that
$$ f . r= k$$

**Question:** What is a probability distribution function?

**Answer:** A probability function or probability distribution function distributes a probability mass of 1 throughout given sample space. 

**Question:** What is Uniform Distribution?

**Answer:** A situation where each outcome is equaly likely is called a *Uniform Distribution*.

## Mathematical Definations:
+ **Conditional Probability:** It is updated probability of an event given some prior knowledge.
The conditinal probability of an event A given that event B has occured is:

$$ P(A|B) = \frac{P(A\bigcap B)}{P(B)}$$   

If $P(B) = 0$ we have 

$$P(A \bigcap B) = P(B)P(A|B) = P(A)P(B|A)$$  

+ **Prior Probability:** The probability of an event before we consider our additional knowledge.

+ **Posterior Probability:** The new probability that results from using of additional knowledge. 

+ **Chain Rule:**  

$$P(A_4,A_3,A_2,A_1) = P(A_4|A_3,A_2,A_1).P(A_3|A_2,A_1).P(A_2|A_1).P(A_1)$$

> *Example: Urn 1 has 1 black ball and 2 white balls and Urn 2 has 1 black ball and 3 white balls. Suppose we pick an urn at random and then select a ball from that urnUrn 1 has 1 black ball and 2 white balls and Urn 2 has 1 black ball and 3 white balls. Suppose we pick an urn at random, then what is the probability that we choose a black ball from urn 2*

To answer the above question we can use the chain rule.

Let $P(A)$ be the probabilty of picking urn 2, Hence 

$$P(A) = \frac{1}{2}$$

Then Let $P(B\|A)$ be the probability of picking black ball from urn 2.

$$P(B|A) =\frac{1}{4}$$

Hence, using chain rule we can calculate probaility $P(A,B)$

$$P(A,B) = P(B|A).P(A)$$

$$P(A,B) = \frac{1}{4} * \frac{1}{2}$$

$$P(A,B) = \frac{1}{8}$$


+++
title = "Latent Dirichlet Allocation (Topic Modelling) - BaseLine"
description =  "A baseline for further expriments"
date = "2018-05-27"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "analysis", "LDA", "topic-modelling"]
+++

Topic modeling can be useful when having a large corpus, when we want to unearth the meaning or  of the data we have, which is too large to be done manually.
In simple terms LDA is probabilistic unsupervised models that gives out top topics.
So suppose we have a set of documents. we’ve chosen some fixed number of K topics to discover, and want to use LDA to learn the topic representation of each document and the words associated to each topic. LDA uses collapsed Gibbs sampling. How?

* Go through each document, and randomly assign each word(w) in the document(d) to one of the K topics(t).
* This random assignment already gives us topic assignment and the topic distribution but they won't be good at all.
* So to importve upon them we have to,

    * Go through every word in the document and compute two things
        * p(topic t \| document d), i.e. the words(w) in document d currently assgin to topic t.
        * p(word w \| topic t),  i.e. the assignment of topic  over all the document because of this word(w).
        * This a generative model and hence we have to reassign a new topic to a word and repeat.
   
To learn about it more and to get an intution behind the idea how LDA works [click here](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) I got the intusion from there.


### Steps 
   * Pre-processing and training corpus creation
   * Building dictionary
   * Feature extraction
   * LDA model training
   
Pre-processing text for LDA is a little bit different for LDA than what I did for RNN screated in this [post](https://jdvala.github.io/blog.io/thesis/2018/05/23/Creating-Data-Set-Again-!.html).


```python
# Lets pre-process them.
import os
import random
from nltk.corpus import stopwords
import string
import sys
import spacy
import re
import logging
from gensim.models.ldamodel import LdaModel as Lda
from gensim import corpora
```


```python
# NLP model from spacy
nlp = spacy.load('en')
```

+++--
### Pre-processing Text


```python
def preprocess(text):
    """Returns text after preprocessing
    :params:list of text
    :returns:list of text after manipulation"""
    
    pun = string.punctuation+'$€¥₹£|–'
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    punct = " ".join([i for i in text.lower().split() if i.split() not in pun.split()])
    stop = " ".join([i for i in punct.lower().split() if i not in stopwords.words('english')])
    digit = re.sub(r'\d+','',stop)
    # removing punct again
    punct_ = regex.sub('',digit)
    doc = nlp(punct_)
    lam = " ".join(word.lemma_ for word in doc)
    x = lam.split()
    y = [s for s in x if len(s) > 2]
    return y
```


```python
# Lets load text from every doc into a list 

document_list = []

for root, dirs, files in os.walk('/home/jay/Thesis_1/Data/Data_EN'):
    for file in files:
        if file != 'log.txt':
            with open(os.path.join(root, file), 'r') as f:
                document_list.append(f.read())
```


```python
# Lets divide the documents into test and train set
# I am taking 20% of documents for test set, but before that lets just suffle it.
random.shuffle(document_list)
train = document_list[round(len(document_list)*.2):]
test = document_list[:round(len(document_list)*.2)]
```


```python
len(train)
```




    3692




```python
# preprocess the training set
cleaned = [preprocess(doc) for doc in train]
```

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
### Dictionary Building
For dictionary building gensim requires all the words in corpus. So lets create a list of words in dictionary


```python
# For building dictionary I will use gensim.
# Dictionary are nothing but every unique term with its unique id as we have already created for training RNNs.
# We can also create 'hashdictionary' were it uses hashing algorithm which will increase speed, but I will not worry
# about it as my corpus is small

dictionary = corpora.Dictionary(cleaned)
```


```python
print(dictionary)
```

    Dictionary(19634 unique tokens: ['-PRON-', 'access', 'achieve', 'achievement', 'act']...)


Now that dictonary is created we need to filter out the dictonary, we will filter out the words that occur in less than 4 document and words that occur in more than 40% of the documents. We do this because these words do not contibute in the different themes and topics that are in the corpus.



```python
# removing extremes
dictionary.filter_extremes(no_below=4, no_above=0.4)
```


```python
print(dictionary)
```

    Dictionary(8231 unique tokens: ['-PRON-', 'access', 'achieve', 'achievement', 'adapt']...)


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
### Feature Extraction
Now that dictionary is created, moving on to the next step, extracting features. Gensim provides use to necessary tools to extract features out of the courpus.
Feature extraction is nothing but generating but the frequencies of all the words in the vocabulary for that particular word.


```python
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned]
```

+++++++++++++++++++++
### Model Building


```python
# As I know that I have only 32 topics in the corpus, I will set the num_topic argument as 32
# To see the progress I added loggig as suggested in Gensim Tutorial 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ldamodel = Lda(doc_term_matrix, num_topics=32, id2word = dictionary, passes=50, iterations=500)
```

    2018-05-28 18:52:24,623 : INFO : using symmetric alpha at 0.03125
    2018-05-28 18:52:24,627 : INFO : using symmetric eta at 0.03125
    2018-05-28 18:52:24,631 : INFO : using serial LDA version on this node
    2018-05-28 18:52:24,696 : INFO : running online (multi-pass) LDA training, 32 topics, 50 passes over the supplied corpus of 3692 documents, updating model once every 2000 documents, evaluating perplexity every 3692 documents, iterating 500x with a convergence threshold of 0.001000
    2018-05-28 18:52:24,698 : INFO : PROGRESS: pass 0, at document #2000/3692
    2018-05-28 18:53:04,688 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:53:04,760 : INFO : topic #14 (0.031): 0.013*"research" + 0.010*"innovation" + 0.009*"sector" + 0.007*"protection" + 0.007*"woman" + 0.006*"europe" + 0.006*"medium" + 0.005*"content" + 0.005*"financial" + 0.005*"interactive"
    2018-05-28 18:53:04,763 : INFO : topic #1 (0.031): 0.007*"person" + 0.007*"directive" + 0.006*"product" + 0.006*"right" + 0.005*"article" + 0.005*"request" + 0.005*"network" + 0.004*"law" + 0.004*"mercury" + 0.004*"agency"
    2018-05-28 18:53:04,764 : INFO : topic #29 (0.031): 0.009*"research" + 0.007*"human" + 0.007*"right" + 0.007*"health" + 0.006*"directive" + 0.005*"employment" + 0.005*"animal" + 0.005*"law" + 0.004*"woman" + 0.004*"committee"
    2018-05-28 18:53:04,766 : INFO : topic #22 (0.031): 0.013*"financial" + 0.010*"fund" + 0.009*"education" + 0.006*"directive" + 0.005*"strategy" + 0.005*"training" + 0.004*"instrument" + 0.004*"management" + 0.004*"nuclear" + 0.004*"safety"
    2018-05-28 18:53:04,768 : INFO : topic #3 (0.031): 0.023*"passenger" + 0.014*"tax" + 0.013*"directive" + 0.010*"right" + 0.008*"air" + 0.007*"car" + 0.007*"travel" + 0.007*"carrier" + 0.006*"liability" + 0.005*"water"
    2018-05-28 18:53:04,769 : INFO : topic diff=18.296263, rho=1.000000
    2018-05-28 18:53:22,841 : INFO : -7.718 per-word bound, 210.6 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 18:53:22,843 : INFO : PROGRESS: pass 0, at document #3692/3692
    2018-05-28 18:53:35,697 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 18:53:35,757 : INFO : topic #11 (0.031): 0.015*"custom" + 0.014*"security" + 0.009*"directive" + 0.009*"control" + 0.007*"law" + 0.006*"trade" + 0.006*"document" + 0.005*"issue" + 0.005*"management" + 0.005*"common"
    2018-05-28 18:53:35,758 : INFO : topic #10 (0.031): 0.017*"euro" + 0.011*"capital" + 0.009*"financial" + 0.008*"directive" + 0.007*"right" + 0.007*"pension" + 0.007*"coin" + 0.006*"company" + 0.006*"bank" + 0.006*"care"
    2018-05-28 18:53:35,760 : INFO : topic #1 (0.031): 0.013*"person" + 0.008*"directive" + 0.007*"citizen" + 0.007*"network" + 0.006*"right" + 0.006*"sis" + 0.006*"contract" + 0.006*"request" + 0.006*"drug" + 0.005*"product"
    2018-05-28 18:53:35,762 : INFO : topic #18 (0.031): 0.010*"directive" + 0.007*"application" + 0.007*"minor" + 0.007*"condition" + 0.006*"applicant" + 0.006*"air" + 0.006*"network" + 0.005*"lay" + 0.005*"import" + 0.005*"common"
    2018-05-28 18:53:35,764 : INFO : topic #26 (0.031): 0.015*"treaty" + 0.011*"financial" + 0.007*"procedure" + 0.006*"progress" + 0.006*"acquis" + 0.006*"security" + 0.005*"court" + 0.004*"agreement" + 0.004*"rate" + 0.004*"government"
    2018-05-28 18:53:35,765 : INFO : topic diff=0.868584, rho=0.707107
    2018-05-28 18:53:35,767 : INFO : PROGRESS: pass 1, at document #2000/3692
    2018-05-28 18:53:47,128 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:53:47,218 : INFO : topic #28 (0.031): 0.026*"right" + 0.016*"datum" + 0.009*"protection" + 0.009*"judicial" + 0.008*"justice" + 0.007*"directive" + 0.007*"treaty" + 0.006*"security" + 0.006*"fundamental" + 0.006*"matter"
    2018-05-28 18:53:47,222 : INFO : topic #24 (0.031): 0.020*"general" + 0.018*"interest" + 0.015*"paper" + 0.013*"research" + 0.013*"medium" + 0.010*"green" + 0.008*"consultation" + 0.008*"sport" + 0.007*"organisation" + 0.007*"society"
    2018-05-28 18:53:47,225 : INFO : topic #11 (0.031): 0.018*"custom" + 0.015*"security" + 0.010*"control" + 0.008*"directive" + 0.007*"trade" + 0.007*"document" + 0.007*"internal" + 0.006*"law" + 0.006*"management" + 0.006*"issue"
    2018-05-28 18:53:47,229 : INFO : topic #5 (0.031): 0.019*"transport" + 0.016*"regional" + 0.015*"region" + 0.015*"vehicle" + 0.011*"fund" + 0.010*"road" + 0.010*"partnership" + 0.009*"initiative" + 0.008*"sea" + 0.008*"integration"
    2018-05-28 18:53:47,231 : INFO : topic #26 (0.031): 0.015*"treaty" + 0.012*"financial" + 0.007*"procedure" + 0.006*"budgetary" + 0.006*"deficit" + 0.006*"progress" + 0.006*"rate" + 0.006*"court" + 0.005*"acquis" + 0.005*"government"
    2018-05-28 18:53:47,233 : INFO : topic diff=0.585917, rho=0.509912
    2018-05-28 18:54:00,950 : INFO : -7.385 per-word bound, 167.2 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 18:54:00,951 : INFO : PROGRESS: pass 1, at document #3692/3692
    2018-05-28 18:54:10,696 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 18:54:10,778 : INFO : topic #31 (0.031): 0.020*"progress" + 0.018*"accession" + 0.016*"candidate" + 0.013*"acquis" + 0.010*"environment" + 0.009*"turkey" + 0.009*"sec" + 0.009*"partnership" + 0.009*"priority" + 0.008*"negotiation"
    2018-05-28 18:54:10,780 : INFO : topic #3 (0.031): 0.024*"directive" + 0.024*"passenger" + 0.023*"air" + 0.021*"transport" + 0.021*"tax" + 0.012*"carrier" + 0.011*"vehicle" + 0.010*"vat" + 0.009*"emission" + 0.009*"rail"
    2018-05-28 18:54:10,781 : INFO : topic #18 (0.031): 0.015*"directive" + 0.010*"minor" + 0.010*"rail" + 0.009*"application" + 0.008*"condition" + 0.008*"interoperability" + 0.008*"network" + 0.008*"standard" + 0.007*"lay" + 0.007*"agency"
    2018-05-28 18:54:10,784 : INFO : topic #30 (0.031): 0.019*"security" + 0.012*"border" + 0.010*"external" + 0.008*"migration" + 0.007*"defence" + 0.007*"strategy" + 0.006*"relation" + 0.006*"management" + 0.006*"crisis" + 0.006*"agreement"
    2018-05-28 18:54:10,785 : INFO : topic #29 (0.031): 0.017*"research" + 0.014*"woman" + 0.013*"human" + 0.012*"right" + 0.011*"child" + 0.010*"health" + 0.008*"man" + 0.007*"equality" + 0.006*"scientific" + 0.006*"protection"
    2018-05-28 18:54:10,787 : INFO : topic diff=0.722667, rho=0.509912
    2018-05-28 18:54:10,790 : INFO : PROGRESS: pass 2, at document #2000/3692
    2018-05-28 18:54:21,708 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:54:21,815 : INFO : topic #27 (0.031): 0.018*"datum" + 0.010*"access" + 0.009*"protection" + 0.007*"internet" + 0.006*"plan" + 0.006*"partner" + 0.006*"regional" + 0.006*"network" + 0.006*"transport" + 0.005*"asylum"
    2018-05-28 18:54:21,818 : INFO : topic #3 (0.031): 0.026*"transport" + 0.024*"air" + 0.023*"directive" + 0.023*"passenger" + 0.020*"tax" + 0.012*"vehicle" + 0.012*"carrier" + 0.010*"rail" + 0.009*"vat" + 0.009*"emission"
    2018-05-28 18:54:21,819 : INFO : topic #31 (0.031): 0.022*"progress" + 0.019*"accession" + 0.017*"candidate" + 0.016*"acquis" + 0.011*"turkey" + 0.010*"sec" + 0.010*"negotiation" + 0.010*"environment" + 0.009*"enlargement" + 0.009*"capacity"
    2018-05-28 18:54:21,820 : INFO : topic #14 (0.031): 0.025*"research" + 0.018*"people" + 0.018*"innovation" + 0.017*"young" + 0.016*"youth" + 0.014*"cultural" + 0.011*"europe" + 0.009*"sector" + 0.008*"culture" + 0.008*"field"
    2018-05-28 18:54:21,822 : INFO : topic #18 (0.031): 0.017*"directive" + 0.013*"rail" + 0.011*"interoperability" + 0.010*"network" + 0.010*"minor" + 0.010*"agency" + 0.009*"equipment" + 0.008*"application" + 0.008*"technical" + 0.008*"condition"
    2018-05-28 18:54:21,823 : INFO : topic diff=0.694411, rho=0.454264
    2018-05-28 18:54:34,992 : INFO : -7.278 per-word bound, 155.2 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 18:54:34,993 : INFO : PROGRESS: pass 2, at document #3692/3692
    2018-05-28 18:54:43,851 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 18:54:43,920 : INFO : topic #20 (0.031): 0.022*"trade" + 0.011*"agreement" + 0.011*"consumer" + 0.010*"committee" + 0.009*"standard" + 0.008*"sector" + 0.007*"product" + 0.007*"business" + 0.007*"legislation" + 0.006*"directive"
    2018-05-28 18:54:43,924 : INFO : topic #27 (0.031): 0.020*"datum" + 0.011*"access" + 0.010*"protection" + 0.007*"plan" + 0.007*"internet" + 0.006*"network" + 0.006*"partner" + 0.006*"region" + 0.006*"regional" + 0.006*"instrument"
    2018-05-28 18:54:43,928 : INFO : topic #9 (0.031): 0.018*"financial" + 0.012*"crime" + 0.011*"terrorist" + 0.009*"fraud" + 0.009*"prevention" + 0.009*"infrastructure" + 0.009*"combat" + 0.008*"noise" + 0.008*"terrorism" + 0.008*"money"
    2018-05-28 18:54:43,931 : INFO : topic #3 (0.031): 0.028*"transport" + 0.026*"air" + 0.023*"directive" + 0.022*"passenger" + 0.019*"tax" + 0.015*"vehicle" + 0.012*"carrier" + 0.011*"vat" + 0.011*"duty" + 0.009*"emission"
    2018-05-28 18:54:43,937 : INFO : topic #13 (0.031): 0.041*"directive" + 0.019*"law" + 0.014*"court" + 0.013*"right" + 0.010*"person" + 0.010*"legal" + 0.009*"proceeding" + 0.008*"eec" + 0.008*"worker" + 0.008*"case"
    2018-05-28 18:54:43,940 : INFO : topic diff=0.752502, rho=0.454264
    2018-05-28 18:54:43,942 : INFO : PROGRESS: pass 3, at document #2000/3692
    2018-05-28 18:54:54,499 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:54:54,594 : INFO : topic #18 (0.031): 0.020*"directive" + 0.016*"rail" + 0.013*"interoperability" + 0.013*"agency" + 0.012*"equipment" + 0.012*"network" + 0.011*"technical" + 0.010*"minor" + 0.009*"railway" + 0.009*"condition"
    2018-05-28 18:54:54,595 : INFO : topic #12 (0.031): 0.027*"fishing" + 0.017*"fishery" + 0.012*"committee" + 0.011*"vessel" + 0.010*"ecb" + 0.010*"bank" + 0.009*"humanitarian" + 0.009*"central" + 0.008*"group" + 0.007*"financial"
    2018-05-28 18:54:54,597 : INFO : topic #1 (0.031): 0.018*"person" + 0.012*"contract" + 0.010*"citizen" + 0.010*"drug" + 0.009*"condition" + 0.008*"request" + 0.008*"network" + 0.008*"entry" + 0.007*"residence" + 0.007*"right"
    2018-05-28 18:54:54,598 : INFO : topic #4 (0.031): 0.019*"right" + 0.015*"agreement" + 0.014*"europol" + 0.013*"online" + 0.012*"property" + 0.012*"patent" + 0.012*"protection" + 0.011*"intellectual" + 0.010*"access" + 0.010*"enforcement"
    2018-05-28 18:54:54,600 : INFO : topic #15 (0.031): 0.057*"directive" + 0.031*"safety" + 0.021*"health" + 0.018*"risk" + 0.014*"worker" + 0.013*"eec" + 0.010*"exposure" + 0.010*"requirement" + 0.010*"protection" + 0.009*"convention"
    2018-05-28 18:54:54,602 : INFO : topic diff=0.675248, rho=0.413591
    2018-05-28 18:55:07,337 : INFO : -7.219 per-word bound, 149.0 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 18:55:07,339 : INFO : PROGRESS: pass 3, at document #3692/3692
    2018-05-28 18:55:15,041 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 18:55:15,124 : INFO : topic #11 (0.031): 0.027*"security" + 0.021*"custom" + 0.015*"control" + 0.010*"export" + 0.009*"document" + 0.009*"internal" + 0.008*"common" + 0.007*"trade" + 0.007*"management" + 0.007*"issue"
    2018-05-28 18:55:15,127 : INFO : topic #6 (0.031): 0.037*"product" + 0.029*"food" + 0.020*"consumer" + 0.014*"agency" + 0.014*"health" + 0.007*"foodstuff" + 0.007*"name" + 0.007*"application" + 0.007*"modify" + 0.006*"agricultural"
    2018-05-28 18:55:15,129 : INFO : topic #20 (0.031): 0.026*"trade" + 0.013*"consumer" + 0.011*"agreement" + 0.010*"committee" + 0.010*"standard" + 0.008*"sector" + 0.008*"business" + 0.007*"legislation" + 0.007*"product" + 0.006*"access"
    2018-05-28 18:55:15,131 : INFO : topic #3 (0.031): 0.033*"transport" + 0.027*"air" + 0.022*"passenger" + 0.022*"directive" + 0.018*"vehicle" + 0.017*"tax" + 0.012*"carrier" + 0.011*"duty" + 0.011*"vat" + 0.010*"airport"
    2018-05-28 18:55:15,133 : INFO : topic #25 (0.031): 0.073*"aid" + 0.023*"competition" + 0.018*"article" + 0.012*"treaty" + 0.012*"grant" + 0.009*"agreement" + 0.009*"guideline" + 0.009*"sector" + 0.008*"application" + 0.008*"investment"
    2018-05-28 18:55:15,134 : INFO : topic diff=0.667393, rho=0.413591
    2018-05-28 18:55:15,136 : INFO : PROGRESS: pass 4, at document #2000/3692
    2018-05-28 18:55:23,624 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:55:23,720 : INFO : topic #22 (0.031): 0.030*"fund" + 0.022*"financial" + 0.017*"education" + 0.016*"training" + 0.015*"project" + 0.011*"eur" + 0.011*"assistance" + 0.011*"period" + 0.011*"million" + 0.010*"instrument"
    2018-05-28 18:55:23,723 : INFO : topic #3 (0.031): 0.035*"transport" + 0.028*"air" + 0.022*"passenger" + 0.021*"directive" + 0.019*"vehicle" + 0.016*"tax" + 0.012*"carrier" + 0.010*"airport" + 0.010*"duty" + 0.009*"vat"
    2018-05-28 18:55:23,726 : INFO : topic #9 (0.031): 0.020*"financial" + 0.017*"crime" + 0.012*"terrorist" + 0.011*"prevention" + 0.010*"combat" + 0.010*"fraud" + 0.010*"terrorism" + 0.009*"infrastructure" + 0.009*"noise" + 0.008*"fight"
    2018-05-28 18:55:23,731 : INFO : topic #23 (0.031): 0.036*"convention" + 0.035*"maritime" + 0.031*"ship" + 0.019*"sea" + 0.018*"port" + 0.017*"pollution" + 0.016*"vessel" + 0.015*"marine" + 0.012*"party" + 0.012*"protocol"
    2018-05-28 18:55:23,735 : INFO : topic #1 (0.031): 0.019*"person" + 0.013*"contract" + 0.011*"citizen" + 0.011*"drug" + 0.010*"condition" + 0.008*"entry" + 0.008*"residence" + 0.008*"network" + 0.008*"request" + 0.007*"right"
    2018-05-28 18:55:23,741 : INFO : topic diff=0.576249, rho=0.382192
    2018-05-28 18:55:37,283 : INFO : -7.182 per-word bound, 145.2 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 18:55:37,284 : INFO : PROGRESS: pass 4, at document #3692/3692
    2018-05-28 18:55:45,100 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 18:55:45,188 : INFO : topic #2 (0.031): 0.028*"animal" + 0.027*"product" + 0.019*"directive" + 0.013*"control" + 0.012*"material" + 0.011*"substance" + 0.011*"waste" + 0.009*"article" + 0.008*"import" + 0.008*"eec"
    2018-05-28 18:55:45,191 : INFO : topic #30 (0.031): 0.019*"security" + 0.011*"external" + 0.010*"strategy" + 0.009*"border" + 0.008*"migration" + 0.008*"relation" + 0.008*"management" + 0.007*"defence" + 0.007*"political" + 0.007*"crisis"
    2018-05-28 18:55:45,193 : INFO : topic #22 (0.031): 0.031*"fund" + 0.023*"financial" + 0.016*"education" + 0.016*"training" + 0.015*"project" + 0.011*"eur" + 0.011*"assistance" + 0.011*"million" + 0.011*"period" + 0.011*"instrument"
    2018-05-28 18:55:45,194 : INFO : topic #19 (0.031): 0.066*"energy" + 0.017*"emission" + 0.016*"environmental" + 0.014*"gas" + 0.010*"electricity" + 0.010*"directive" + 0.010*"renewable" + 0.009*"source" + 0.009*"climate" + 0.009*"efficiency"
    2018-05-28 18:55:45,196 : INFO : topic #29 (0.031): 0.020*"woman" + 0.018*"human" + 0.018*"child" + 0.017*"right" + 0.014*"equal" + 0.012*"health" + 0.012*"research" + 0.011*"equality" + 0.011*"discrimination" + 0.011*"man"
    2018-05-28 18:55:45,197 : INFO : topic diff=0.544776, rho=0.382192
    2018-05-28 18:55:45,199 : INFO : PROGRESS: pass 5, at document #2000/3692
    2018-05-28 18:55:54,388 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:55:54,467 : INFO : topic #11 (0.031): 0.028*"security" + 0.019*"custom" + 0.018*"control" + 0.010*"internal" + 0.010*"export" + 0.009*"document" + 0.009*"electronic" + 0.008*"common" + 0.008*"management" + 0.007*"audit"
    2018-05-28 18:55:54,470 : INFO : topic #28 (0.031): 0.026*"right" + 0.019*"datum" + 0.016*"justice" + 0.013*"judicial" + 0.013*"protection" + 0.011*"criminal" + 0.010*"freedom" + 0.009*"crime" + 0.009*"fundamental" + 0.008*"matter"
    2018-05-28 18:55:54,472 : INFO : topic #12 (0.031): 0.031*"fishing" + 0.028*"fishery" + 0.018*"committee" + 0.013*"vessel" + 0.011*"ecb" + 0.010*"fish" + 0.010*"bank" + 0.009*"group" + 0.009*"central" + 0.009*"board"
    2018-05-28 18:55:54,475 : INFO : topic #15 (0.031): 0.060*"directive" + 0.039*"safety" + 0.030*"health" + 0.021*"risk" + 0.015*"worker" + 0.013*"eec" + 0.012*"protection" + 0.010*"exposure" + 0.010*"requirement" + 0.010*"nuclear"
    2018-05-28 18:55:54,478 : INFO : topic #4 (0.031): 0.026*"right" + 0.016*"agreement" + 0.015*"property" + 0.015*"protection" + 0.015*"online" + 0.014*"europol" + 0.013*"intellectual" + 0.013*"patent" + 0.011*"copyright" + 0.011*"enforcement"
    2018-05-28 18:55:54,482 : INFO : topic diff=0.467382, rho=0.357006
    2018-05-28 18:56:07,223 : INFO : -7.157 per-word bound, 142.7 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 18:56:07,224 : INFO : PROGRESS: pass 5, at document #3692/3692
    2018-05-28 18:56:15,141 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 18:56:15,234 : INFO : topic #9 (0.031): 0.020*"financial" + 0.020*"crime" + 0.012*"terrorist" + 0.012*"combat" + 0.011*"prevention" + 0.010*"fraud" + 0.010*"terrorism" + 0.009*"money" + 0.009*"fight" + 0.008*"organise"
    2018-05-28 18:56:15,236 : INFO : topic #19 (0.031): 0.067*"energy" + 0.018*"emission" + 0.017*"environmental" + 0.014*"gas" + 0.011*"electricity" + 0.010*"directive" + 0.010*"renewable" + 0.010*"source" + 0.009*"climate" + 0.009*"efficiency"
    2018-05-28 18:56:15,238 : INFO : topic #28 (0.031): 0.026*"right" + 0.018*"datum" + 0.017*"justice" + 0.013*"judicial" + 0.013*"protection" + 0.011*"criminal" + 0.011*"freedom" + 0.009*"fundamental" + 0.009*"crime" + 0.008*"police"
    2018-05-28 18:56:15,240 : INFO : topic #5 (0.031): 0.036*"regional" + 0.031*"region" + 0.030*"transport" + 0.014*"partnership" + 0.013*"network" + 0.012*"infrastructure" + 0.012*"integration" + 0.011*"partner" + 0.010*"road" + 0.010*"sea"
    2018-05-28 18:56:15,242 : INFO : topic #23 (0.031): 0.044*"convention" + 0.036*"maritime" + 0.032*"ship" + 0.020*"sea" + 0.019*"port" + 0.017*"pollution" + 0.015*"marine" + 0.015*"vessel" + 0.014*"party" + 0.014*"protocol"
    2018-05-28 18:56:15,244 : INFO : topic diff=0.434388, rho=0.357006
    2018-05-28 18:56:15,246 : INFO : PROGRESS: pass 6, at document #2000/3692
    2018-05-28 18:56:25,687 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 18:56:25,775 : INFO : topic #4 (0.031): 0.029*"right" + 0.017*"property" + 0.017*"protection" + 0.016*"agreement" + 0.015*"online" + 0.014*"intellectual" + 0.014*"europol" + 0.013*"patent" + 0.012*"copyright" + 0.011*"enforcement"
    2018-05-28 18:56:25,779 : INFO : topic #11 (0.031): 0.029*"security" + 0.019*"control" + 0.018*"custom" + 0.010*"internal" + 0.010*"export" + 0.010*"electronic" + 0.010*"document" + 0.008*"common" + 0.008*"management" + 0.007*"audit"
    2018-05-28 18:56:25,785 : INFO : topic #7 (0.031): 0.038*"custom" + 0.036*"tax" + 0.028*"request" + 0.020*"assistance" + 0.018*"taxation" + 0.015*"agreement" + 0.013*"legislation" + 0.011*"administrative" + 0.010*"party" + 0.010*"mutual"
    2018-05-28 18:56:25,786 : INFO : topic #6 (0.031): 0.045*"product" + 0.035*"food" + 0.020*"consumer" + 0.013*"health" + 0.009*"modify" + 0.008*"agricultural" + 0.008*"agency" + 0.008*"genetically" + 0.008*"foodstuff" + 0.008*"feed"
    2018-05-28 18:56:25,788 : INFO : topic #8 (0.031): 0.033*"acquis" + 0.027*"legislation" + 0.025*"progress" + 0.017*"accession" + 0.014*"field" + 0.012*"law" + 0.011*"note" + 0.011*"november" + 0.011*"effort" + 0.011*"sector"
    2018-05-28 18:56:25,790 : INFO : topic diff=0.374680, rho=0.336222
    2018-05-28 18:56:38,800 : INFO : -7.138 per-word bound, 140.9 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    ......
    2018-05-28 19:15:44,378 : INFO : topic diff=0.019104, rho=0.140240
    2018-05-28 19:15:44,381 : INFO : PROGRESS: pass 49, at document #2000/3692
    2018-05-28 19:15:53,096 : INFO : merging changes from 2000 documents into a model of 3692 documents
    2018-05-28 19:15:53,189 : INFO : topic #18 (0.031): 0.056*"directive" + 0.042*"agency" + 0.021*"railway" + 0.021*"safety" + 0.018*"infrastructure" + 0.016*"network" + 0.016*"technical" + 0.016*"equipment" + 0.016*"requirement" + 0.015*"rail"
    2018-05-28 19:15:53,192 : INFO : topic #16 (0.031): 0.018*"sector" + 0.012*"propose" + 0.012*"europe" + 0.011*"would" + 0.008*"increase" + 0.008*"could" + 0.008*"industry" + 0.008*"future" + 0.007*"change" + 0.007*"therefore"
    2018-05-28 19:15:53,195 : INFO : topic #1 (0.031): 0.027*"person" + 0.014*"right" + 0.013*"citizen" + 0.013*"condition" + 0.013*"residence" + 0.012*"application" + 0.012*"noneu" + 0.012*"visa" + 0.012*"document" + 0.011*"entry"
    2018-05-28 19:15:53,198 : INFO : topic #26 (0.031): 0.033*"treaty" + 0.022*"article" + 0.011*"euro" + 0.010*"procedure" + 0.010*"central" + 0.009*"rate" + 0.009*"bank" + 0.009*"government" + 0.009*"deficit" + 0.009*"budgetary"
    2018-05-28 19:15:53,203 : INFO : topic #29 (0.031): 0.028*"right" + 0.025*"woman" + 0.024*"child" + 0.021*"human" + 0.018*"equal" + 0.014*"equality" + 0.014*"discrimination" + 0.013*"man" + 0.011*"gender" + 0.011*"combat"
    2018-05-28 19:15:53,207 : INFO : topic diff=0.018864, rho=0.138881
    2018-05-28 19:16:05,395 : INFO : -7.047 per-word bound, 132.3 perplexity estimate based on a held-out corpus of 1692 documents with 605190 words
    2018-05-28 19:16:05,396 : INFO : PROGRESS: pass 49, at document #3692/3692
    2018-05-28 19:16:11,532 : INFO : merging changes from 1692 documents into a model of 3692 documents
    2018-05-28 19:16:11,610 : INFO : topic #20 (0.031): 0.033*"trade" + 0.032*"consumer" + 0.014*"standard" + 0.010*"business" + 0.010*"legislation" + 0.008*"practice" + 0.007*"access" + 0.007*"protection" + 0.007*"barrier" + 0.006*"internal"
    2018-05-28 19:16:11,612 : INFO : topic #26 (0.031): 0.034*"treaty" + 0.022*"article" + 0.011*"euro" + 0.010*"procedure" + 0.009*"central" + 0.009*"rate" + 0.009*"government" + 0.009*"bank" + 0.009*"deficit" + 0.009*"budgetary"
    2018-05-28 19:16:11,615 : INFO : topic #10 (0.031): 0.035*"financial" + 0.026*"company" + 0.024*"euro" + 0.021*"directive" + 0.018*"bank" + 0.017*"payment" + 0.016*"capital" + 0.012*"credit" + 0.012*"account" + 0.010*"institution"
    2018-05-28 19:16:11,617 : INFO : topic #22 (0.031): 0.043*"fund" + 0.029*"financial" + 0.020*"project" + 0.017*"million" + 0.016*"eur" + 0.016*"assistance" + 0.015*"budget" + 0.015*"period" + 0.014*"instrument" + 0.011*"finance"
    2018-05-28 19:16:11,619 : INFO : topic #11 (0.031): 0.048*"security" + 0.023*"control" + 0.021*"border" + 0.017*"defence" + 0.012*"internal" + 0.012*"common" + 0.012*"management" + 0.011*"export" + 0.011*"military" + 0.011*"external"
    2018-05-28 19:16:11,621 : INFO : topic diff=0.018733, rho=0.138881



```python
# Print topics
# I also know that none of my topic is more than 5 words i will set the num_words argument to 5
for i,topic in enumerate(ldamodel.print_topics(num_topics=32, num_words=5)):
    words = topic[1].split("+")
    print (words,"\n")
```


    ['0.045*"employment" ', ' 0.029*"labour" ', ' 0.019*"worker" ', ' 0.017*"job" ', ' 0.013*"people"'] 
    
    ['0.027*"person" ', ' 0.014*"right" ', ' 0.013*"citizen" ', ' 0.013*"application" ', ' 0.013*"residence"'] 
    
    ['0.043*"directive" ', ' 0.027*"animal" ', ' 0.015*"eec" ', ' 0.015*"product" ', ' 0.012*"substance"'] 
    
    ['0.071*"transport" ', ' 0.034*"air" ', ' 0.028*"vehicle" ', ' 0.027*"passenger" ', ' 0.022*"road"'] 
    
    ['0.066*"right" ', ' 0.034*"property" ', ' 0.026*"protection" ', ' 0.026*"intellectual" ', ' 0.017*"copyright"'] 
    
    ['0.051*"region" ', ' 0.044*"regional" ', ' 0.025*"transport" ', ' 0.019*"network" ', ' 0.016*"cohesion"'] 
    
    ['0.065*"product" ', ' 0.041*"food" ', ' 0.016*"agricultural" ', ' 0.011*"consumer" ', ' 0.010*"label"'] 
    
    ['0.061*"tax" ', ' 0.059*"custom" ', ' 0.029*"duty" ', ' 0.025*"taxation" ', ' 0.024*"request"'] 
    
    ['0.034*"acquis" ', ' 0.027*"progress" ', ' 0.026*"legislation" ', ' 0.020*"accession" ', ' 0.013*"law"'] 
    
    ['0.026*"crime" ', ' 0.021*"criminal" ', ' 0.016*"combat" ', ' 0.013*"offence" ', ' 0.012*"europol"'] 
    
    ['0.035*"financial" ', ' 0.026*"company" ', ' 0.024*"euro" ', ' 0.021*"directive" ', ' 0.018*"bank"'] 
    
    ['0.048*"security" ', ' 0.023*"control" ', ' 0.021*"border" ', ' 0.017*"defence" ', ' 0.012*"internal"'] 
    
    ['0.035*"fishery" ', ' 0.033*"committee" ', ' 0.029*"fishing" ', ' 0.016*"vessel" ', ' 0.016*"statistic"'] 
    
    ['0.038*"directive" ', ' 0.027*"law" ', ' 0.020*"court" ', ' 0.013*"contract" ', ' 0.013*"legal"'] 
    
    ['0.060*"education" ', ' 0.045*"training" ', ' 0.025*"people" ', ' 0.021*"young" ', ' 0.020*"youth"'] 
    
    ['0.063*"health" ', ' 0.047*"safety" ', ' 0.033*"nuclear" ', ' 0.030*"risk" ', ' 0.024*"directive"'] 
    
    ['0.018*"sector" ', ' 0.012*"propose" ', ' 0.012*"europe" ', ' 0.011*"would" ', ' 0.009*"increase"'] 
    
    ['0.027*"research" ', ' 0.024*"strategy" ', ' 0.015*"innovation" ', ' 0.014*"plan" ', ' 0.013*"technology"'] 
    
    ['0.056*"directive" ', ' 0.042*"agency" ', ' 0.021*"safety" ', ' 0.021*"railway" ', ' 0.019*"infrastructure"'] 
    
    ['0.082*"energy" ', ' 0.025*"emission" ', ' 0.022*"environmental" ', ' 0.017*"gas" ', ' 0.016*"water"'] 
    
    ['0.033*"trade" ', ' 0.032*"consumer" ', ' 0.014*"standard" ', ' 0.010*"business" ', ' 0.010*"legislation"'] 
    
    ['0.126*"agreement" ', ' 0.022*"party" ', ' 0.021*"research" ', ' 0.011*"joint" ', ' 0.011*"part"'] 
    
    ['0.043*"fund" ', ' 0.029*"financial" ', ' 0.020*"project" ', ' 0.017*"million" ', ' 0.016*"eur"'] 
    
    ['0.063*"convention" ', ' 0.038*"maritime" ', ' 0.033*"ship" ', ' 0.022*"sea" ', ' 0.019*"port"'] 
    
    ['0.027*"paper" ', ' 0.023*"medium" ', ' 0.022*"audiovisual" ', ' 0.022*"green" ', ' 0.020*"general"'] 
    
    ['0.080*"aid" ', ' 0.028*"competition" ', ' 0.019*"article" ', ' 0.013*"grant" ', ' 0.012*"guideline"'] 
    
    ['0.034*"treaty" ', ' 0.022*"article" ', ' 0.011*"euro" ', ' 0.010*"procedure" ', ' 0.009*"central"'] 
    
    ['0.062*"datum" ', ' 0.026*"access" ', ' 0.021*"electronic" ', ' 0.017*"network" ', ' 0.015*"internet"'] 
    
    ['0.025*"right" ', ' 0.023*"justice" ', ' 0.016*"asylum" ', ' 0.015*"freedom" ', ' 0.015*"judicial"'] 
    
    ['0.028*"right" ', ' 0.025*"child" ', ' 0.025*"woman" ', ' 0.021*"human" ', ' 0.019*"equal"'] 
    
    ['0.014*"strategy" ', ' 0.012*"dialogue" ', ' 0.011*"security" ', ' 0.011*"partnership" ', ' 0.010*"political"'] 
    
    ['0.033*"progress" ', ' 0.027*"accession" ', ' 0.024*"candidate" ', ' 0.023*"acquis" ', ' 0.016*"turkey"'] 
    



```python
# Now lets save the model, dictonary, and corpus to use it for further use
# Saving Model
ldamodel.save('/home/jay/ANN_Models/LDA/LDA_Model')

# Saving Corpus
import pickle
with open('/home/jay/ANN_Models/LDA/LAD_Corpus.pickle', 'wb') as p:
    pickle.dump(cleaned,p)
    
# Saving Dictonary
dictionary.save('/home/jay/ANN_Models/LDA/LDA_Dictonary')
```

    2018-05-28 19:17:04,097 : INFO : saving LdaState object under /home/jay/ANN_Models/LDA/LDA_Model.state, separately None
    2018-05-28 19:17:04,111 : INFO : saved /home/jay/ANN_Models/LDA/LDA_Model.state
    2018-05-28 19:17:04,120 : INFO : saving LdaModel object under /home/jay/ANN_Models/LDA/LDA_Model, separately ['expElogbeta', 'sstats']
    2018-05-28 19:17:04,122 : INFO : storing np array 'expElogbeta' to /home/jay/ANN_Models/LDA/LDA_Model.expElogbeta.npy
    2018-05-28 19:17:04,126 : INFO : not storing attribute id2word
    2018-05-28 19:17:04,127 : INFO : not storing attribute state
    2018-05-28 19:17:04,128 : INFO : not storing attribute dispatcher
    2018-05-28 19:17:04,132 : INFO : saved /home/jay/ANN_Models/LDA/LDA_Model
    2018-05-28 19:17:04,848 : INFO : saving Dictionary object under /home/jay/ANN_Models/LDA/LDA_Dictonary, separately None
    2018-05-28 19:17:04,856 : INFO : saved /home/jay/ANN_Models/LDA/LDA_Dictonary


### Notebooks
You can view this notebook [here](http://nbviewer.jupyter.org/github/jdvala/blog.io/blob/master/notebooks/Latent%20Dirichlet%20Allocation%20%28Topic%20Modeling%29%20-%20Baseline.ipynb)

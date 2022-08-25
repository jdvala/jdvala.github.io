+++
title = "English Corpus Preprocessing"
author = "Jay Vala"
description =  "Steps involved in the processing and cleaning the English corpus of EU summaries"
date = "2018-05-04"
tags = ["python", "text", "text-preprocessing", "nlp", "text-cleaning", "preprocessing"]
+++

# Steps in preprocessing English text

Before feeding text to any deep learning model it is advisable to preprocess the text to make it more suitable for the model. Preprocessing in general means cleaning the data to make it more convenient for any or all deep learning models to learn properly and efficiently.

Preprocessing is subjective, that means it depends on the purpose of preprocessing, depends on data, depends on language and also depends on the type of technique one would be adapting, but there are a lot of steps in preprocessing that does not depend on any of the above mentioned things, but its more of something that has to be done no matter what.

In my thesis I will be using set of preprocessing rules that I think will make learning more effective for my model that I am planning on using.

### Rules:

* All characters to lower case or upper case.
* Sentence tokenize the text.
* Remove numbers.
* Remove currency symbols.
* Remove all the punctuation.
* Remove English stop words.
* Remove other unicode characters(if any).
* Perform Lemmatization or Stemming.

Let's go ahead and see how to do these things in python.

**NOTE: I am using anaconda as my virtual environment, I will soon add how to setup anaconda and virtual environment in any linux system(Debain based).**

First step is to import necessary modules to perform the operations, I have used [Regular expressions](https://docs.python.org/3/library/re.html), [String](https://docs.python.org/3/library/string.html) Class of Python, [NLTK Library](https://www.nltk.org/), and [Spacy library](https://spacy.io/usage/).

So, the first step is to sentence tokenize the text we receive, for that I am using sentence tokenizer from NLTK library.

Let's take an example text file for the corpus.

```bash
SUMMARY
During the period 2004-06, the Commission is implementing a Community programme covering plant, microbial and animal genetic resources * which are or could be of use in agriculture. The amount allocated to the programme is 10 million.

The Commission selects the actions to be part-financed under the programme on the basis of calls for proposals and following evaluation by independent experts. Proposals may be submitted by a public sector body or any natural or legal person who is a national of a Member State and established in the Community, in an EFTA/EEA country, or in an associated country in accordance with the conditions stipulated in a bilateral agreement.

The actions, which may last for a maximum of four years, may be of three types:
targeted actions, part-financed up to a maximum of 50% of their total cost and including:- transnational actions promoting the ex situ and in situ conservation *, characterisation, collection and utilisation of genetic resources in agriculture;- the establishment of a European decentralised, permanent and widely accessible web-based inventory of genetic resources currently conserved in situ including in situ/on-farm genetic resources conservation activities;- the establishment of a European decentralised, permanent and widely accessible web-based inventory of the ex situ collections (gene banks) and in situ resources;- the promotion of regular exchanges of technical and scientific information among competent organisations in the Member States;
concerted actions, part-financed up to a maximum of 80% of their total cost, transnational in character and promoting the exchange of information on thematic issues for the purpose of improving the coordination of actions and programmes in the sphere concerned;

accompanying actions, part-financed up to a maximum of 80% of their total cost and comprising information, dissemination and advisory actions, training courses and the preparation of technical reports.

Once an action has been approved, the Commission will conclude a grant agreement with the participants setting out detailed criteria for the reporting, dissemination, protection and exploitation of the results of the action.

The Commission is assisted by a Committee on the conservation, characterisation, collection and utilisation of genetic resources in agriculture. The Commission may also call on the assistance of scientific and technical experts for the implementation of the programme.

At the end of the programme, the Commission will appoint a group of independent experts to report on the implementation of the Regulation, to assess the results and to make appropriate recommendations. The group's report will be submitted to the European Parliament, the Council and the European Economic and Social Committee.
```

```python

# Let's read the file 
from nltk.tokenize import sent_tokenize

with open('Conservation, characterisation, collection and utilisation of genetic resources in agriculture.txt', 'r') as f:
    contents = f.readlines()

#NOTE: The function readlines() return a list of lines in the same format as the file is.


def sentences(list_):
    """Returns sentence tokenized text list"""
    text = ''.join(list_)

    # Sentence tokenize with help of sent_tokenize from nltk
    sentence = sent_tokenize(text)

    return sentence
```

Now lets pass the ```contents``` to the function ```sentences(list_)``` and see what is the output

```python
after_sent_token = sentences(contents)
```

The output of the above code is

```bash
['SUMMARY\nDuring the period 2004-06, the Commission is implementing a Community programme covering plant, microbial and animal genetic resources * which are or could be of use in agriculture.',
 'The amount allocated to the programme is 10\xa0million.',
 'The Commission selects the actions to be part-financed under the programme on the basis of calls for proposals and following evaluation by independent experts.',
 'Proposals may be submitted by a public sector body or any natural or legal person who is a national of a Member State and established in the Community, in an EFTA/EEA country, or in an associated country in accordance with the conditions stipulated in a bilateral agreement.',
 'The actions, which may last for a maximum of four years, may be of three types:\ntargeted actions, part-financed up to a maximum of 50% of their total cost and including:- transnational actions promoting the ex situ and in situ conservation *, characterisation, collection and utilisation of genetic resources in agriculture;- the establishment of a European decentralised, permanent and widely accessible web-based inventory of genetic resources currently conserved in situ including in situ/on-farm genetic resources conservation activities;- the establishment of a European decentralised, permanent and widely accessible web-based inventory of the ex situ collections (gene banks) and in situ resources;- the promotion of regular exchanges of technical and scientific information among competent organisations in the Member States;\nconcerted actions, part-financed up to a maximum of 80% of their total cost, transnational in character and promoting the exchange of information on thematic issues for the purpose of improving the coordination of actions and programmes in the sphere concerned;\naccompanying actions, part-financed up to a maximum of 80% of their total cost and comprising information, dissemination and advisory actions, training courses and the preparation of technical reports.',
 'Once an action has been approved, the Commission will conclude a grant agreement with the participants setting out detailed criteria for the reporting, dissemination, protection and exploitation of the results of the action.',
 'The Commission is assisted by a Committee on the conservation, characterisation, collection and utilisation of genetic resources in agriculture.',
 'The Commission may also call on the assistance of scientific and technical experts for the implementation of the programme.',
 'At the end of the programme, the Commission will appoint a group of independent experts to report on the implementation of the Regulation, to assess the results and to make appropriate recommendations.',
 "The group's report will be submitted to the European Parliament, the Council and the European Economic and Social Committee."]
```

Now, that we have sentence tokenize, we will make everything into a lower case. 

```python
# after_sent_token is a list.
# Make a new list.

after_lower = []

for line in after_sent_token:
    after_lower.append(line.lower())
```

When we print the ```after_lower``` list, we get the output to be

```bash
['summary\nduring the period 2004-06, the commission is implementing a community programme covering plant, microbial and animal genetic resources * which are or could be of use in agriculture.',
 'the amount allocated to the programme is 10\xa0million.',
 'the commission selects the actions to be part-financed under the programme on the basis of calls for proposals and following evaluation by independent experts.',
 'proposals may be submitted by a public sector body or any natural or legal person who is a national of a member state and established in the community, in an efta/eea country, or in an associated country in accordance with the conditions stipulated in a bilateral agreement.',
 'the actions, which may last for a maximum of four years, may be of three types:\ntargeted actions, part-financed up to a maximum of 50% of their total cost and including:- transnational actions promoting the ex situ and in situ conservation *, characterisation, collection and utilisation of genetic resources in agriculture;- the establishment of a european decentralised, permanent and widely accessible web-based inventory of genetic resources currently conserved in situ including in situ/on-farm genetic resources conservation activities;- the establishment of a european decentralised, permanent and widely accessible web-based inventory of the ex situ collections (gene banks) and in situ resources;- the promotion of regular exchanges of technical and scientific information among competent organisations in the member states;\nconcerted actions, part-financed up to a maximum of 80% of their total cost, transnational in character and promoting the exchange of information on thematic issues for the purpose of improving the coordination of actions and programmes in the sphere concerned;\naccompanying actions, part-financed up to a maximum of 80% of their total cost and comprising information, dissemination and advisory actions, training courses and the preparation of technical reports.',
 'once an action has been approved, the commission will conclude a grant agreement with the participants setting out detailed criteria for the reporting, dissemination, protection and exploitation of the results of the action.',
 'the commission is assisted by a committee on the conservation, characterisation, collection and utilisation of genetic resources in agriculture.',
 'the commission may also call on the assistance of scientific and technical experts for the implementation of the programme.',
 'at the end of the programme, the commission will appoint a group of independent experts to report on the implementation of the regulation, to assess the results and to make appropriate recommendations.',
 "the group's report will be submitted to the european parliament, the council and the european economic and social committee."]
```

Similarly, we can do everything as we did above. The full code for english preprocessing is below.

```python 
# This preprocessing is intended for english
# coding: utf-8

import os
import sys
import re
import string
from nltk.corpus import stopwords
import spacy
from nltk.tokenize import sent_tokenize


def sentences(list_):
    """Returns sentence tokenized text list"""
    text = ''.join(list_)

    # Sentence tokenize with help of sent_tokenize from nltk  
    sentence = sent_tokenize(text)

    return sentence


def remove(text):
    """Returns text with all the filtering necessary"""
    t = re.sub(r"(\d+\.\d+)","",text)
    #t = re.sub(r"(\d+th?|st?|nd?|rd?)","", t)
    t = re.sub(r"\d{2}.\d{2}.\d{4}","",t)
    t = re.sub(r"\d{2}\/\d{2}\/\d{4}","",t)
    t = re.sub(r"\d{2}(\/|\.)\d{2}(\/|\.)\d{2}","",t)
    t = re.sub(r"($|€|¥|₹|£)","",t)
    t = re.sub(r"(%)","",t)
    t = re.sub(r"\d+","",t)
    t = re.sub(r"\n","",t)
    t = re.sub(r"\xa0", "", t)
    return t

def pun(text):
    """Return punctuations from text"""
    table = str.maketrans("","", string.punctuation)
    t = text.translate(table)
    return t


nlp = spacy.load('en')

def lemmatizer(text):
    """Returns text after lemmatization"""
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


def extras(sentences):
    """Returns text after removing some extra symbols"""
    t = re.sub(r"\"|\—|\'|\’","",sentences)
    word_list = t.split()
    for index, word in enumerate(word_list):
        if len(word) <=1:
            del word_list[index]
    t = ' '.join(word_list)

    return t

# Stop word removal
def stop_word(sentence):
    list_ = []
    stop_words = stopwords.words('english')
    words_list = sentence.split()
    for word in words_list:
        if word not in stop_words:
            list_.append(word)
    return ' '.join(list_)




def main():
    """Main Function"""
    with open('Conservation, characterisation, collection and utilisation of genetic resources in agriculture.txt', 'r') as f:
        contents = f.readlines()
    # Joining the lines to make text block 
    contents = ''.join(contents)
    print("Starting to preprocess file")
    # Sentence tokenize the text
    sent_tokenized = sentences(contents)
    # Removing stop words
    t1 = [lemmatizer(sent) for sent in sent_tokenized]
    # lemmatization 
    t2 = [stop_word(sent) for sent in t1]
    # Removing all the unnecessary things from the text 
    t3 = [remove(line) for line in t2]
    # Removing punctuations
    t4 =[pun(line.lower()) for line in t3]

    t5 = [extras(sent) for sent in t4]
    print("Preprocessing done for file)

    print(t5)

if __name__ == "__main__":
    main()
```

The output of the file after preprocessing will be

```bash
['summary period commission implement community programme covering plant microbial animal genetic resource could use agriculture', 'amount allocate programme million', 'commission select action part finance programme basis call proposal follow evaluation independent expert', 'proposal may submit public sector body natural legal person national member state establish community efta eea country associated country accordance condition stipulate bilateral agreement', 'action may last maximum four year may three type target action part finance maximum pron total cost including transnational action promote ex situ situ conservation characterisation collection utilisation genetic resource agriculture establishment european decentralise permanent widely accessible web base inventory genetic resource currently conserve situ include situ farm genetic resource conservation activities establishment european decentralise permanent widely accessible web base inventory ex situ collection gene bank situ resources promotion regular exchange technical scientific information among competent organisation member states concerted action part finance maximum pron total cost transnational character promote exchange information thematic issue purpose improve coordination action programme sphere concern accompanying action part finance maximum pron total cost comprise information dissemination advisory action train course preparation technical report', 'action approve commission conclude grant agreement participant set detailed criterion reporting dissemination protection exploitation result action', 'commission assist committee conservation characterisation collection utilisation genetic resource agriculture', 'commission may also call assistance scientific technical expert implementation programme', 'end programme commission appoint group independent expert report implementation regulation assess result make appropriate recommendation', 'group report submit european parliament council european economic social committee']
```

This is probably all the preprocessing necessary for creating word vectors, that is what my next task is.

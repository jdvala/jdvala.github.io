+++
title = "Curating the Dataset again!"
description =  "Creating the dataset again after quantitative analysis"
date = "2018-05-23"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "frustration"]
+++

After the initial results it was clear that I have a lot less data then what the neural networks need to perform better. So I have decided that I will use everything in the files as my training data instead of removing everything except SUMMARY of the document. Also I will remove all the line with all the captial letters in order to remove the tags. 


```python
import os
import sys
import re
import string
from string import punctuation
from nltk.corpus import stopwords
import spacy
```


```python
# NLP model from spacy
nlp = spacy.load('en')
```


```python
# First lets create list of topics
label = []
for root, dirs, files in os.walk('/home/jay/GITHUB/Data_Thesis_1/Ready/'):
    label.append(dirs)
```


```python
labels = dict(list(enumerate(label[0])))
```


```python
def preprocess(line):
    """Preprocess given line.
    :params: line as string
    :return: manipulated line as string
    """
    
    # lower case
    lower = line.lower()
    
    t = re.sub(r"(\d+\.\d+)","",lower)
    t = re.sub(r"\d{2}.\d{2}.\d{4}","",t)
    t = re.sub(r"\d{2}\/\d{2}\/\d{4}","",t)
    t = re.sub(r"\d{2}(\/|\.)\d{2}(\/|\.)\d{2}","",t)
    t = re.sub(r"($|€|¥|₹|£)","",t)
    t = re.sub(r"(%)","",t)
    t = re.sub(r"\d+","",t)
    t = re.sub(r"\n","",t)
    t = re.sub(r"\xa0", "", t)
    
    # Removing punctuations
    
    table = str.maketrans("","", string.punctuation)
    text = t.translate(table)
    
    
    # Removing other extras
    t = re.sub(r"\"|\—|\'|\’|–","",text)
    
    # Removing stop words
    
    stop_words = stopwords.words('english')
    t_ = [word for word in t.split() if word not in stop_words]
    t = ' '.join(t_)
    
    
    # Lemmetizer
    sent = []
    doc = nlp(t)
    for word in doc:
        sent.append(word.lemma_)
        
    text = " ".join(sent)
    
    return text
```


```python
caps_with_question_mark = re.compile(r"^([A-Z ':]+\?$)", re.M)
caps_without_question_mark = re.compile(r"^([A-Z ':]+$)", re.M)
```


```python
# Iterate over files and get the content.
sents = []
label = []
for root, dirs, files in os.walk('/home/jay/Thesis_1/Data/Data_EN/'):
    for file in files:
        if file.endswith('.txt'):
            topic = root.split(os.path.sep)[-2] 
            if file != 'log.txt':
                with open(os.path.join(root, file)) as f:
                    content = f.read()
                    
                    
                    for key,value in labels.items():
                        if str(value) == str(topic):      # If the dir is equal to any value in the labels dict then
                            label_ = key
                    
                    upp = caps_with_question_mark.sub('',content)
                
                    upp_ = caps_without_question_mark.sub('',upp)
                
                    # remove first two lines   
                    cont = upp_.split('\n')
                    cont = cont[2:]
                    
                    # preprocess every line
                    lines = []
                    for line in cont:
                        lines.append(preprocess(line))
                    line_ = []    
                    for l in lines:
                        line_.append((re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '',l)))
                    

                    
                    _line = []
                    for l in line_:
                        k = l.replace('oj', '')
                        n = k.replace('ii', '')
                        _line.append(n)
                        
                    final = [line for line in _line if line]
                    
                    
                    for line in final:
                        sents.append(line)
                        label.append(label_)
                    
                  
                    
                    
```


```python
len(sents), len(label)
```




    (164679, 164679)




```python
import pickle
with open('/home/jay/pickled/sent.pkl', 'wb') as f:
    pickle.dump(sents,f)
    
with open('/home/jay/pickled/label.pkl', 'wb') as g:
    pickle.dump(label,g)
```

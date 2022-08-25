+++
title = "Data Discrepancies"
description = "Even after cleaning the data there were a lot of discrepancies that were to be delt with"
date = "2018-05-16"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "frustration", "data-discrepancies"]
+++

# Analysizing Discrepancies in Dataset


a lot of discrepancies in the dataset created previously, I can to realize that when I was building the model. I always analyze the dataset before building the model because 90% of the time people don't realize that their model is performing bad because their data was bad. I am not gonna let that happen.

So what all discrepancies I found?
* The largest sequence was of 242 words. This is really bad becasue then we have to pad each sequence to that lenght and that would be really bad when training as it will require more resource, which is scarce.
* There are squences or sentences with just one word, well again this is not right because I will have to pad them with zeros. Hence, I need to remove them also.

So what I am gonna do is I am gonna analyze the dataframe and come up with some strategy.


```python
import pandas as pd
import numpy as np
import keras
```

    /home/jay/.local/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.



```python
dataset = pd.read_csv('/home/jay/Data_Set_Creation/Data_to_Use.csv')
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentence</th>
      <th>Label</th>
      <th>Numerical_Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>summary the directive seek ensure eu country f...</td>
      <td>transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mlc set minimum global standard ensure right s...</td>
      <td>transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pron also seek limit social dump secure fair c...</td>
      <td>transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>the main point new directive follow</td>
      <td>transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>monitor compliance eu country introduce effect...</td>
      <td>transport</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sentence column to list
sent = dataset.Sentence.tolist()
```

We can not perform any numpy operation on string data, so our best bet is to convert strings to its encoded version



```python
# Using Keras tokenizer
tokenizer = keras.preprocessing.text.Tokenizer()

# Tokenizing the sentences (This process may take some time depending on your corpus size)
tokenizer.fit_on_texts(sent)
```


```python
sent_encoded = tokenizer.texts_to_sequences(sent)
```

Now that all the sentences have been encoded, lets lets what I was telling in the starting. Check the maximum length of sentence in the sent_encoder


```python
len(max(sent_encoded, key=len))
```




    242



As I said the longest string is of 242 words which is not convientent. Let's check for all the squences that are one word, we will just sort the list


```python
sorted(sent,key=len)
```




    ['nb',
     'pdf',
     'sbs',
     'act',
     'sum',
     'sum',
     'air',
     'van',
     'swd',
     'yes',
     'yes',
     'new',
     'iii',
     'mhz',
     'case',
     'next',
     'csdp',
     'coin',
     'coin',
     'loan',
     'summary',
     'summary',
     'summary',
     'summary',
     'summary',
     'erasmus',
     'summary',
     'summary',
     'summary',
     'summary',
     'context',
     'summary',
     'summary',
     'summary'
     'summary decision',
     'exception permit',
     'summary decision',
     'instrument adopt',
     'idea eur million',
     'permanent impact',
     'cyprus coal mine',
     ...]



Now we will see the percentile of length of different sequences and analyse what can be done of this


```python
len_list = []
for a in sent:
    len_list.append(len(a.split()))
```


```python
# The length of longest sequence(in words)
max(len_list)
```




    242




```python
array = np.array(len_list)
```

Percentile is a good way to see what percent of things fall at or below this number. In our case if we do percentile 99 of a list of lengths then it will show me the lenght(average) of 99 percent of the sequence in the list which is 66. 


```python
pre = np.percentile(array, 99)
```


```python
pre
```




    66.0




```python
precentile = []
number = []
for i in range(0,101):
    number.append(i)
    precentile.append(np.percentile(array, i))
```


```python
percentile = list(map(int, precentile))
```


```python
percentile
```




    [1,
     2,
     3,
     4,
     5,
     5,
     5,
     6,
     6,
     6,
     7,
     7,
     7,
     7,
     7,
     8,
     8,
     8,
     8,
     8,
     9,
     9,
     9,
     9,
     9,
     9,
     10,
     10,
     10,
     10,
     10,
     10,
     10,
     11,
     11,
     11,
     11,
     11,
     11,
     12,
     12,
     12,
     12,
     12,
     12,
     13,
     13,
     13,
     13,
     13,
     13,
     14,
     14,
     14,
     14,
     14,
     14,
     15,
     15,
     15,
     15,
     15,
     16,
     16,
     16,
     16,
     16,
     17,
     17,
     17,
     17,
     18,
     18,
     18,
     19,
     19,
     19,
     20,
     20,
     20,
     21,
     21,
     21,es)
     22,
     22,
     23,
     24,
     24,
     25,
     26,
     27,
     28,
     30,
     31,
     33,
     36,
     40,
     45,
     52,
     66,
     242]



Lets plot the distribution using matplot lib


```python
import matplotlib.pyplot as plt
%matplotlib inline 
plt.figure(figsize=(15,10))
plt.bar(number, percentile)
plt.title("Length Distribution")
plt.xlabel("No words in a line")
plt.ylabel("Frequency")
#for i, v in enumerate(precentile):
    #plt.text(i, v + 1, str(v), color='blue', fontweight='bold')
plt.show()
```


<img src="{{ site.baseurl }}/images/output_23_0.png">


This is not good, We have to have a stretegy to counter this. Before I end this post. I want to see a few things that would help me get more insight of the data. 
Firstly I want to see what is the ratio of each class


```python
# List of Numerical_Labels
labels = dataset.Numerical_Label.tolist()
```


```python
from collections import Counter
```


```python
count = Counter()

for label in labels:
    count[label] += 1
```


```python
count
```




    Counter({0: 7513,
             1: 1313,
             2: 4596,
             3: 11685,
             4: 13170,
             5: 4512,
             6: 5447,
             7: 1372,
             8: 3207,
             9: 2453,
             10: 3148,
             11: 3329,
             12: 1931,
             13: 808,
             14: 2378,
             15: 8426,
             16: 3765,
             17: 1478,
             18: 1193,
             19: 2469,
             20: 3273,
             21: 3942,
             22: 1176,
             23: 963,
             24: 9876,
             25: 3776,
             26: 4142,
             27: 1873,
             28: 2058,
             29: 922,
             30: 1710,
             31: 3632})



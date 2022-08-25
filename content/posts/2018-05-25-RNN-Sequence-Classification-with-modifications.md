+++
title = "LSTM with 8 Classes"
description =  "Training a Long Short Term Memory Network on subset of classes"
date = "2018-05-25"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "AI", "RNN"]
+++

# New LSTM with 8 classes

In this script I will be using only 8 out of 32 classes that were originaly present in the dataset, this is necessary because the data in other classes is much less compared to these 8 classes, this makes it difficult for the neural network to learn anything off of those classes.


```python
import pickle
from gensim.models import Word2Vec
import keras
import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
```

    /home/jay/.local/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


I have already stored the senteces and their corresponding labels as stated in this [post](https://jdvala.github.io/blog.io/thesis/2018/05/23/Creating-Data-Set-Again-!.html). Hence I am going to use those sentences and labels.


```python
# Loading data
with open('/home/jay/pickled/sent.pkl', 'rb') as f:
    sents = pickle.load(f)
    
with open('/home/jay/pickled/label.pkl', 'rb') as g:
    label = pickle.load(g)
```

As the model was not learning, I have decided to reduce the number of classes to 8, so I will pick up top 8 classes with maximum number of samples. 


```python
# lets count number of sample in each class
from collections import Counter

counter = Counter()

for l in label:
    counter[l] +=1
```

As I have counted number of samples in each class, I will pick out top 8 classes from them and use only the 8 


```python
import heapq
classes = heapq.nlargest(8,zip(counter.values(), counter.keys()))
classes
```




    [(19018, 4),
     (15679, 24),
     (12455, 15),
     (10219, 3),
     (10011, 0),
     (7008, 26),
     (6606, 5),
     (6392, 11)]



These are the classes I am going to use for training the neural network.


```python
sample_8 =[]
label_8 = []

for label, sent in zip(label, sents):
    if (label == 4) or (label ==24) or (label == 15) or (label==3) or (label==0) or (label==26) or (label==5) or (label==11):
        sample_8.append(sent)
        label_8.append(label)
```


```python
len(sample_8)
```




    87388



As I have stated in my [Data Descrepancies post](https://jdvala.github.io/blog.io/thesis/2018/05/16/Data-Discrepancies.html) that some sentences are very long. This is a problem when traning recurrent neural network with long sequences.
For example think of a situation where you have a sentence of 10 words and another sentence which is 200 word sentence which also happens to be the maximum length of sentence in dataset. So when you are padding all the sentences the sentence with 10 words will have 190 padded zeros, so you see how this is very inefficient and that there has to be some solution to things like this.
To tackle this problem what I think is a easy and effective solution is to break the 200 word sentence into number of smaller sentneces. This is however not ideal as we are loosing semantics and realations between words but this seams to be the only viable option right now. There are other options which you can learn about it [here](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/).

So for me what I will do is I will use sliding window or rolling window algorithm. To preserve the context of words in the longer sequence I will take smaller steps. For example, I have a sentence of 100 words, so I will first take a window of 20 words and make it into a sentence, then I will move five words from the starting and take another 20 words. So in my first sentence I will have words from index ```0-19``` and in my second sentence i will have words from index ```4-24``` and so on. This helps in preserving the context of the sentences.

The setting I used is ```winSize=20``` and ```step=10``` 



```python
def slidingWindow(sequence,winSize,step):
    """Returns a generator that will iterate through
       the defined chunks of input sequence. Input sequence
       must be sliceable."""
    
    # Pre-compute number of chunks to emit
    numOfChunks = round(((len(sequence)-winSize)/step)+1)

    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]
```

Make new sentece and labels list and check the length of every sentnce in ```sample_8```, if there are more than 20 words in the sentence apply sliding window on it and store them into new list


```python
new_sent = []
new_label = []

for s, l in zip(sample_8, label_8):
    if (len(s.split()))<=20:
        new_sent.append(s)
        new_label.append(l)
    if (len(s.split()))>20:
        slides = slidingWindow(s, 20, 10)
        for slide in slides:
            new_sent.append(' '.join(slide))
            new_label.append(l)
```


```python
len(new_sent)
```




    329068




```python
# Defining max_length
max_sent = max(new_sent, key=len)    # Get the longest sentence in the list
max_length = len(max_sent.split())  # split it and set the max_length
max_length
```




    17




```python
# Using Keras tokenizer
# Here I am using 10000 words to keep based on the word frequency

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)

# Tokenizing the sentences (This process may take some time depending on your corpus size)
tokenizer.fit_on_texts(new_sent)
```


```python
# Lets see what our vocabulary size is
vocab_size = len(tokenizer.word_index) + 1   # We are adding 1 here because it takes indexing from zero
vocab_size
```




    15532



Even though it shows the vocab size to be 15532 it will only use 10000 words as we have spacified above.


```python
# Sentence encoding
sent_encoded = tokenizer.texts_to_sequences(new_sent)
```


```python
# Start padding with the max_length
padded_sents = keras.preprocessing.sequence.pad_sequences(sent_encoded, maxlen=max_length, padding='post')
# Note: We are using post padding
```


```python
# Split the data into test and train
from sklearn.model_selection import train_test_split

train_set, test_set, train_label, test_label = train_test_split(padded_sents, label_8,test_size=0.33)
```


```python
from sklearn.preprocessing import OneHotEncoder

# Define one_hot_encoder object
onehot_encoder = OneHotEncoder(sparse=False)

train_labels = onehot_encoder.fit_transform(np.reshape(train_label,(-1,1)))
test_labels = onehot_encoder.fit_transform(np.reshape(test_label,(-1,1)))
```

As I know that classes are unbalanced, the most common parctice is to use class weights to award penalty to a wrong prediction. For example, if we have 2 classes with ```class_1 = 80``` and ```class_2 = 20``` samples, then we can award a penalty to our network if the sample being predicted is of ```class_2``` and is predicted to ```class_1```. To calculate penalty we take a reference class, I usually take the class with maximum samples, and then calculate penalty for other classes with this reference class. The penalty for ```class_1 = (no. of sample in reference class)/(no. of sample in class_1)``` which for our example will be ```80/80=1``` and for ```class_2 = 80/20 = 4.0```


```python
# Computing the class weights
class_ = []
weights = []

ref_class = classes[0][0]

for i, j in enumerate(classes):
    class_.append(i)
    weights.append(ref_class/j[0])
```


```python
# Creating dictonary out of the two lists
class_weight = dict(zip(class_, weights))
```


```python
class_weight
```




    {0: 1.0,
     1: 1.2129600102047324,
     2: 1.5269369731031714,
     3: 1.8610431549075253,
     4: 1.8997103186494855,
     5: 2.713755707762557,
     6: 2.8788979715410234,
     7: 2.9752816020025032}




```python
# Create sequential model
model = Sequential()
reduce_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=0, verbose=1, 
                                                mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                           patience=1, verbose=1, mode='auto')

model.add(Embedding(vocab_size, 1000, input_length=max_length))   # adding embedding layer, which we have defined earlier

model.add(LSTM(30,recurrent_regularizer=keras.regularizers.l2(0.1),return_sequences=True))     # LSTM layer 
model.add(Dropout(0.1))
model.add(LSTM(50,recurrent_regularizer=keras.regularizers.l2(0.06),return_sequences=False))
#model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_set, train_labels, validation_split=.1, epochs=10, batch_size=512, 
          verbose=1,class_weight=class_weight ,callbacks=[reduce_rate,early_stop])

```

```bash
/home/jay/.local/lib/python3.5/site-packages/keras/callbacks.py:928: UserWarning: `epsilon` argument is deprecated and will be removed, use `min_delta` insted.
  warnings.warn('`epsilon` argument is deprecated and '
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 20, 1000)          15356000  
_________________________________________________________________
lstm_1 (LSTM)                (None, 20, 30)            123720    
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 30)            0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 50)                16200     
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 408       
=================================================================
Total params: 15,496,328
Trainable params: 15,496,328
Non-trainable params: 0
Train on 68274 samples, validate on 7587 samples
Epoch 1/10
68274/68274 [==============================] - 16s 227us/step - loss: 5.3191 - acc: 0.3646 - val_loss: 2.8831 - val_acc: 0.5347
Epoch 2/10
68274/68274 [==============================] - 13s 192us/step - loss: 2.3372 - acc: 0.5942 - val_loss: 2.1347 - val_acc: 0.6038
Epoch 3/10
68274/68274 [==============================] - 13s 193us/step - loss: 1.8782 - acc: 0.6433 - val_loss: 2.0397 - val_acc: 0.6060
Epoch 4/10
68274/68274 [==============================] - 13s 194us/step - loss: 1.6965 - acc: 0.6656 - val_loss: 2.0175 - val_acc: 0.6137
Epoch 5/10
68274/68274 [==============================] - 13s 193us/step - loss: 1.5890 - acc: 0.6803 - val_loss: 2.0452 - val_acc: 0.6171

Epoch 00005: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 00005: early stopping
```

```python
scores = model.evaluate(test_set, test_labels, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
```

```bash
37365/37365 [==============================] - 23s 622us/step
Accuracy: 61.45%
```


```python
# Classification Report (Precision, Recall and F1-Score)
from sklearn.metrics import classification_report, confusion_matrix
classificationReport = classification_report(y_true, y_pred)

print(classificationReport)

print(confusion_matrix(y_true, y_pred))
```

```bash
precision    recall  f1-score   support

          0       0.73      0.47      0.57      4268
          1       0.67      0.45      0.54      4169
          2       0.79      0.67      0.73      8563
          3       0.60      0.52      0.56      2816
          4       0.52      0.46      0.49      2400
          5       0.62      0.67      0.64      5409
          6       0.46      0.75      0.57      6799
          7       0.66      0.68      0.67      2941

avg / total       0.64      0.61      0.62     37365

[[1989  106  147   80  245  769  890   42]
 [  90 1886  370  130  287  230 1039  137]
 [  58  215 5753  380  127  297 1506  227]
 [  60   95  376 1469   52  155  537   72]
 [ 114  185  101   39 1108  316  506   31]
 [ 309   49  142  102  206 3635  882   84]
 [  87  239  272  168   99  399 5114  421]
 [   3   21   99   66   22   78  645 2007]]
```

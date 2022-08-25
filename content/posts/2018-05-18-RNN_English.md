+++
title = "(UPDATED)Sequence Classification using Recurrent Neural Networks"
description = "Training Recurrent Neural Networks on the english EU summaries data"
date = "2018-05-18"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "AI", "RNN"]
+++


### Update:
Please go to the end of the post to see the updated results.

### Aim:
The aim of this model is to classify sentences. I am going to use Recurrent Neural Network for this purpose, specifically LSTMs. LSTMs have an advantage compared to simple RNN cell or GRU cell. You can learn about [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/), There is no point on going in detail here. May be I will write another post wherein I will describe RNNs and LSTMs more intutively, but for today these two links will suffice.


```python
# Import dependencies
from gensim.models import Word2Vec
import keras
import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Ignoring Gensim Warinings
import warnings
warnings.filterwarnings("ignore")

```


```python
# Let's load our model
model = Word2Vec.load('/home/jay/Saved_Models/english/english')
```

Firstly, we can not use this model directly into tensorflow. So what do we do now? Its simple we have to convert this model into numpy matrix that we can use to train our model.



```python
# Create a matrix with the shape of Length of Vocab * total Dimensions 
# (for this model it will be 6442*4000, we have 6442 words in 4000 dimension)

embedding_matrix = np.zeros((len(model.wv.vocab), 4000))

for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```


```python
embedding_matrix.shape
```




    (6442, 4000)



Thats about right, So we have loaded the model and created the embedding matrix to be used insted of embedding layer in Keras.

The next step would be to prepare the data to be feed into the model(LSTM), but we have prepared dataset for feeding to a neural network, so what is this now?? Well the dataset we created is not suitable for feeding it to neural network, there are a few steps that needs to be done before running the all fancy LSTM. 

LSTMs take a fixed lenght sentences and also it will not take raw alpha numeric values, the words are to be converted into numbers, also all the sentences are to made with fixed lenght.

In the next step we are going to do that.


```python
# Lets load the dataset using pandas

dataset = pd.read_csv('/home/jay/Data_Set_Creation/Data_to_Use.csv') 
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
# We can not use this dataframe so lets transform them to list

sent = dataset.Sentence.tolist()
label = dataset.Numerical_Label.tolist()
```

For converting the words to numerical representations we have to tokenize the words


```python
# Using Keras tokenizer
tokenizer = keras.preprocessing.text.Tokenizer()

# Tokenizing the sentences (This process may take some time depending on your corpus size)
tokenizer.fit_on_texts(sent)
```


```python
# Lets see what our vocabulary size is
vocab_size = len(tokenizer.word_index) + 1   # We are adding 1 here because it takes indexing from zero
vocab_size
```




    16024



We have 16,387 words in our vocabulary, Now let's encode the sentences to its repecitve numbers, if you don't get what I say right now be patitent, I will show an example which will clear it out. 


```python
# Sentence encoding
sent_encoded = tokenizer.texts_to_sequences(sent)
```

So, back to what I was expalining how encoding works, Let's say you have a document with only one sentence **The quick brown fox jumps over the lazy dog** and you want to use this sentence for training a RNN(LSTM), but the LSTM doesn't understand this, it only understands numbers, so how to go on with this? Well its fairly simple, you tokenize the sentence and give each word a unique number.


```python
# Lets visualize whatever is written above
print("{} \n {}".format(sent[1], sent_encoded[1]))
```

    mlc set minimum global standard ensure right seafarer decent live working condition irrespective nationality irrespective flag ship serve : [7243, 42, 372, 364, 141, 31, 46, 2048, 2921, 870, 696, 99, 1674, 1577, 1674, 1218, 374, 971]


What the hell is that? I don't understand. Well it means the word *mlc* is at index *7243* in the dictonary of our model. Don't believe me? Let's verify that.


```python
word_dict = tokenizer.word_index
word_dict['mlc']
```




    7243



Well well well, I should say I told you so. Now remember as we said that we need to make all the sentences in our dataset of a fixed length. This raises more questions, what is padding? how to do it?. Keras provides with functions to do it. So what is padding?

Padding is used to ensure that all sequences in a list have the same length, So by default ```0``` is added at the end. 
Example: We have a squence ```[1, 2, 3], [3, 4, 5, 6], [7, 8]``` which are all of different length. So if we pad it it will be ```[1, 2, 3, 0], [3, 4, 5, 6], [7, 8, 0, 0]```, Note that we have added zeros at the end of the sentences, its called *post padding*, there is also *pre padding* where the zeros are added at the start of the squence.


```python
# padding all the encoded sequences
# Before padding we need to define what should be the maximum padding length, for that we need to check what
# is the maximum length of sentence in out dataset, we can simply do it by
import heapq
heapq.nlargest(5, sent, key=len)
```




    ['lithuania adoption community acquisi lithuania adoption community acquisi lithuaniaarchif lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives lithuaniaarchives',
     'slovakia adoption community acquisi slovakia adoption community acquisi slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives slovakiaarchives',
     'bulgaria adoption community acquisi bulgaria adoption community acquisi bulgariaarchives bulgariaarchives bulgariaarchives bulgariaarchives bulgariarchives bulgariaarchives bulgariaarchives bulgariaarchives bulgariaarchives bulgariaarchives bulgariaarchives bulgariaarchives',
     'hungary adoption community acquisi hungary adoption community acquisi hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives hungaryarchives',
     'estonia adoption community acquis estonia adoption community acquis estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives estoniaarchives']




```python
sorted(sent,key=len)
```




    ['aid amount set euro per box',
     'pron aim ensure pron safe use',
     'pron valid ten year may renew',
     'pron aim ensure pron safe use',
     'eu large aid donor global oda',
     'article set value upon eu base',
     'less red tape less risky trial',
     'datum may keep file three year',
     'one year term office may renew',
     'datum may keep file three year',
     'datum may keep file three year',
     'time sector may add pron scope',
     'datum may keep file three year',
     'article set value upon eu base',
     'date bank issue euro note coin',
     'community aid eur kg type milk',
     'diet calf feed least twice day',
     'word levy use cover tax charge',
      ...]
     


This looks promising, so I would like to go ahead with padding I described earlier. Keras provides simple padding function for us to use. Before we use that we have to define ```max_lenght```, for padding we need to define the maximum size to which we pad the sequence. This is our ```max_length```. And it is highly  advisable that we use the maximum length of sentence from our corpus. 


```python
# Defining max_length
max_sent = max(sent, key=len)    # Get the longest sentence in the list
max_length = len(max_sent.split())  # split it and set the max_length
max_length
```




    20




```python
# Start padding with the max_length
padded_sents = keras.preprocessing.sequence.pad_sequences(sent_encoded, maxlen=max_length, padding='post')
# Note: We are using post padding
```


```python
padded_sents.shape
```




    (128075, 20)




```python
# Initializing weights matrix of the embedding layer
embedding_layer = keras.layers.Embedding(vocab_size, 4000, weights=[embedding_matrix], input_length= max_length, trainable=False)
```

**NOTE** that I have delibrately kept the *flag trainable=False*, because I have already trained the model once and I don't want to train or change the weight matrix any further.

### Spliting the dataset into test and train sets
We have the whole dataset but for supervised training of a deep neural network we need training and testing set. 
Also in our dataset we have all the data stored squentially, meaning that all the classes are clustered which is not ideal, so firstly we will suffle the dataset then divide it into training and testing sets.



```python
# Split the data into test and train
from sklearn.model_selection import train_test_split

train_set, test_set, train_label, test_label = train_test_split(padded_sents, label,test_size=0.33)
```


```python
train_set.shape
```




    (85810, 20)



**NOTE:** Keras expects numpy arrays, if you just provide it the training data in form of list, it will throw an error, so covert the list to arrays

Prepraing Labels is another task at hand, which needs to be done before feeding data into the neural network, agian using sklearn one hot encoding will do the trick


```python
from sklearn.preprocessing import OneHotEncoder

# Define one_hot_encoder object
onehot_encoder = OneHotEncoder(sparse=False)

train_labels = onehot_encoder.fit_transform(np.reshape(train_label,(-1,1)))
test_labels = onehot_encoder.fit_transform(np.reshape(test_label,(-1,1)))
```


```python
train_labels.shape
```




    (85810, 32)



### Details about the model
Now that we have every thing ready we can start building the model, but there are a few questions that are needed to be answered before making the model, How many layers we are gonna use, what would be the activation function, how many neurons in each layer, what kind of regularization and what not. So I am going to build a very basic model and see how it does on the data. After the first results, I will be in a state to make changes. 


```python
# Create sequential model
model = Sequential()
model.add(embedding_layer)   # adding embedding layer, which we have defined earlier
model.add(LSTM(50, return_sequences=True)) # LSTM layer with 50 units
model.add(LSTM(50, return_sequences=True)) # LSTM layer with 50 units
model.add(LSTM(50)) # LSTM layer with 50 units
model.add(Dense(32, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_set, train_labels, validation_split=.1, epochs=3, batch_size=64, verbose=1)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 20, 4000)          64096000  
    _________________________________________________________________
    lstm_16 (LSTM)               (None, 20, 50)            810200    
    _________________________________________________________________
    lstm_17 (LSTM)               (None, 20, 50)            20200     
    _________________________________________________________________
    lstm_18 (LSTM)               (None, 50)                20200     
    _________________________________________________________________
    dense_6 (Dense)              (None, 32)                1632      
    =================================================================
    Total params: 64,948,232
    Trainable params: 852,232
    Non-trainable params: 64,096,000
    _________________________________________________________________
    None
    Train on 77229 samples, validate on 8581 samples
    Epoch 1/3
    77229/77229 [==============================] - 309s 4ms/step - loss: 2.6437 - acc: 0.2746 - val_loss: 2.2583 - val_acc: 0.3832
    Epoch 2/3
    77229/77229 [==============================] - 344s 4ms/step - loss: 2.0753 - acc: 0.4300 - val_loss: 2.0260 - val_acc: 0.4301
    Epoch 3/3
    77229/77229 [==============================] - 392s 5ms/step - loss: 1.8524 - acc: 0.4794 - val_loss: 1.9480 - val_acc: 0.4477





    <keras.callbacks.History at 0x7ff3d8408128>




```python
scores = model.evaluate(test_set, test_labels, verbose=1)
```

    42265/42265 [==============================] - 107s 3ms/step



```python
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    Accuracy: 45.04%

The process was painstakingly slow, hence only performed 3 epochs, but we can conclued that there is still a chance of improvement if I have enough resources or a lot of patience(which I don't have). So I will update this post with the updated results as soon as I have them. Till then, Have a great day. 



## Update

I managed to get a gpu enabled instance on google cloud. So I decided to run the algorithm there. The result is not that good but afterall its a trial which shed light on my dataset so I decided to update this post and pin in the results.


```bash
Train on 77229 samples, validate on 8581 samples
2018-05-22 17:35:16.036372: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-05-22 17:35:16.513689: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-05-22 17:35:16.514222: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-05-22 17:35:16.514250: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-05-22 17:35:29.328739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-05-22 17:35:29.328808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2018-05-22 17:35:29.328818: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2018-05-22 17:35:29.379302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10747 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
Epoch 1/30
77229/77229 [==============================] - 148s 2ms/step - loss: 2.4540 - acc: 0.3338 - val_loss: 2.0469 - val_acc: 0.4494
Epoch 2/30
77229/77229 [==============================] - 126s 2ms/step - loss: 1.8124 - acc: 0.5052 - val_loss: 1.8633 - val_acc: 0.4860
Epoch 3/30
77229/77229 [==============================] - 126s 2ms/step - loss: 1.5552 - acc: 0.5599 - val_loss: 1.8173 - val_acc: 0.4929
Epoch 4/30
77229/77229 [==============================] - 127s 2ms/step - loss: 1.3846 - acc: 0.5952 - val_loss: 1.8096 - val_acc: 0.4868
Epoch 5/30
77229/77229 [==============================] - 124s 2ms/step - loss: 1.2465 - acc: 0.6237 - val_loss: 1.8556 - val_acc: 0.4905
Epoch 6/30
77229/77229 [==============================] - 126s 2ms/step - loss: 1.1322 - acc: 0.6489 - val_loss: 1.9120 - val_acc: 0.4833
Epoch 7/30
77229/77229 [==============================] - 127s 2ms/step - loss: 1.0355 - acc: 0.6686 - val_loss: 1.9496 - val_acc: 0.4738
Epoch 8/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.9489 - acc: 0.6863 - val_loss: 2.0237 - val_acc: 0.4705
Epoch 9/30
77229/77229 [==============================] - 127s 2ms/step - loss: 0.8778 - acc: 0.7015 - val_loss: 2.1231 - val_acc: 0.4698
Epoch 10/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.8160 - acc: 0.7145 - val_loss: 2.1606 - val_acc: 0.4650
Epoch 11/30
77229/77229 [==============================] - 129s 2ms/step - loss: 0.7668 - acc: 0.7255 - val_loss: 2.2607 - val_acc: 0.4594
Epoch 12/30
77229/77229 [==============================] - 127s 2ms/step - loss: 0.7277 - acc: 0.7324 - val_loss: 2.3309 - val_acc: 0.4572
Epoch 13/30
77229/77229 [==============================] - 126s 2ms/step - loss: 0.6896 - acc: 0.7437 - val_loss: 2.4108 - val_acc: 0.4551
Epoch 14/30
77229/77229 [==============================] - 125s 2ms/step - loss: 0.6581 - acc: 0.7490 - val_loss: 2.4212 - val_acc: 0.4541
Epoch 15/30
77229/77229 [==============================] - 127s 2ms/step - loss: 0.6302 - acc: 0.7557 - val_loss: 2.4824 - val_acc: 0.4454
Epoch 16/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.6087 - acc: 0.7588 - val_loss: 2.5772 - val_acc: 0.4463
Epoch 17/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.5889 - acc: 0.7633 - val_loss: 2.6004 - val_acc: 0.4496
Epoch 18/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.5739 - acc: 0.7672 - val_loss: 2.6675 - val_acc: 0.4369
Epoch 19/30
77229/77229 [==============================] - 129s 2ms/step - loss: 0.5528 - acc: 0.7719 - val_loss: 2.7079 - val_acc: 0.4402
Epoch 20/30
77229/77229 [==============================] - 127s 2ms/step - loss: 0.5413 - acc: 0.7748 - val_loss: 2.7494 - val_acc: 0.4378
Epoch 21/30
77229/77229 [==============================] - 126s 2ms/step - loss: 0.5309 - acc: 0.7774 - val_loss: 2.8162 - val_acc: 0.4357
Epoch 22/30
77229/77229 [==============================] - 125s 2ms/step - loss: 0.5209 - acc: 0.7780 - val_loss: 2.8276 - val_acc: 0.4360
Epoch 23/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.5125 - acc: 0.7803 - val_loss: 2.8808 - val_acc: 0.4301
Epoch 24/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.4969 - acc: 0.7844 - val_loss: 2.8847 - val_acc: 0.4360
Epoch 25/30
77229/77229 [==============================] - 127s 2ms/step - loss: 0.4885 - acc: 0.7848 - val_loss: 2.9314 - val_acc: 0.4304
Epoch 26/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.4869 - acc: 0.7844 - val_loss: 2.9672 - val_acc: 0.4328
Epoch 27/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.4778 - acc: 0.7890 - val_loss: 3.0229 - val_acc: 0.4336
Epoch 28/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.4750 - acc: 0.7870 - val_loss: 3.0632 - val_acc: 0.4298
Epoch 29/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.4697 - acc: 0.7894 - val_loss: 3.0448 - val_acc: 0.4314
Epoch 30/30
77229/77229 [==============================] - 128s 2ms/step - loss: 0.4620 - acc: 0.7892 - val_loss: 3.0588 - val_acc: 0.4236
42265/42265 [==============================] - 37s 869us/step
Accuracy: 42.71%
             precision    recall  f1-score   support

          0       0.58      0.57      0.58      2703
          1       0.26      0.31      0.28       438
          2       0.36      0.41      0.38      1530
          3       0.36      0.38      0.37      3955
          4       0.62      0.60      0.61      4794
          5       0.36      0.34      0.35      1536
          6       0.69      0.58      0.63      1712
          7       0.34      0.22      0.27       498
          8       0.29      0.33      0.31      1102
          9       0.25      0.25      0.25       829
         10       0.53      0.53      0.53      1118
         11       0.24      0.27      0.26      1126
         12       0.34      0.31      0.32       679
         13       0.39      0.36      0.37       285
         14       0.31      0.30      0.30       822
         15       0.40      0.43      0.41      2865
         16       0.31      0.31      0.31      1326
         17       0.26      0.21      0.23       506
         18       0.29      0.31      0.30       413
         19       0.23      0.18      0.20       829
         20       0.41      0.39      0.40      1198
         21       0.47      0.45      0.46      1345
         22       0.28      0.26      0.27       409
         23       0.26      0.26      0.26       344
         24       0.47      0.54      0.50      3541
         25       0.42      0.37      0.40      1271
         26       0.57      0.60      0.59      1549
         27       0.18      0.14      0.16       600
         28       0.51      0.47      0.49       732
         29       0.31      0.31      0.31       307
         30       0.39      0.34      0.37       639
         31       0.30      0.30      0.30      1264

avg / total       0.43      0.43      0.43     42265

[[1547    3   27 ...    2    4   23]
 [   4  135   13 ...    0    0   54]
 [  27   21  631 ...    3    2   32]
 ...
 [   4    2    5 ...   94    3    1]
 [   3    7    1 ...    2  220   19]
 [  28   62   35 ...    0   19  374]]

```

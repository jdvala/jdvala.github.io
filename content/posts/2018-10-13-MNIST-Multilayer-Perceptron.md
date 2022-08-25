+++
title =  "Multilayer Perceptron"
description =  "This is what happens when we club up multiple of something."
date = "2018-10-13"
author = "Jay Vala"
tags = ["python", "text", "nlp", "analysis", "AI", "MLP"]
+++


# Introduction to Deep Learning
## Assignment 1: Multilayer Perceptron MNIST Dataset

> NOTE: This Tutorial was done on *google colab* so it includes some google cloud helper functions.

In this assignment we are going to build a very basic deep learning model called as multilayer perceptron. Also we are going to expriment on it a little bit as suggested on the course website.

### Multilayer Perceptron

Multilayer perceptron is a type of feed forward network, it has minimum 3 layers(although we don't count input and output as layers) input layer, hidden layer and output layer. It uses **backpropogation**, a technique for superviesed learning, where in it calculates error between given input and the ouput produce by the network and propogates error backwards. This is very interseting concept but it is a topic for some other time. It uses **non-linear**  activation function to introduce non-linearity because most of the real world data is non-linear in nature. You can learn more about them [here](http://deeplearning.net/tutorial/mlp.html).


```python
# importing libraries
import tensorflow as tf
```


```python
# uploading the data to the drive
from google.colab import files
files.upload()
```
    Saving t10k-images-idx3-ubyte.gz to t10k-images-idx3-ubyte.gz
    Saving t10k-labels-idx1-ubyte.gz to t10k-labels-idx1-ubyte.gz
    Saving train-images-idx3-ubyte.gz to train-images-idx3-ubyte.gz
    Saving train-labels-idx1-ubyte.gz to train-labels-idx1-ubyte.gz
    

Once all the data has been loaded we need to convert this data into some machine readable format, for this on the course website there is a script called  *conversion.py* which is given for this same purpose, I will stick to this method as it is suggested on the course website. However, this is one of the method to obtain the MNIST data, there are other eaiser methods which can be used. For example, 

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```

This is the most simple one, but you can also download the ready made csv file from Kaggle website [here](https://www.kaggle.com/c/digit-recognizer/data) 




```python
# Uploading the conversion script
from google.colab import files
files.upload()
```



     <input type="file" id="files-43a260c5-fcc0-4b5e-8cfc-a08fc2ad3ecf" name="files[]" multiple disabled />
     <output id="result-43a260c5-fcc0-4b5e-8cfc-a08fc2ad3ecf">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving conversions.py to conversions.py
    



Running this conversion file from here we can use the magic function of jupyter notebook and run them from herem, but before that we need to unpack these files to use them or to be able to covert them using the conversion script. So for that we are going to use *gzip*  library which is inbuilt in the python library module. Lets do that


```python
# Iterating over the files in the folder, checking for '.gz' extinsion and then unpacking them and saving them
import os
import sys
import gzip
import shutil

# Creating a new folder to save all the uncompressed files

dirToSave = os.path.join(os.getcwd(),'data')

try:
    os.stat(dirToSave)
except:
    os.mkdir(dirToSave)
    
for files in os.listdir():
    if files.endswith('.gz'):
        fileName = files.split('.')[0]
        with gzip.open(os.path.join(files), 'rb') as f_input, open(os.path.join(dirToSave,fileName), 'wb') as f_output:
            shutil.copyfileobj(f_input, f_output)

```

Now that the folder for the data is created I will move the conversion file to the same folder and run the file to do its magic


```python
!ls
```

    conversions.py	t10k-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz
    data		t10k-labels-idx1-ubyte.gz
    sample_data	train-images-idx3-ubyte.gz
    


```python
!mv conversions.py /content/data
```


```python
! cd data && ls && pwd

```

    conversions.py		t10k-labels-idx1-ubyte	 train-labels-idx1-ubyte
    t10k-images-idx3-ubyte	train-images-idx3-ubyte
    /content/data
    

Now lets run the conversion file


```python
!cd data/ && python3 conversions.py -c -n
```

    Converting to .csv...
    Training data...
    Test data...
    ...Done.
    
    Converting to .npy...
    Training data...
    Test data...
    ...Done.
    

As the training and testing data csv files are created, lets examin them to see if they are actually there. I do this just to be sure


```python
! cd data/ && ls
```

    conversions.py	     mnist_train.csv	     t10k-labels-idx1-ubyte
    mnist_test.csv	     mnist_train_imgs.npy    train-images-idx3-ubyte
    mnist_test_imgs.npy  mnist_train_lbls.npy    train-labels-idx1-ubyte
    mnist_test_lbls.npy  t10k-images-idx3-ubyte
    


```python
# Loading the csv files for use
import pandas as pd

train_df = pd.read_csv('/content/data/mnist_train.csv', index_col=False)
test_df = pd.read_csv('/content/data/mnist_test.csv', index_col=False)
```


```python
train_df.shape
```




    (59999, 785)




```python
test_df.shape
```




    (9999, 785)




```python
train_df.head()
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
      <th>5</th>
      <th>0</th>
      <th>0.1</th>
      <th>0.2</th>
      <th>0.3</th>
      <th>0.4</th>
      <th>0.5</th>
      <th>0.6</th>
      <th>0.7</th>
      <th>0.8</th>
      <th>...</th>
      <th>0.608</th>
      <th>0.609</th>
      <th>0.610</th>
      <th>0.611</th>
      <th>0.612</th>
      <th>0.613</th>
      <th>0.614</th>
      <th>0.615</th>
      <th>0.616</th>
      <th>0.617</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>



As we can see that the first column is the class value and we have to seperate it, so lets do that


```python
train = train_df.values
trainSet = train[:,1:]
trainLabel = train[:,:1]
test = test_df.values
testSet = test[:,1:]
testLabel = test[:,:1]
```


```python
# checking the shape again
print(trainSet.shape, trainLabel.shape, testSet.shape, testLabel.shape)
```

    (59999, 784) (59999, 1) (9999, 784) (9999, 1)
    


```python
# The labels are not useful as they are categorical hence we need to encode them
from sklearn.preprocessing import OneHotEncoder
import numpy as np
le = OneHotEncoder(handle_unknown='ignore')

le.fit(np.vstack((train[:,:1],testSet[:,:1])))

trainLabels = le.transform(trainLabel).toarray()
testLabels = le.transform(testLabel).toarray()
```


```python
testLabels[1]
```




    array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
testLabels.shape
```




    (9999, 10)



There are here so lets not worry about this too much and proceed with building the model. The first thing I will do is define my *placeholders*.
So *Placeholders*, think of them as containers, they will be used to hold data in them. They are *variables*. So when we create a *graph* in tensorflow we need to feed them data and we can not do this directly and hence we need some container to hold the data and these containers in tensorflow are called as ***Placeholder***

### About the Dataset:

MNIST data set is set of handwritten digits, which are 28 x 28 pixles, in the conversion script, what we have done is we have got the values of intensity of the each pixle. So the total input is 784, hence the shape in the palce holder will be `[None, 784]`, Here the **None** signifies the amount of data being feed and that will be decided while creating the network, for example: every dataset is of different length and when we define the batch size say in MNIST to 100 than it will create 600 batches(60000 training examples divided by 100) but if the batch size is different, its going to be because of hardware restrictions then the length of last batch might not be the same with every other batch and that makes putting anything in place of *None* "foolish", so we dont need to worry about that now. In the second palceholder *y* we will have shape `[None, 10]`, 10 here represents the number of classes that we are classifing the data into.


### Why should one not initialize the weights with 0?
It is really not a good idea to initialize the weights with *0* because all the neurons will calculate the same thing. Do we need it? No because if we would want every neuron to calculate same value than why do we need those deep networks, we can have just one neuron or one layer and do that, we don't need the deep neural networks.
> NOTE: It is not necessary to initialize the baises randomly.
 


```python
# Placeholders
X = tf.placeholder('float', [None,784]) # Placeholder for inputing the data into the network.
Y = tf.placeholder('float', [None, 10]) # Placeholder for output layer.
```


```python
# Now defining the weights and baises. The input layer have no weight and no baise, 
# and as I am creating a network with two hidden layer I will create weight and bais for hidden layer and output layer
# These weight are 2D arrays and baises are 1D arrays

hidden_1_Weight = tf.Variable(tf.random_uniform([784, 256]))  # here we need to define what input this layer will be getting and what will be its output(This output is actually the number of hidden units this layer will have)
hidden_2_Weight = tf.Variable(tf.random_uniform([256, 256]))
outputWeight = tf.Variable(tf.random_uniform([256, 10]))   # here the input to this layer will be the output of previous layer i.e. 256 and its output will be number of classes it will predict i.e. 10

# biases
hidden_1_Bias = tf.Variable(tf.random_uniform([256]))
hidden_2_Bias = tf.Variable(tf.random_uniform([256]))
outputBias = tf.Variable(tf.random_uniform([10]))
```

![Perceptron](https://upload.wikimedia.org/wikipedia/commons/8/8c/Perceptron_moj.png)

By Mayranna [CC BY-SA 3.0  (https://creativecommons.org/licenses/by-sa/3.0)], from Wikimedia Commons

So in our case the first layer will calculate     

$layer 1 = sigmoid((x * h_{1}) + b_{1})$


for layer 2 

$layer 2 = sigmoid((x * h_{2}) + b_{2})$


For the ouput layer I must add softmax activation to get the probabilities, but tensorflow has made it easy in such a way that when we calculate the loss we can do that. 

If we don't want to do it for some reason we can add softmax here in the network defination and then just calculate loss in the loss function without applying softmax there a simple `cross_entropy` from tenosrflow can do that job.







```python
# Lets define the function for multilayer 
def perceptron(x):
     # Hidden fully connected layer with 128 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, hidden_1_Weight), hidden_1_Bias))
    # Hidden fully connected layer with 128 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, hidden_2_Weight), hidden_2_Bias))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, outputWeight) + outputBias
    
    return out_layer

```


```python
# build the network
preceptron_ = perceptron(X)
```

Loss function minimizes the error between the classification and the original label. 
I am using here, `softmax_cross_entropy_with_logits` which is

$cross entropy = - \sum_{1}^{N} y_{o,c} Log(p,c)$


where, 

N: total number of classes

y: if the classification is correct, i.e.  class label *c* belongs to the observation *o*

p: predicted probability of observation *o* belongs to class *c* 



So, `softmax_cross_entropy_with_logits` will be 

$softmax crossentropy with logits = softmax(- \sum_{1}^{N} y_{o,c} Log(p,c))$


```python
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preceptron_, labels=Y)) 

optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # I am using AdamOprimizer optimizer as I think it works better in most of the cases, it comes from my personal experience.
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()
```

Now as all the variables are initialized we can move to create batches to feed to our network


```python
def batches(data, label,batch_size):
    for i in range(0,data.shape[0],batch_size):
        yield [data[i:i+batch_size], label[i:i+batch_size]]
```


```python
batch_size = 100
```


```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batches = len(train)//batch_size
    print(total_batches)
    for epoch in range(200):
        avg_cost = 0
        batchedData  = batches(trainSet,trainLabels,batch_size)
        # Loop over all batches
        for batch_data in batchedData:
            trainSetBatch = batch_data[0]
            trainLabelBatch = batch_data[1]
            _, c = sess.run([train_op, loss_op], feed_dict={X: trainSetBatch,Y: trainLabelBatch})
            # Compute average loss
            avg_cost += c / total_batches
            
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            
    print("Optimization Finished!")
    # Test model
    pred = tf.nn.softmax(preceptron_)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: testSet, Y: testLabels}))
```

    599
    Epoch: 0001 cost= 233697.866718579
    Epoch: 0002 cost= 13862.419193223
    Epoch: 0003 cost= 15239.867756075
    Epoch: 0004 cost= 14801.515713954
    Epoch: 0005 cost= 13218.383375674
    Epoch: 0006 cost= 15048.362955593
    Epoch: 0007 cost= 15605.303243443
    Epoch: 0008 cost= 14244.286502895
    Epoch: 0009 cost= 14137.246602818
    Epoch: 0010 cost= 13643.962565580
    Epoch: 0011 cost= 14308.345506671
    Epoch: 0012 cost= 13012.518639057
    Epoch: 0013 cost= 12799.497285821
    Epoch: 0014 cost= 13468.869916250
    Epoch: 0015 cost= 11905.231280484
    Epoch: 0016 cost= 11331.125025601
    Epoch: 0017 cost= 10646.379243625
    Epoch: 0018 cost= 9626.861044366
    Epoch: 0019 cost= 9001.553334091
    Epoch: 0020 cost= 7626.579372336
    Epoch: 0021 cost= 6334.278111014
    Epoch: 0022 cost= 5293.664380053
    Epoch: 0023 cost= 3738.413927234
    Epoch: 0024 cost= 2760.077281130
    Epoch: 0025 cost= 2006.898739831
    Epoch: 0026 cost= 1393.388739605
    Epoch: 0027 cost= 1061.398122862
    Epoch: 0028 cost= 720.131391260
    Epoch: 0029 cost= 528.895305433
    Epoch: 0030 cost= 371.910963326
    Epoch: 0031 cost= 289.990093832
    Epoch: 0032 cost= 211.158631602
    Epoch: 0033 cost= 141.923613281
    Epoch: 0034 cost= 112.115534271
    Epoch: 0035 cost= 90.701224615
    Epoch: 0036 cost= 94.358168000
    Epoch: 0037 cost= 61.984902694
    Epoch: 0038 cost= 60.687041363
    Epoch: 0039 cost= 42.428540849
    Epoch: 0040 cost= 45.945703943
    Epoch: 0041 cost= 35.301657116
    Epoch: 0042 cost= 33.531888366
    Epoch: 0043 cost= 24.714800139
    Epoch: 0044 cost= 26.019582580
    Epoch: 0045 cost= 17.728049701
    Epoch: 0046 cost= 17.548398735
    Epoch: 0047 cost= 39.107907718
    Epoch: 0048 cost= 21.521840305
    Epoch: 0049 cost= 8.725962788
    Epoch: 0050 cost= 9.524569323
    Epoch: 0051 cost= 5.639854976
    Epoch: 0052 cost= 13.697699557
    Epoch: 0053 cost= 28.470761760
    Epoch: 0054 cost= 11.467136455
    Epoch: 0055 cost= 10.369889293
    Epoch: 0056 cost= 7.239818920
    Epoch: 0057 cost= 11.569948143
    Epoch: 0058 cost= 11.443468886
    Epoch: 0059 cost= 8.718404733
    Epoch: 0060 cost= 13.267106948
    Epoch: 0061 cost= 9.522318864
    Epoch: 0062 cost= 7.704388247
    Epoch: 0063 cost= 6.505781692
    Epoch: 0064 cost= 4.287839644
    Epoch: 0065 cost= 4.067495077
    Epoch: 0066 cost= 5.511625610
    Epoch: 0067 cost= 9.277953480
    Epoch: 0068 cost= 5.771217371
    Epoch: 0069 cost= 7.822138380
    Epoch: 0070 cost= 5.067580908
    Epoch: 0071 cost= 3.542238488
    Epoch: 0072 cost= 2.314385025
    Epoch: 0073 cost= 4.829883155
    Epoch: 0074 cost= 6.732986233
    Epoch: 0075 cost= 4.147152578
    Epoch: 0076 cost= 4.814162664
    Epoch: 0077 cost= 4.195318871
    Epoch: 0078 cost= 4.816954190
    Epoch: 0079 cost= 6.405923215
    Epoch: 0080 cost= 3.548350426
    Epoch: 0081 cost= 6.843513339
    Epoch: 0082 cost= 3.730189537
    Epoch: 0083 cost= 2.395861579
    Epoch: 0084 cost= 1.291224121
    Epoch: 0085 cost= 2.609014182
    Epoch: 0086 cost= 8.136346506
    Epoch: 0087 cost= 6.911538018
    Epoch: 0088 cost= 1.613900936
    Epoch: 0089 cost= 1.792066425
    Epoch: 0090 cost= 1.509146715
    Epoch: 0091 cost= 2.816762689
    Epoch: 0092 cost= 5.840670672
    Epoch: 0093 cost= 4.645757224
    Epoch: 0094 cost= 2.435126027
    Epoch: 0095 cost= 8.339016823
    Epoch: 0096 cost= 1.770155966
    Epoch: 0097 cost= 1.422567249
    Epoch: 0098 cost= 1.212486510
    Epoch: 0099 cost= 1.109428637
    Epoch: 0100 cost= 3.898369765
    Epoch: 0101 cost= 3.376135472
    Epoch: 0102 cost= 4.315077318
    Epoch: 0103 cost= 4.274662701
    Epoch: 0104 cost= 3.012072559
    Epoch: 0105 cost= 1.683379329
    Epoch: 0106 cost= 1.264098068
    Epoch: 0107 cost= 1.163033118
    Epoch: 0108 cost= 1.178344884
    Epoch: 0109 cost= 3.371009719
    Epoch: 0110 cost= 2.942797631
    Epoch: 0111 cost= 1.653774604
    Epoch: 0112 cost= 2.065821347
    Epoch: 0113 cost= 9.031573909
    Epoch: 0114 cost= 2.762784667
    Epoch: 0115 cost= 3.431726249
    Epoch: 0116 cost= 1.516621503
    Epoch: 0117 cost= 1.402665269
    Epoch: 0118 cost= 3.369300493
    Epoch: 0119 cost= 1.436823709
    Epoch: 0120 cost= 0.989465624
    Epoch: 0121 cost= 1.465438136
    Epoch: 0122 cost= 2.069973967
    Epoch: 0123 cost= 1.271443909
    Epoch: 0124 cost= 1.228639121
    Epoch: 0125 cost= 3.035652603
    Epoch: 0126 cost= 4.367832982
    Epoch: 0127 cost= 1.883182552
    Epoch: 0128 cost= 0.988018134
    Epoch: 0129 cost= 1.219982436
    Epoch: 0130 cost= 2.460366394
    Epoch: 0131 cost= 1.444995319
    Epoch: 0132 cost= 0.796145169
    Epoch: 0133 cost= 1.866650649
    Epoch: 0134 cost= 2.729383657
    Epoch: 0135 cost= 1.498433346
    Epoch: 0136 cost= 0.894979498
    Epoch: 0137 cost= 0.685542778
    Epoch: 0138 cost= 1.086774313
    Epoch: 0139 cost= 1.573417341
    Epoch: 0140 cost= 1.511079420
    Epoch: 0141 cost= 3.592942662
    Epoch: 0142 cost= 1.756971295
    Epoch: 0143 cost= 2.371372009
    Epoch: 0144 cost= 1.350900717
    Epoch: 0145 cost= 3.918655687
    Epoch: 0146 cost= 1.380350943
    Epoch: 0147 cost= 1.348579527
    Epoch: 0148 cost= 1.050694610
    Epoch: 0149 cost= 0.912132167
    Epoch: 0150 cost= 0.782553376
    Epoch: 0151 cost= 0.597382387
    Epoch: 0152 cost= 0.594046318
    Epoch: 0153 cost= 0.592238037
    Epoch: 0154 cost= 0.592521275
    Epoch: 0155 cost= 0.594104953
    Epoch: 0156 cost= 0.589922921
    Epoch: 0157 cost= 0.703515567
    Epoch: 0158 cost= 4.097817484
    Epoch: 0159 cost= 3.709944400
    Epoch: 0160 cost= 3.080095094
    Epoch: 0161 cost= 1.277377238
    Epoch: 0162 cost= 0.838328730
    Epoch: 0163 cost= 0.941523782
    Epoch: 0164 cost= 1.633839919
    Epoch: 0165 cost= 1.591967852
    Epoch: 0166 cost= 0.826101982
    Epoch: 0167 cost= 0.960754743
    Epoch: 0168 cost= 0.991728673
    Epoch: 0169 cost= 1.230600796
    Epoch: 0170 cost= 1.051815032
    Epoch: 0171 cost= 1.671720224
    Epoch: 0172 cost= 1.550530605
    Epoch: 0173 cost= 0.948061457
    Epoch: 0174 cost= 0.775590160
    Epoch: 0175 cost= 0.687840205
    Epoch: 0176 cost= 0.688999365
    Epoch: 0177 cost= 1.429095754
    Epoch: 0178 cost= 3.105333766
    Epoch: 0179 cost= 1.452365677
    Epoch: 0180 cost= 1.176313281
    Epoch: 0181 cost= 0.766753364
    Epoch: 0182 cost= 3.400296347
    Epoch: 0183 cost= 1.933165025
    Epoch: 0184 cost= 1.105114213
    Epoch: 0185 cost= 1.771966526
    Epoch: 0186 cost= 2.085469130
    Epoch: 0187 cost= 0.894252141
    Epoch: 0188 cost= 1.149476664
    Epoch: 0189 cost= 0.906636394
    Epoch: 0190 cost= 1.220944617
    Epoch: 0191 cost= 0.875083064
    Epoch: 0192 cost= 1.119831198
    Epoch: 0193 cost= 0.861701711
    Epoch: 0194 cost= 0.755726190
    Epoch: 0195 cost= 1.848937889
    Epoch: 0196 cost= 1.442416712
    Epoch: 0197 cost= 5.017063204
    Epoch: 0198 cost= 0.906185662
    Epoch: 0199 cost= 1.311479466
    Epoch: 0200 cost= 0.750225017
    Optimization Finished!
    Accuracy: 0.7509751
    

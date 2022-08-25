+++
title =  "Tensorboard"
description =  "A picture is worth thousand words"
date = "2018-10-23"
author = "Jay Vala"
tags = ["python", "text", "nlp", "analysis", "AI", "tensorflow", "visualization"]
+++


In this assignment we are going to use *Tensorboard* a tool provided with tensorflow to visualize and debug(if necessary) our neural networks visually. I am going to visualize all the parameteres that have the properties which enables us to debug our neural networks or gives us a fair intution on how and what goes wrong.

I will try and do this assignment in google colab, although it will be difficult, I will try my best.

For using tensorboard on google colab we need something called  we just need to install a single library using pip.


```python
!pip install tensorboardcolab
```

    Requirement already satisfied: tensorboardcolab in /usr/local/lib/python3.6/dist-packages (0.0.19)


Now that we have everything setup and running we have to train a model to get some output on tensorboard


```python
import tensorflow as tf

from tensorboardcolab import TensorBoardColab

```

    Using TensorFlow backend.



```python
# Getting MNIST
# uploading the data
from google.colab import files
files.upload()
```

Now that the data is loaded, I will create train set and test set from these files and then convert the labels into *One-hot encoded ones*


```python
import pandas as pd
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')
```


```python
# Seperating labels from the training and testing data
train = train_df.values
trainSet = train[:,1:]
trainLabel = train[:,:1]
test = test_df.values
testSet = test[:,1:]
testLabel = test[:,:1]
```


```python
# converting labels into one-hot-encoded values
from sklearn.preprocessing import OneHotEncoder
le = OneHotEncoder(handle_unknown='ignore')

le.fit(trainLabel.reshape(-1,1))
trainLabels = le.transform(trainLabel.reshape(-1,1)).toarray()
testLabels = le.transform(testLabel.reshape(-1,1)).toarray()
```


```python
# Now defining the weights and baises. The input layer have no weight and no baise, 
# and as I am creating a network with two hidden layer I will create weight and bais for hidden layer and output layer
# These weight are 2D arrays and baises are 1D arrays

hidden_1_Weight = tf.Variable(tf.random_uniform([784, 256]),name='weight_hidden_layer1')  # here we need to define what input this layer will be getting and what will be its output(This output is actually the number of hidden units this layer will have)
hidden_2_Weight = tf.Variable(tf.random_uniform([256, 256]), name='weight_hidden_layer2')
outputWeight = tf.Variable(tf.random_uniform([256, 10]), name='weight_output')   # here the input to this layer will be the output of previous layer i.e. 256 and its output will be number of classes it will predict i.e. 10

# biases
hidden_1_Bias = tf.Variable(tf.random_uniform([256]), name='bais_hidden_layer1')
hidden_2_Bias = tf.Variable(tf.random_uniform([256]), name='bais_hidden_layer2')
outputBias = tf.Variable(tf.random_uniform([10]), name='bais_output')
```

Now as the model weights and baises are completly defined, I will now add summaries to be displayed on tensorboard. So we have different summary options for different things, for example, we have `tf.summary.scalar` for displaying single values such as loss, accuracy and then we have `tf.summary.histogram` to visualize matries such as weights and baises . Also we can visualize layer activations and all other variables, but for the scope of this assignment I will only focus on a few of them.


```python
# # Adding summaries
hist_hidden_1 = tf.summary.histogram('hidden_1_weight', hidden_1_Weight)
hist_hidden_2 = tf.summary.histogram('hidden_2_weight', hidden_2_Weight)
output_layer_hist = tf.summary.histogram('Output_Weight', outputWeight)
```


```python
# PlaceHolders
X = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
Y = tf.placeholder(tf.float32, [None, 10], name='class_placeholder')
```


```python
# Lets define the function for multilayer perceptron
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
output = perceptron(X)
```

For this assignment I am also going to calculate accuracy on  each step or epoch just for visualization.



```python
# Evaluate model
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(output), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# adding accuracy scalar to the tensorboard
accuracy_vis = tf.summary.scalar('accu', accuracy)
```


```python
# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y)) 
loss_vis = tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01) 
train_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()
```


```python
# Merge all summaries into a single operator
merged_summary = tf.summary.merge_all()

```


```python
batch_size = 100
def batches(data, label,batch_size):
    for i in range(0,data.shape[0],batch_size):
        yield [data[i:i+batch_size], label[i:i+batch_size]]
```


```python
summary_writer = tf.summary.FileWriter('/content/log',sess.graph)
```


```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    total_batches = len(train)//batch_size
    print(total_batches)
    for epoch in range(10):
        avg_cost = 0
      
        # Loop over all batches
        batchedData  = batches(trainSet,trainLabels,batch_size)
        for batch_data in batchedData:
            _,c, summary_,acc = sess.run([train_op, loss, merged_summary, accuracy], feed_dict={X: batch_data[0],Y: batch_data[1]})
            # run summary and accuracy 
            #summary = sess.run([],feed_dict={X: batch_data[0], Y: batch_data[1]})
            
            # Compute average loss
            avg_cost += c / total_batches
            
            # write all the data to the file
            summary_writer.add_summary(summary_, epoch)
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f} accuracy={:.2f}%".format(avg_cost,acc*100 ))
            
            
    print("Optimization Finished!")
    # Test model
    pred = tf.nn.softmax(output)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: testSet, Y: testLabels}))
```

    599
    Epoch: 0001 cost=343591.474908259 accuracy=84.85%
    Epoch: 0002 cost=1363.470883059 accuracy=89.90%
    Epoch: 0003 cost=353.140610641 accuracy=93.94%
    Epoch: 0004 cost=145.729113662 accuracy=96.97%
    Epoch: 0005 cost=71.671148529 accuracy=98.99%
    Epoch: 0006 cost=48.250662688 accuracy=96.97%
    Epoch: 0007 cost=34.012532867 accuracy=98.99%
    Epoch: 0008 cost=32.737998139 accuracy=95.96%
    Epoch: 0009 cost=29.617493640 accuracy=98.99%
    Epoch: 0010 cost=28.458745954 accuracy=96.97%
    Optimization Finished!
    Accuracy: 0.93049306



```python
tbc=TensorBoardColab(graph_path='/content/log')
```

    Wait for 8 seconds...
    TensorBoard link:
    http://4304f716.ngrok.io

Now you can copy the above link or click [here](http://4304f716.ngrok.io) to visit tensorboard page and explore how the training is done.

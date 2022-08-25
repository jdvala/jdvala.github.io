+++
title =  "Finding K in K-Means"
description =  "The misterious $k$"
date = "2018-07-09"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "analysis", "kmeans", "clustering", "findk", "silhouette-analysis", "elbow-analysis"]
+++
# Finding K in K-Means

So when applying K-Means Clustring, one comes at a point where he/she has to decide how many cluster they want, now one can not go and tell 2,3, or 4 clusters there should be a some evidence that clustering the data into *'k'* clusters will yeild good results. So I met this problem, after searching for what can be done about this problem I stumble upon a something called **'Silhouette Analysis'**

### Silhouette Analysis

Silhouette Analysis takes into considration of how well a particular data point lies within a given cluster. So lets take an example to understand how this works, Say we are using K-Means Clustring for *'k'* number of clusters. Now for a data point *'i'*, we define average distance from all points in the same cluster as *avg(i) = a(i)*, so we can interpret *a(i)* as measure of how well that point belongs in the cluster or to put it simply how well that point belongs in that cluster, so smaller the value, better the assignment of that data point to that cluster. Similarly, lets say *b(i)* is the average dissimlarity, that means *b(i)* the lowest average distance of data point *i*  to all points in any other cluster, of which *i* is not a member of. The cluster with this lowest average dissimilarity is said to be the "neighbouring cluster" of *i* because it is the next best fit cluster for point *i*


```python
from IPython.display import Math
Math(r's(i) = \frac{b(i)-a(i)}{max(b(i),a(i))}')
```




$$s(i) = \frac{b(i)-a(i)}{max(b(i),a(i))}$$




```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
```


```python
# Getting all the english and german data

import os, re

label = []
content = []

for root, dirs, files in os.walk('/home/jay/Data/English/'):
    for file in files:
        topic = root.split(os.path.sep)[-2]
        if file.endswith('.txt') and file != 'log.txt':
            with open(os.path.join(root, file)) as f:
                content.append(f.read())
                label.append(topic)
                

        
for root, dirs, files in os.walk('/home/jay/Data/German/'):
    for file in files:
        topic = root.split(os.path.sep)[-2]
        if file.endswith('.txt') and file != 'log.txt':
            with open(os.path.join(root, file)) as i:
                content.append(i.read())
                label.append(topic)


```


```python
# running the k-means for 32 clusters
# creating tf-idf matrix

tfidf_matrix_constructor = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, ngram_range=(1,3))

tfidf_matrix = tfidf_matrix_constructor.fit_transform(content)

# printing the shape of the resultant matrix to ensure I am right
print(tfidf_matrix.shape)
```

    (9986, 59)



```python
# now applying k-means for range 1-32 and calculating silhouette_score for each cluster

for k in range(2 ,33):
    
    # define k-means constructor
    kmeans = KMeans(n_clusters=k,random_state=10)
    
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
    
    # Calculating silhouette_score for k
    
    score = silhouette_score(tfidf_matrix.toarray(), cluster_labels, random_state=10)
    print("The silhouette score for {} clusters is {}".format(k,score))
```

    The silhouette score for 2 clusters is 0.2092956601357311
    The silhouette score for 3 clusters is 0.1463292693620873
    The silhouette score for 4 clusters is 0.14046381676564526
    The silhouette score for 5 clusters is 0.14694926636316893
    The silhouette score for 6 clusters is 0.14478803184773398
    The silhouette score for 7 clusters is 0.17189746569523418
    The silhouette score for 8 clusters is 0.17280711786498046
    The silhouette score for 9 clusters is 0.18248360175164505
    The silhouette score for 10 clusters is 0.185809463489821
    The silhouette score for 11 clusters is 0.19185632542163303
    The silhouette score for 12 clusters is 0.19569188868141868
    The silhouette score for 13 clusters is 0.19476772871615036
    The silhouette score for 14 clusters is 0.20251183229887623
    The silhouette score for 15 clusters is 0.19103405902025516
    The silhouette score for 16 clusters is 0.19483617751124183
    The silhouette score for 17 clusters is 0.19140067934731583
    The silhouette score for 18 clusters is 0.19934328007322114
    The silhouette score for 19 clusters is 0.19446860597424745
    The silhouette score for 20 clusters is 0.19904443285585602
    The silhouette score for 21 clusters is 0.195384733891542
    The silhouette score for 22 clusters is 0.19990294699010308
    The silhouette score for 23 clusters is 0.20434103038355314
    The silhouette score for 24 clusters is 0.20162229208943913
    The silhouette score for 25 clusters is 0.20001554425267193
    The silhouette score for 26 clusters is 0.20297321222570094
    The silhouette score for 27 clusters is 0.2023863657790054
    The silhouette score for 28 clusters is 0.19960601038803405
    The silhouette score for 29 clusters is 0.1985121190442007
    The silhouette score for 30 clusters is 0.20256036983356498
    The silhouette score for 31 clusters is 0.2001042315801357
    The silhouette score for 32 clusters is 0.20251412402591076


So the higher the value of Silhouette Score, better the number of clusters.

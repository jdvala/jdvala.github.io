+++
title = "Quantitative Analysis of Dataset"
description =  "Looking at the numbers...closly"
date = "2018-05-22"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp", "AI", "analysis"]
+++


I have got bad results after training the RNN. Hence I have decided to analyse the data set and see the distribution of samples across different classes. This may give me some idea to why my network is performing poor.


```python
import pandas as pd
from collections import Counter
import numpy as np
```


```python
data = pd.read_csv('/home/jay/Data_Set_Creation/Data_to_Use.csv')
```


```python
data.columns
```




    Index(['Sentence', 'Label', 'Numerical_Label'], dtype='object')




```python
labels =  data.Numerical_Label.tolist()
```

After the labels are into the list, I will count the number of samples in each class to see what is the distribution.


```python
count = Counter()
for label in labels:
    count[label] +=1
```


```python
count
```




    Counter({0: 8115,
             1: 1353,
             2: 4643,
             3: 12267,
             4: 14056,
             5: 4660,
             6: 5227,
             7: 1520,
             8: 3319,
             9: 2525,
             10: 3310,
             11: 3438,
             12: 2046,
             13: 838,
             14: 2434,
             15: 8628,
             16: 4052,
             17: 1563,
             18: 1277,
             19: 2585,
             20: 3561,
             21: 4233,
             22: 1211,
             23: 1026,
             24: 10869,
             25: 3840,
             26: 4786,
             27: 1887,
             28: 2151,
             29: 917,
             30: 1900,
             31: 3838})




```python
value = list(count.values())
keys = list(count.keys())
```

Find the minimum class, that is class with minimum number of samples


```python
np.min(value)
```




    838



Calculating the percentage of samples in each class.


```python
per = [(val/np.sum(value))*100 for val in value]
```


```python
np.mean(per)
```




    3.125




```python
sorted((zip(per, keys)))
```




    [(0.6543041186804607, 13),
     (0.7159867265274253, 29),
     (0.8010931095061488, 23),
     (0.9455397228186609, 22),
     (0.9970720281085301, 18),
     (1.056412258442319, 1),
     (1.1868046066757758, 7),
     (1.2203786843646303, 17),
     (1.4733554557876245, 27),
     (1.48350575834472, 30),
     (1.5975014639859457, 12),
     (1.6794846769471012, 28),
     (1.9004489556900255, 14),
     (1.9715010735896936, 9),
     (2.0183486238532113, 19),
     (2.5844231895373806, 10),
     (2.5914503220769083, 8),
     (2.684364630099551, 11),
     (2.780402108139762, 20),
     (2.996681631856334, 31),
     (2.998243216865118, 25),
     (3.163771227796213, 16),
     (3.3050946710911577, 21),
     (3.62521959789186, 2),
     (3.6384930704665233, 5),
     (3.73687292601991, 26),
     (4.081202420456764, 6),
     (6.336131173140738, 0),
     (6.736677727893812, 15),
     (8.48643373023619, 24),
     (9.577981651376147, 3),
     (10.97481944173336, 4)]



It looks like we have a very unbalanced dataset. Also the number of samples in some of the classes are very less. This was the reason why my model started overfitting after just 3 epoch. 

There are numerious way in which this problem can be takcled. 
* Increase the number of samples.
* Apply Class weights while training.
* Use smaller architecture for training.

I will try to and apply all these in the next trail.

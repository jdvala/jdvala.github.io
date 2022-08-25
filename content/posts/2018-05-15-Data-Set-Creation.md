+++
title = "English EurLex dataset creation"
description = "Creating Dataset out of English Corpus"
date = "2018-05-15"
author = "Jay Vala"
tags = ["python", "text", "dataset", "nlp"]
+++


In this notebook I will create the data set for english corpus. I will assign every senetence in the corpus with it corresponding label. So if the sentence is from Agriculture it will be assigned label decided for Agriculture and so on.


```python
import os, sys
import pandas as pd
```


```python
# First lets create list of topics
label = []
for root, dirs, files in os.walk('/home/jay/GITHUB/Data_Thesis/Ready/'):
    label.append(dirs)
```


```python
label[0]
```




    ['transport',
     'audiovisual_and_media',
     'enlargement',
     'internal_market',
     'justice_freedom_security',
     'institutional_affairs',
     'economic_and_monetary_affairs',
     'human_rights',
     'agriculture',
     'enterprise',
     'food_safety',
     'consumers',
     'competition',
     'humanitarian_aid',
     'maritime_affairs_and_fisheries',
     'environment',
     'regional_policy',
     'external_trade',
     'fight_against_fraud',
     'research_innovation',
     'development',
     'external_relations',
     'foreign_and_security_policy',
     'culture',
     'employment_and_social_policy',
     'energy',
     'education_training_youth',
     'customs',
     'taxation',
     'budget',
     'public_health',
     'information_society']




```python
len(label[0])
```




    32




```python
# Make a dictionary to hold the labels and corresponding category
labels = dict(list(enumerate(label[0])))
```


```python
labels
```




    {0: 'transport',
     1: 'audiovisual_and_media',
     2: 'enlargement',
     3: 'internal_market',
     4: 'justice_freedom_security',
     5: 'institutional_affairs',
     6: 'economic_and_monetary_affairs',
     7: 'human_rights',
     8: 'agriculture',
     9: 'enterprise',
     10: 'food_safety',
     11: 'consumers',
     12: 'competition',
     13: 'humanitarian_aid',
     14: 'maritime_affairs_and_fisheries',
     15: 'environment',
     16: 'regional_policy',
     17: 'external_trade',
     18: 'fight_against_fraud',
     19: 'research_innovation',
     20: 'development',
     21: 'external_relations',
     22: 'foreign_and_security_policy',
     23: 'culture',
     24: 'employment_and_social_policy',
     25: 'energy',
     26: 'education_training_youth',
     27: 'customs',
     28: 'taxation',
     29: 'budget',
     30: 'public_health',
     31: 'information_society'}



Now that we have the numbers we can create dataset and for training the RNN. 


```python
# Lets go through the files and take file name, dir name, and subdir name

sent = []  # List to store the sentence
label = [] # List to store the label(alpha)
label_num = []  # List to store numerical label

for root, dirs, files in os.walk('/home/jay/GITHUB/Data_Thesis/Ready/'):
    for file in files:
        if file.endswith('.txt'):
            topic = root.split(os.path.sep)[-2]  
            with open(os.path.join(root, file)) as f:
                cont = f.readlines()  ## we prefere to get the sentences in the list form so that it is easy for us to label it
             
                
                # Store every sentence with its corresponding labels in a list
                # Get the label
                for key,value in labels.items():
                    if str(value) == str(topic):      # If the dir is equal to any value in the labels dict then
                        label_ = key
                        
                for line in cont:
                    sent.append(line)
                    label.append(topic)
                    label_num.append(label_)
                           
                
```


```python
# Just checking to see if all of them are equal
len(sent) == len(label) == len(label_num)
```




    True



We are going to make a dataframe to hold the dataset, we could have used a dictonary for that but it would have been no fun!!


```python
# Dataframe to hold the dataset, for now we are going to create a dataframe with just the columns as we are going to
# fill in the columns later

dataset = pd.DataFrame(data = list(map(list, zip(sent,label,label_num))),columns=['Sentence','Label', 'Numerical_Label'])
```


```python
# Visualize dataframe
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
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mlc set minimum global standard ensure right s...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pron also seek limit social dump secure fair c...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>the main point new directive follow\n</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>\n</td>
      <td>transport</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>



As we see that there are lines in our dataframe with newline charaters, So we remove these as well as the lines which are empty after removal of these new line characters


```python
dataset.Sentence =dataset.Sentence.str.strip('\n')
# Visualize dataframe
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
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mlc set minimum global standard ensure right s...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pron also seek limit social dump secure fair c...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>the main point new directive follow</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>transport</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replace the empty line with NaN values
dataset.Sentence.replace('', np.nan, inplace=True)

# Drop the NaN values
dataset = dataset.dropna()
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
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mlc set minimum global standard ensure right s...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pron also seek limit social dump secure fair c...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>the main point new directive follow</td>
      <td>transport</td>
      <td>31</td>
    </tr>
    <tr>
      <th>5</th>
      <td>monitor compliance eu country introduce effect...</td>
      <td>transport</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>



We can use this dataframe to train the RNN. Save it as CSV file to use it for later


```python
dataset.to_csv('/home/jay/Data_Set_Creation/Data_to_Use.csv',index=False)
```

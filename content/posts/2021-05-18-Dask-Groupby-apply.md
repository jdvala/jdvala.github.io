+++
title = "Groupby and Apply with Dask on pandas DataFrame"
description =  "Grouby-apply dask"
date = "2021-05-18"
author = "Jay Vala"
tags =  ["dask", "python", "pandas"]
+++

Recently, I have started using [Dask](https://dask.org/) to make some pandas computations faster, its a really good tool. It has a bit of learning curve, but its worth it.

Until now I used to use [Pandarallel](https://github.com/nalepae/pandarallel/tree/v1.5.2) for parallelization of pandas apply functions. Its a good little tool and there are many alternatives to it as well. One of it is written by a good friend of my, and its called [Mappy](https://pypi.org/project/mapply/). 

Let's, look at how to use Dask to first groupby and then apply. For the example I will use titanic dataset which you can find [here](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv). Its a csv file so lets load it,

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

df.head()
```

|   | PassengerId | Survived | Pclass |                                              Name |    Sex |  Age | SibSp | Parch |           Ticket |    Fare | Cabin | Embarked |
|--:|++++++++++++:|+++++++++:|++++++-:|++++++++++++++++++++++++++++++++++++++++++++++++--:|++++++-:|+++--:|++++++:|++++++:|+++++++++++++++--:|++++++--:|++++++:|+++++++++-|
| 0 |           1 |        0 |      3 |                           Braund, Mr. Owen Harris |   male | 22.0 |     1 |     0 |        A/5 21171 |  7.2500 |   NaN |        S |
| 1 |           2 |        1 |      1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 |     1 |     0 |         PC 17599 | 71.2833 |   C85 |        C |
| 2 |           3 |        1 |      3 |                            Heikkinen, Miss. Laina | female | 26.0 |     0 |     0 | STON/O2. 3101282 |  7.9250 |   NaN |        S |
| 3 |           4 |        1 |      1 |      Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 |     1 |     0 |           113803 | 53.1000 |  C123 |        S |
| 4 |           5 |        0 |      3 |                          Allen, Mr. William Henry |   male | 35.0 |     0 |     0 |           373450 |  8.0500 |   NaN |        S |

Now, to formulate the problem lets see how many male and female passengers survived.

```python
def survived(x):
    return x.Survived.count()
```

If we were to do it pandas way, we can do it like

```python
%time
titanic.groupby("Sex").apply(survived)
```
Which gives out 

|        | survived |     |
|++++++-:|+++++++++:|+++--|
|    Sex |          |     |
| female |        0 | 314 |
|   male |        0 | 577 |


To do this same thing with help of dask, we first need to translate the dataframe into what dask would understand, for dask and pandas a dataframe object is different.

```python
import dask.dataframe as dd

dask_dataframe = dd.from_pandas(df, npartitions=2)

dask_dataframe
```

|               | PassengerId | Survived | Pclass |   Name |    Sex |     Age | SibSp | Parch | Ticket |    Fare |  Cabin | Embarked |
|++++++++++++--:|++++++++++++:|+++++++++:|++++++-:|++++++-:|++++++-:|++++++--:|++++++:|++++++:|++++++-:|++++++--:|++++++-:|+++++++++:|
| npartitions=2 |             |          |        |        |        |         |       |       |        |         |        |          |
|             0 |       int64 |    int64 |  int64 | object | object | float64 | int64 | int64 | object | float64 | object |   object |
|           446 |         ... |      ... |    ... |    ... |    ... |     ... |   ... |   ... |    ... |     ... |    ... |      ... |
|           890 |         ... |      ... |    ... |    ... |    ... |     ... |   ... |   ... |    ... |     ... |    ... |      ... |


The above statment will translate the dataframe from pandas to something dask will understand. Now you might ask what the hell is this `npartitions` mean. Its very very simple, dask when converting the pandas dataframe object into dask dataframe object will divide the dataframe into `n` partitions so that it can be processed on different core (CPUs). This is how dask makes processing faster.

> Note: There is however one drawback of such division, if there is dependency between the dataframe partitions when processing, then dask will not be useful. The overhead of passing information between the chunks or partitions in different CPU cores will make use of dask futile for such a case. 

Once the dataframe object is translated into dask, all we have got to do is 

```python
output = dask_dataframe.groupby("Sex").apply(survived)
```

Once we run the cell, you will note that nothing happens, this is because dask works on principle of lazy execution. You can learn more about it [here](https://tutorial.dask.org/01x_lazy.html), but on a higher level when we run the above cell, dask will generate a graph of how the data will distributed and where which operation will occur.

>Note: You can use `output.visualize()` to see a graph of the execution.

So to run the actual computation you need to use compute function 

```python
output.compute()
```

|        | survived |     |
|++++++-:|+++++++++:|+++--|
|    Sex |          |     |
| female |        0 | 314 |
|   male |        0 | 577 |

This ends a small tutorial on how to use dask for groupby and apply.

I am planning on writing a whole series on dask and how its an alternative to some of the things you can do with pandas.

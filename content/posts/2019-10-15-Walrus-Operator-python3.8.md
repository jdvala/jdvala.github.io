+++
title =  "Walrus Operator - The new and shinny Python 3.8"
description =  "Welcome python 3.8 with much fun fair"
date = "2019-10-15"
author = "Jay Vala"
tags = ["python", "python3"]
+++

Yesterday we have welcomed **python 3.8** with much fun fair. The whole change log can be found [here](https://docs.python.org/3/whatsnew/3.8.html) but the highlight of the new python is the **Walrus Operator**.

> According to the official documentation `:=` will be the new walrus operator. This syntax assigns values to variables as a part of larger expression

Consider the following code snippet

```python
if len(a) > 10:
    n = len(a)
    print(f"List is too long expected len < 10 but found {n}")
```
In the above expression `len(a)` is redundent but we can not help but write it twice, but not in python 3.8 cause we have the `walrus operator`

```python
if (n:=len(a)) > 10:
    print(f"List is too long expected len < 10 but found {n}")
```

Now think of a situation where we first read a file then find the length of the text inside it and then if the length satisfies our needs we go ahead with further processing

```python
with open('<file_name.txt>') as fp:
    contents = fp.read()
    if len(contents) <= 100000:
        # process further
```

Using python 3.8
```python
with open('<file_name.txt>') as fp:
    if (contents := len(fp.read())) <= 100000:
        # further processing
```

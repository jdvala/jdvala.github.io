+++
title = "Writing my own little interactive python environment"
description =  "Inspired by a video from Sebastiaan Mathôt"
date = "2021-02-15"
author = "Jay Vala"
tags = ["python", "python3", "ipython"]
+++

Wouldn't it be cool to build just for fun one of the most used and beloved tool in Data Science and Python community, the `Interactive Python (ipython) shell`.

The other day I was watching youtube videos, learning all about python and I came across this video by Sebastiaan Mathôt, in which he had built his own `ipython` shell. I thought it was very cool and I wanted to replicate it. The link for the video is [here](https://youtu.be/uSmOry4PY0Q)

We start by thinking about how `ipython` works, we enter a valid python code and the `ipython` interpreter evaluates it and gives back the results and spawns a new line for us to write another peice of code. So how can we replicate it? Lets create our main function that gets users input and prints it out. 

```python
from itertools import count

def get_user_input():
    
    for i in count():
        yield i, input(f"In [{i}]: ")


def main():
    
    for i, user_input in get_user_input():
        pass
    
if __name__ ==  "__main__":
    main()
```

Here in the code above I could have used `while` loop but I did not want to keep track of `i` and increment it everytime. `Itertools` provide `count` function which returns `i` and we now have a for loop. 

Now if we run the above code with python we should see something familiar, nice!

```bash
In [0]: 
```

And if you keep pressing `Enter`, new lines will be spawned. So we are good on the part where we need new lines when the previous lines are executed. Cool!

As we have the first thing out of the way, where user can input their python code. Now what? Do you remember how to quit the prompt? `Ctrl + d` right? So when you run the program and press `Ctrl + d`, we get 

```bash 
In [0]: Traceback (most recent call last):
  File "test.py", line 19, in <module>
    main()
  File "test.py", line 14, in main
    for i, user_input in get_user_input():
  File "test.py", line 5, in get_user_input
    yield i, input(f"In [{i}]: ")
EOFError
```
And if you press `Ctrl + c` 
```bash
In [0]: ^CTraceback (most recent call last):
  File "test.py", line 19, in <module>
    main()
  File "test.py", line 14, in main
    for i, user_input in get_user_input():
  File "test.py", line 5, in get_user_input
    yield i, input(f"In [{i}]: ")
KeyboardInterrupt
```
We don't want our interpreter to quit when we presee `Ctrl + c` we only want it to quit on `Ctrl + d`. So lets try and catch these exceptions and make our `ipython` tool to behave better.

```python
from itertools import count

def get_user_input():
    
    for i in count():
        try:
            yield i, input(f"In [{i}]: ")
        except KeyboardInterrupt:
            pass
        except EOFError:
            break
```

This would now mean that pressing `Ctrl + c` won't break the program and you would have to press `Ctrl + d` to get out. Cool!

It will not be cool, if you write valid python code in the tool and it won't do anything, right? Lets add the functionality to execute the code we write, but before that, we need to understand some concepts in python.

In Python, we have **Statements** and **Expressions**. 

> An Expression in python is something that evalutates and produces some result. Example 10 + 10 -> 20 or a function that returns something. So expression produce at least one value.

> An Statement on the other end, are everything that make up a line, or that does something example, for loops, assignments. 

Now lets continue building our interactive python interpreter. 

The first question comes to mind is how to differentiate between an expression and statement, because they both are executed differently in python. 

Python uses `exec` to execute a python statement, while `eval` is used to execute a python expression. 

Its pretty stright forward to differentiate between them. Python's `compile` function will compile python's expressions with `eval` which we know for sure that works only for expressions and not for statements (one can go the other way around as well), and would throw an `SyntaxError` for statemates. 

```python
In [1]: compile("for i in range(10): print(i)", "<stdin>", "eval")                                           
Traceback (most recent call last):

  File "/home/user/miniconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3343, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)

  File "<ipython-input-1-3dcd780a8875>", line 1, in <module>
    compile("for i in range(10): print(i)", "<stdin>", "eval")

  File "<stdin>", line 1
    for i in range(10): print(i)
      ^
SyntaxError: invalid syntax
```
So now we just try to compile a user input and if it throws an error we know for sure that it is an expression and not a statement and then we can return an `eval` or `exec` based on that. 

```python
def exec_function(unser_input):
    
    try:
        compile(user_input, "<stdin>", "eval")
    except SyntaxError:
        return exec
    return eval
```

Ok so now we have a way to evaluate which type of input the user has provided. We can build it from here, once we apply this function to the user input, we need to also take care of the error the execution of the user code would possibly have. 

```python
from itertools import count

def get_user_input():
    
    for i in count():
        try:
            yield i, input(f"In [{i}]: ")
        except KeyboardInterrupt:
            pass
        except EOFError:
            break


def exec_function(user_input):
    
    try:
        compile(user_input, "<stdin>", "eval")
    except SyntaxError:
        return exec
    return eval


def main():
    
    for i, user_input in get_user_input():
        return_value = exec_function(user_input)(user_input)
        if return_value:
            print(f"Out[{i}]: {return_value}")

if __name__ == "__main__":
    main()
```

You see in the `main` function that when we call the `exec_function` we pass in the `user_input` and then it returns us `exec` or `eval` based on the compile condition and then to that return value we again pass in the `user_input` to run it.

Now to catch the error that the `exec_function` will return we will put the return_value in try exect block
```python
def main():
    
    for i, user_input in get_user_input():
        try:
            return_value = exec_function(user_input)(user_input)
        except Exception as e:
            print(f"{e.__class__.__name__}, {e}")
        else:
            if return_value:
                print(f"Out[{i}]: {return_value}")

```
And now we have our own little `ipython`. The whole code can be found below.

```python
from itertools import count

def get_user_input():
    
    for i in count():
        try:
            yield i, input(f"In [{i}]: ")
        except KeyboardInterrupt:
            pass
        except EOFError:
            break


def exec_function(user_input):
    
    try:
        compile(user_input, "<stdin>", "eval")
    except SyntaxError:
        return exec
    return eval


def main():
    
    for i, user_input in get_user_input():
        return_value = exec_function(user_input)(user_input)
        if return_value:
            print(f"Out[{i}]: {return_value}")

if __name__ == "__main__":
    main()
```

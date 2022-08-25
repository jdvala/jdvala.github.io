+++
author = "Jay Vala"
title = "Pytest Environment Variables"
date = "2022-08-24"
description = "How to correctly set environment variables in pytest?"
tags = ["python", "testing", "pytest"]
imagelink = "https://upload.wikimedia.org/wikipedia/commons/b/ba/Pytest_logo.svg"
+++


### How would you?

… in my opinion 🧐

Writing test for code can quickly go out of hand if there are a huge number of conditions in the code, which in general there is. I recently came across such a situation where my code was 200 lines and the tests I wrote for it were 400 lines 🙈. I know it was too much but I have this practice where in I want to have 100% of test coverage. This practice of mine had sometimes made me pull my hair and prolonged my task for a day or two but it had ensured that there are minimum bugs in my code (at least that’s what I like to think 😜). 

I can’t post the code here but I will try to create a minimum working example. 

Consider the following application, 



![example.png](/example.png)

In this example, 

- First get which environment we are running the app in?
- Second, get the username and password for this environment
- Using this username and password generate a token
- Then using this token call the APP which does things…

Now to test this workflow, I will start writing unit tests. The first test is for `setup_environment` and then second for `get_auth_token` and the third test for `call_the_app` (please excuse my function names).

For all these functions I will mock different environment variables

```python
import os
from unittest.mock import Mock, patch

import pytest

@patch.dict(os.environ, {"WHICH_ENV": "dummy", "USERNAME": "PEPE", "PASSWORD": "PASSWORD"})
def test_setup_environment():
	# test my code here.
	...

@patch.dict(os.environ, {"WHICH_ENV": "", "USERNAME": "", "PASSWORD": ""})
def test_setup_environment_failed():
	# test the failing code here.
	...

@patch.dict(os.environ, {"WHICH_ENV": "dev", "USERNAME": "PEPE", "PASSWORD": "PASSWORD123"})
# Imagine here that we had other environment variables as well.
def test_generate_auth_token():
	# test my code here.
	...

@patch.dict(os.environ, {"WHICH_ENV": "dev", "USERNAME": "PEPE", "PASSWORD": "PASSWORD123"})
# Imagine here that we had other environment variables as well.
def test_call_the_app():
	# test my code here.
	...
```

Now as you can see that I was continuously using the same line of code to decorate all the functions which is not good as I am a firm believer of DRY (Don’t repeat yourself). This habit combined with my obsession of testing 100% of my code, had me writing 1000 lines of code for testing a few functions. There was a lot of repetition, I didn’t like it but most of the time had to give up on DRY as I had other task to finish. 

Today I had some time on my hand and wanted to see how could I refactor my current code (200 lines of code and almost 500 lines in test cases) to have less repetitions. I came across an answer on stackoverflow (which I don’t have the link for now) but the answer was to use `monkeypatch` 

```python
import os
from unittest.mock import Mock, patch

import pytest

@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
	monkeypatch.setenv("WHICH_ENV", "dummy")
	monkeypatch.setenv("USERNAME", "PEPE")
	monkeypatch.setenv("PASSWORD", "PASSWORD")

def test_setup_environment():
	# test my code here.
	...
```

But what if we need to change the environment variable, it was easy when the function was decorated with `patch.dict` and I can easily change it at one place. Well it is possible in unset them, or better set them again for just a single function,

```python
import os
from unittest.mock import Mock, patch

import pytest

@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
	monkeypatch.setenv("WHICH_ENV", "dummy")
	monkeypatch.setenv("USERNAME", "PEPE")
	monkeypatch.setenv("PASSWORD", "PASSWORD")

def test_setup_environment():
	# test my code here.
	...

def test_setup_environment_failed(monkeypatch):
	monkeypatch.setenv("WHICH_ENV", "")
	monkeypatch.setenv("USERNAME", "")
	monkeypatch.setenv("PASSWORD", "")
	# test the failing code here.
	...
```

And you can go about testing your code normally afterwards not having to worry about the environment variables.
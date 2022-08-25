+++
title = "Create and Manage Python virtual environment"
description =  "The bash way"
date = "2021-04-25"
author = "Jay Vala"
tags =  ["virtual-env", "python", "bash", "shell"]
+++

In any fast paced AI startup, developers and data scientist have to manage a lot of repositories. For me personally, I manage 3 or sometimes 4 repositories at a time and switching between virtual environemts for these repositories is pain according to me, writing `source deactivate` to first deactivate the exisiting environment and then activating it by `cd`ing into the repo and `source env/bin/activate`.

To avoid this I wrote a function that will ease this pain.

For this to work, lets create a directory to manage and store all your virtual environments. I like to keep them separate.

```bash
mkdir ~/environments
```

Once this is done, paste the below code into your `rc` file. For linux users it will be `.bashrc`

```bash
create_venv(){
        # First deactivate any existing environment
        deactivate
        
        # check if folder exist
        if test -d ~/environments/"$1"; then
                echo "Environment already present, activating it"
                source ~/environments/"$1"/bin/activate
        else
                python3 -m venv ~/environments/"$1"
                echo "Activating environment"
                echo $1
                source ~/environments/"$1"/bin/activate 
        fi
}
```

Let me explain what the above script does. First it `deactivates` any existing environemt, second it checks if the argument to the `create_env` function exists that is, if you have a repo called `example` then `create_env example` will first check if `example` exist, if it does then it will activate the environment present for it in the `~/environment` directory.
If the environment is not present it will create a python environment (python3) for the above script and then activate it.

It is a very basic script. 

Usage:

Go to any repository directory
```bash
create_env "repo/folder name"
```

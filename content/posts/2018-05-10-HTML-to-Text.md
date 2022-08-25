+++
title =  "Converting HTML to Text"
description =  "Converting the large EU summaries html corpus into human readable text files."
date = "2018-05-10"
author = "Jay Vala"
tags = ["python", "text", "text-preprocessing", "nlp", "text-cleaning", "preprocessing", "html2txt"]
+++


I have recently downloaded all the html files for German corpus text, and it take a lot of time to download those files, I did that to see all the html tags which helps me to device a strategy to to parse the text files further, same as in the post, [Parsing EU Summaries](https://jdvala.github.io/blog.io/2018-05-02/Parsing-EU-Summaries).


So the main idea here is read all the html text and convert it into respective text file to further preprocess it.


```python
import html2text
import codecs
import os
```


```python
def html_to_text(file_path):
    """Retruns list of lines for a given file"""
    
    # read the html file
    
    f=codecs.open(file_path, 'r')
    contents = f.read()
    
    # Create html parser object
    h = html2text.HTML2Text()   # Html parser object
    h.ignore_links = True       # Ignoring all the links if there are any
    
    a =h.handle(contents)  # parsing the contents
    a =a.strip('\n\n')     # remove unwanted new line charaters att start and end of the file
    
    return a.split('\n')      # return list of the sentences of the file to save

```


```python
def main():
    """Main function"""
    
    for root, dirs, files in os.walk('/home/jay/Data_DE_1/'):
        for file in files:
            if file.endswith('.html'):
                topic = root.split(os.path.sep)[-2]          # getting topic 
                subtopic = root.split(os.path.sep)[-1]       # Getting subtopic
                # Call the html function to convert the file
                to_save = html_to_text(os.path.join(root, file))
                
                # Save the file into a new dir
                
                dir_to_save = os.path.join(os.environ['HOME'],'html_to_text',topic, subtopic)
                
                try:
                    os.stat(dir_to_save)
                except:
                    os.makedirs(dir_to_save)
                    
                    
                print("Saving the text file to: {}".format(dir_to_save))
                
                with open(dir_to_save+'/'+file+'.txt', 'w') as f:
                    f.write('\n'.join(to_save))
                    
```


```python
if __name__ == "__main__":
    main()
```

    Saving the text file to: /home/jay/html_to_text/transport/Mobilityandpassengerrights
    Saving the text file to: /home/jay/html_to_text/transport/Mobilityandpassengerrights
    Saving the text file to: /home/jay/html_to_text/transport/Mobilityandpassengerrights
    Saving the text file to: /home/jay/html_to_text/transport/Mobilityandpassengerrights
    Saving the text file to: /home/jay/html_to_text/transport/Mobilityandpassengerrights
    ....

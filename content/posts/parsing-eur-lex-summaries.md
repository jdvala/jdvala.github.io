+++
author = "Jay Vala"
title = "Parsing EU Summaries"
date = "2018-05-02"
description = "The difficulties in parsing EU summaries"
tags = ["python", "preprocessing-text", "EU-LEX-Summaries"]
imagelink = "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80"
+++


## Notes about findings in latest parsed documents

### Things to look out for when parsing the data

*  WHEN DOES THE DIRECTIVE ENTER INTO FORCE?
*  WHEN DOES THE STRATEGY APPLY?
*  WHEN DO THE RULES OF PROCEDURE APPLY?
*  WHEN DOES THE CONVENTION APPLY?
*  WHEN DOES THE FRAMEWORK DECISION APPLY?
*  WHEN DO THE REGULATION AND DECISION APPLY?
*  WHEN DOES THE AGREEMENT AND PROTOCOL APPLY
*  DATE OF ENTRY INTO FORCE
*  WHEN DOES THE AGREEMENT APPLY  
*  WHEN DOES THIS ACT APPLY  
*  WHEN DO THE CONVENTION AND THE DECISION APPLY  
*  WHEN DOES THE COMMUNICATION APPLY  
*  WHEN DOES THE DECISION APPLY  
*  WHEN DO THE DECISIONS APPLY  
*  WHEN DO THE DECISION AND THE CONVENTION APPLY  
*  WHEN DO THE DECISION AND THE PROTOCOL APPLY  
*  WHEN DOES THE DIRECTIVE APPLY  
*  WHEN DOES THIS DIRECTIVE APPLY  
*  WHEN DOES THE PARTNERSHIP AGREEMENT APPLY  
*  WHEN DOES THE RECOMMENDATION APPLY  
*  WHEN DOES THE REGULATION APPLY  
*  WHEN DOES THIS REGULATION APPLY  
*  WHEN DO THE REGULATIONS APPLY  
*  WHEN DOES THIS REGULATION APPLY  
*  WHEN DO THE RULES APPLY?  
*  WHEN DID THE TREATY APPLY  
*  WHEN DID THE TREATY APPLY  
*  WHEN DOES THE DECISION APPLY  
*  WHEN DOES THIS DIRECTIVE APPLY  
*  FROM WHEN DO THE DECISION AND THE CONVENTION APPLY  
*  RELATED ACT  
*  RELATED DOCUMENTS  
*  RELATED INSTRUMENTS  

#### REMOVE ALL THE ABOVE FROM THE DOCUMENTS IN ORDER TO GET ONLY SUMMARY

Half of these things are removed so far but some of them are still left.

So I used ```grep``` program of linux to see if I have succesfully removed everything from the documets and make sure these are in prefect condition for preprocessing.

**```grep``` Outputs:**


* code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DOES THE DIRECTIVE ENTER INTO FORCE?'```

* output: 

```sh
/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in criminal matters/Fair trial suspects’ right to interpretation and translation in criminal proceedings.txt:113:WHEN DOES THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/last_to_use/enterprise/business environment/Combating late payment in business dealings.txt:61:FROM WHEN DOES THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/last_to_use/consumers/consumer safety/Defective products liability.txt:249:FROM WHEN DOES THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/last_to_use/education_training_youth/education and training/Lawyers practising abroad on a permanent basis — EU rules.txt:81:WHEN DOES THE DIRECTIVE ENTER INTO FORCE?
```

* code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DOES THE STRATEGY APPLY?'```

* output: 

```sh
/home/jay/Data_Pre/last_to_use/justice_freedom_security/combating drugs/EU drugs strategy.txt:133:WHEN DOES THE STRATEGY APPLY?
```

code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DO THE RULES OF PROCEDURE APPLY?'```

* output: 

```sh
/home/jay/Data_Pre/last_to_use/institutional_affairs/the institutions, bodies and agencies of the union/Rules of procedure of the Council of the European Union.txt:109:WHEN DO THE RULES OF PROCEDURE APPLY?

/home/jay/Data_Pre/last_to_use/institutional_affairs/the institutions, bodies and agencies of the union/Rules of procedure of the EU’s General Court.txt:177:FROM WHEN DO THE RULES OF PROCEDURE APPLY?
```

code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DOES THE CONVENTION APPLY?'```

* output: 

```sh
/home/jay/Data_Pre/last_to_use/transport/transport, energy and the environment/Civil liability for oil pollution damage Bunkers Convention.txt:113:FROM WHEN DOES THE CONVENTION APPLY?

/home/jay/Data_Pre/last_to_use/transport/waterborne transport/Civil liability for oil pollution damage Bunkers Convention.txt:113:FROM WHEN DOES THE CONVENTION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/police and customs cooperation/Close cooperation between EU customs administrations (Naples II Convention).txt:89:FROM WHEN DOES THE CONVENTION APPLY?
/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in civil matters/Legal certainty in international trade for EU businesses using choice of court agreements.txt:69:FROM WHEN DOES THE CONVENTION APPLY?

/home/jay/Data_Pre/last_to_use/environment/environment  cooperation with third countries/Civil liability for oil pollution damage Bunkers Convention.txt:113:FROM WHEN DOES THE CONVENTION APPLY?

/home/jay/Data_Pre/last_to_use/environment/water protection and management/Civil liability for oil pollution damage Bunkers Convention.txt:113:FROM WHEN DOES THE CONVENTION APPLY?

/home/jay/Data_Pre/last_to_use/customs/customs cooperation/Close cooperation between EU customs administrations (Naples II Convention).txt:89:FROM WHEN DOES THE CONVENTION APPLY?
```

code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DOES THE FRAMEWORK DECISION APPLY?'```

* output: 

```sh
/home/jay/Data_Pre/last_to_use/justice_freedom_security/police and customs cooperation/Simplifying the exchange of information between EU countries_ police and customs.txt:65:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/police and customs cooperation/Joint investigation teams.txt:53:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/police and customs cooperation/EU cooperation in criminal matters — personal data protection (until 2018).txt:117:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in criminal matters/Recognition and execution of confiscation orders.txt:165:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?
/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in criminal matters/Exchange of information on criminal records between EU countries.txt:125:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in criminal matters/EU mutual recognition system – prison sentences and prisoner transfers.txt:61:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in criminal matters/Mutual recognition of probation measures and alternative sanctions.txt:69:WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/judicial cooperation in criminal matters/EU cooperation in criminal matters — personal data protection (until 2018).txt:117:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?

/home/jay/Data_Pre/last_to_use/justice_freedom_security/fight against organised crime/Fight against organised crime offences linked to participation in a criminal organisation.txt:113:FROM WHEN DOES THE FRAMEWORK DECISION APPLY?
```

code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DO THE REGULATION AND DECISION APPLY?'```

* output:

``` sh
/home/jay/Data_Pre/last_to_use/economic_and_monetary_affairs/stability and growth pact and economic policy coordination/Financial assistance to Cyprus.txt:69:FROM WHEN DO THE REGULATION AND DECISION APPLY?

```


code:  ``` grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e 'WHEN DOES THE AGREEMENT AND PROTOCOL APPLY?'```

* output: 

```sh
NONE FOUND
```

So, it was getting too boring so I decided to use lazy comparisions, at least that is what official documents of regular expressions says ;)

* code: ```grep -rwn -i /home/jay/Data_Pre/last_to_use/ -e '^.*APPLY?$'```

The output of this command is too big, but I would say that its effecitve.

Now, that we know the regualr expression, what we can do is something like this.

```python
import re 
import os
import sys

# Complie the regular expression

apply = re.compile(r'^.*APPLY\?$') # We will not ignore case here because of the fact that our required lines are in uppercase

# Iterate through the files in the target folder

for root, dirs, files in os.walk('/home/jay/Data_Pre/last_to_use/'):
    for file in files:
        topic = root.split(os.path.sep)[-2]    # Topic and subtopic will be useful while saving the files
        subtopic = root.split(os.path.sep)[-1]
        # Open the file and read its contents
        wtih open(os.path.join(root, file)) as f:
            contents = f.readlines()

               # define the ending index of the file
               end_index = 0 

            # Iterate through all the lines in contents and search for the compiled regular expressions
            for index, line in enumerate(contents):
                if apply.search(line) is not None:
                    # If we found the a match, set the end index to be the index of this line 
                    end_index = index
                    break  # because we don't want to search ahead, we found a match 


            new_contents = contents[:end_index]  # We are creating new list of contents but with the end of that being end_index

            # Check where to save
            path_to_write = '~/Data_Pre/preprocessed/'+topic+'/'+subtopic
                try:
                    os.stat(path_to_write)
                except:
                    os.makedirs(path_to_write)
                print("Saving file at: {}".format(path_to_write))
                with open(path_to_write+'/'+file, "w") as file_to_write:
                    file_to_write.write(''.join(new_contents))
```

We used the same regular expression but in python ``` apply = re.compile(r'^.*APPLY\?$') # We will not ignore case here because of the fact that our required lines are in uppercase ```

Let's make sure we are successful

Now, Let's go and see if we are successful in removing the above mentioned expression

code: ```grep -nwr -i ~/Data_Pre/preprocessed/ -e '^.*APPLY?'```

output: 

```sh
/home/jay/Data_Pre/preprocessed/justice_freedom_security/judicial cooperation in civil matters/Law applicable to divorce and legal separation.txt:33:When does the regulation apply?

/home/jay/Data_Pre/preprocessed/justice_freedom_security/free movement of persons, asylum and immigration/Rules on movement of people across EU borders.txt:49:To whom does it apply?
```

We we able to remove all but two lines containing the word apply, so all in all this was a good result


Let's do same for sentences with FORCE at the end 

code:```grep -nwr -i ~/Data_Pre/preprocessed/ -e '^.*FORCE\?$'```

output:

```sh
/home/jay/Data_Pre/preprocessed/transport/air transport/Single European Sky - EU rules on air navigation services.txt:109:WHEN DOES THE REGULATION COME INTO FORCE?
/home/jay/Data_Pre/preprocessed/internal_market/single market for goods/Machinery safety.txt:129:WHEN DID THIS DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/justice_freedom_security/citizenship of the union/EU freedom of movement and residence.txt:77:WHEN DID THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/justice_freedom_security/judicial cooperation in criminal matters/Fair trial suspects’ right to interpretation and translation in criminal proceedings.txt:113:WHEN DOES THE DIRECTIVE ENTER INTO FORCE?
/home/jay/Data_Pre/preprocessed/justice_freedom_security/free movement of persons, asylum and immigration/EU freedom of movement and residence.txt:77:WHEN DID THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/institutional_affairs/the institutions, bodies and agencies of the union/The EU’s spending watchdog how the European Court of Auditors operates.txt:189:WHEN DID THE RULES OF PROCEDURE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/institutional_affairs/the decision-making process and the work of the institutions/Lobbying regulation the EU’s transparency register.txt:53:WHEN DID THE AGREEMENT ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/enterprise/business environment/Combating late payment in business dealings.txt:61:FROM WHEN DOES THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/food_safety/veterinary checks, animal health rules, food hygiene/Horses (semen, ova _ embryos) – trade within the EU.txt:97:FROM WHEN DOES THIS DIRECTIVE COME INTO FORCE?

/home/jay/Data_Pre/preprocessed/consumers/consumer safety/Defective products liability.txt:249:FROM WHEN DOES THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/consumers/consumer safety/Machinery safety.txt:129:WHEN DID THIS DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/employment_and_social_policy/health, hygiene and safety at work/Statistics on public healthhealth _ safety at work.txt:233:WHEN DID THIS REGULATION ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/education_training_youth/education and training/Lawyers practising abroad on a permanent basis — EU rules.txt:81:WHEN DOES THE DIRECTIVE ENTER INTO FORCE?

/home/jay/Data_Pre/preprocessed/education_training_youth/education and training/EU freedom of movement and residence.txt:77:WHEN DID THE DIRECTIVE ENTER INTO FORCE?
```

So, let's remove those, and  ```MAY THE FORCE BE WITH US```

```python
import re 
import os
import sys

# Complie the regular expression

apply = re.compile(r'^.*FORCE\?$') # We will not ignore case here because of the fact that our required lines are in uppercase

# Iterate through the files in the target folder

for root, dirs, files in os.walk('/home/jay/Data_Pre/last_to_use/'):
    for file in files:
        topic = root.split(os.path.sep)[-2]    # Topic and subtopic will be useful while saving the files
        subtopic = root.split(os.path.sep)[-1]
        # Open the file and read its contents
        wtih open(os.path.join(root, file)) as f:
            contents = f.readlines()

               # define the ending index of the file
               end_index = 0 

            # Iterate through all the lines in contents and search for the compiled regular expressions
            for index, line in enumerate(contents):
                if apply.search(line) is not None:
                    # If we found the a match, set the end index to be the index of this line 
                    end_index = index
                    break  # because we don't want to search ahead, we found a match 


            new_contents = contents[:end_index]  # We are creating new list of contents but with the end of that being end_index

            # Check where to save
            path_to_write = '~/Data_Pre/preprocessed/'+topic+'/'+subtopic
                try:
                    os.stat(path_to_write)
                except:
                    os.makedirs(path_to_write)
                print("Saving file at: {}".format(path_to_write))
                with open(path_to_write+'/'+file, "w") as file_to_write:
                    file_to_write.write(''.join(new_contents))
```


Let's verify that we are successful

code: ```grep -nwr -i ~/Data_Pre/preprocessed/ -e '^.*FORCE\?$'```

output:


```sh
None Found
```

We have almost parsed the documents to be used for preprocessing before we create word embeddings out of it, but there are a few more things to be taken care of

*  RELATED ACT  
*  RELATED DOCUMENTS  
*  RELATED INSTRUMENTS  

These above mentioned points are to be removed as well. So let's do the same for these.

For apply we used starting marker ```^``` and ending marker ```$``` which we will do for word ```RELATED```

code: ```grep -nwr -i ~/Data_Pre/preprocessed/ -e '^RELATED.*$'```

output:

```sh
/home/jay/Data_Pre/preprocessed/transport/waterborne transport/Mandatory checks on regular ro-ro ferries and high-speed passenger craft.txt:65:Related legislation on port state control aims to outlaw substandard shipping in the EU. It requires all vessels to comply with EU and international safety standards, as contained in the International Convention for the Safety of Life at Sea (SOLAS) for instance.
/home/jay/Data_Pre/preprocessed/transport/rail transport/An interoperable EU rail system.txt:117:related European standards and other documents allowing the bodies involved to prove their compliance with the essential requirements and TSIs.
/home/jay/Data_Pre/preprocessed/internal_market/businesses in the internal market/Copyright and related rights term of protection.txt:21:Related rights
/home/jay/Data_Pre/preprocessed/internal_market/single market for goods/Electrical equipment designed for use within certain voltage limits.txt:277:RELATED ACT
/home/jay/Data_Pre/preprocessed/internal_market/single market for services/Remuneration policies in the financial services sector.txt:121:RELATED ACT
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against organised crime/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against organised crime/European Cybercrime Centre at Europol.txt:77:RELATED ACT
/home/jay/Data_Pre/preprocessed/justice_freedom_security/free movement of persons, asylum and immigration/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against terrorism/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against terrorism/Fight against terrorism – definitions of terrorist crimes and support to victims.txt:57:Related offences
/home/jay/Data_Pre/preprocessed/agriculture/markets for agricultural products/Towards a sustainable wine sector.txt:249:RELATED ACT
/home/jay/Data_Pre/preprocessed/consumers/consumer safety/Electrical equipment designed for use within certain voltage limits.txt:277:RELATED ACT
/home/jay/Data_Pre/preprocessed/maritime_affairs_and_fisheries/maritime affairs/Maritime Policy Green Paper.txt:29:related services (because of the expertise in marine technology).
/home/jay/Data_Pre/preprocessed/environment/tackling climate change/Maritime Policy Green Paper.txt:29:related services (because of the expertise in marine technology).
/home/jay/Data_Pre/preprocessed/regional_policy/review and the future of regional policy/Cohesion policy and cities.txt:161:RELATED INSTRUMENTS
/home/jay/Data_Pre/preprocessed/regional_policy/review and the future of regional policy/Regions for economic change.txt:53:RELATED INSTRUMENTS
/home/jay/Data_Pre/preprocessed/foreign_and_security_policy/implementation of the cfsp and esdp/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/employment_and_social_policy/social and employment situation in europe/Report on employment in Europe 2004.txt:81:RELATED INSTRUMENTS
/home/jay/Data_Pre/preprocessed/information_society/digital strategy, i2010 strategy, eeurope action plan, digital strategy programmes/The EU_s new digital single market strategy.txt:133:RELATED ACT
```

We have some problem with this approach

```sh
/home/jay/Data_Pre/preprocessed/transport/waterborne transport/Mandatory checks on regular ro-ro ferries and high-speed passenger craft.txt:65:Related legislation on port state control aims to outlaw substandard shipping in the EU. It requires all vessels to comply with EU and international safety standards, as contained in the International Convention for the Safety of Life at Sea (SOLAS) for instance.
/home/jay/Data_Pre/preprocessed/transport/rail transport/An interoperable EU rail system.txt:117:related European standards and other documents allowing the bodies involved to prove their compliance with the essential requirements and TSIs.
/home/jay/Data_Pre/preprocessed/internal_market/businesses in the internal market/Copyright and related rights term of protection.txt:21:Related rights
```

See how we got some lines with apply in it, this can be a problem as we don't want to remove anything except for those tags

The problem is, when we pass the ignore case parameter we get some useful text as output which might be necessary so we will not use the ignore case parameter and run the above command again




code: ```grep -nwr ~/Data_Pre/preprocessed/ -e '^RELATED.*$'```

output:

```sh   
/home/jay/Data_Pre/preprocessed/internal_market/single market for goods/Electrical equipment designed for use within certain voltage limits.txt:277:RELATED ACT
/home/jay/Data_Pre/preprocessed/internal_market/single market for services/Remuneration policies in the financial services sector.txt:121:RELATED ACT
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against organised crime/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against organised crime/European Cybercrime Centre at Europol.txt:77:RELATED ACT
/home/jay/Data_Pre/preprocessed/justice_freedom_security/free movement of persons, asylum and immigration/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/justice_freedom_security/fight against terrorism/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/agriculture/markets for agricultural products/Towards a sustainable wine sector.txt:249:RELATED ACT
/home/jay/Data_Pre/preprocessed/consumers/consumer safety/Electrical equipment designed for use within certain voltage limits.txt:277:RELATED ACT
/home/jay/Data_Pre/preprocessed/regional_policy/review and the future of regional policy/Cohesion policy and cities.txt:161:RELATED INSTRUMENTS
/home/jay/Data_Pre/preprocessed/regional_policy/review and the future of regional policy/Regions for economic change.txt:53:RELATED INSTRUMENTS
/home/jay/Data_Pre/preprocessed/foreign_and_security_policy/implementation of the cfsp and esdp/EU internal security strategy.txt:209:RELATED DOCUMENTS
/home/jay/Data_Pre/preprocessed/employment_and_social_policy/social and employment situation in europe/Report on employment in Europe 2004.txt:81:RELATED INSTRUMENTS
/home/jay/Data_Pre/preprocessed/information_society/digital strategy, i2010 strategy, eeurope action plan, digital strategy programmes/The EU_s new digital single market strategy.txt:133:RELATED ACT
```

And we seem to have solved the problem

So, now we will use same python code from above but with change in the ```compile``` function

```python

related = re.compile(r'^RELATED.*$') # Note that we will not use ignore case flag here
```

Every thing is set and we are ready to preprocess the files!!!

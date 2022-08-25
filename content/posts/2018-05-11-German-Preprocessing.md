+++
title = "German Text Preprocessing"
description = "Preprocessing and cleaning of German text corpus."
date = "2018-05-11"
author = "Jay Vala"
tags = ["python", "text", "text-preprocessing", "nlp", "text-cleaning", "preprocessing", "html2txt", "german"]
+++

# Steps involved in preprocessing German Text

German preprocessing of text is a little different than English text.
There are special charaters with umlauts ä that should be converted first into its native form('ä' to 'ae'), this is a charater level replacement so it will need time.


```python
import os, re, sys
from nltk.corpus import stopwords
```


```python
german_stop_words = stopwords.words('german')
```


```python
german_stop_words[len(german_stop_words)-6]
```




    'würden'



As we can see that even the stop words have those umlauts so we need to convert those too


```python
def umlauts(text):
    """
    Replace umlauts for a given text
    
    :param word: text as string
    :return: manipulated text as str
    """
    
    tempVar = word # local variable
    
    # Using str.replace() 
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    
    return tempVar
```

Lets test it on German Stop words


```python
german_stop_words_to_use = []   # List to hold words after conversion

for word in german_stop_words:
    german_stop_words_to_use.append(umlauts(word))
    
```

Lets see if it worked


```python
german_stop_words_to_use[len(german_stop_words_to_use)-6]
```




    'wuerden'



It worked so lets do it on a file from German Cropus and see it it works there as well


```python
with open('/home/jay/GERMAN_CORPUS/READY_Parsed/agriculture/Agriculture/Bulgarien.html.txt', 'r') as file:
    text = file.read()
```


```python
print(' '.join((text.strip('\n').split())))
```

    In ihrer Stellungnahme vom Juli 1997 war die Europäische Kommission zu der Auffassung gelangt, dass Bulgarien mit der Angleichung seiner Rechtsvorschriften an den gemeinschaftlichen Besitzstand noch sehr wenig vorangekommen ist. Bevor Bulgarien die aus einer Mitgliedschaft erwachsenden Pflichten übernehmen könne, seien grundlegende Reformen in seiner Agrarpolitik erforderlich. Der Bericht vom November 1998 hielt gewisse Fortschritte im Hinblick auf die kurzfristigen Prioritäten der Beitrittspartnerschaft fest. Auch die Liberalisierung der Agrarpreise und die Beseitigung der Ausfuhrabgaben und nichtmengenmäßigen Ausfuhrbeschränkungen wurde vorangetrieben. In vielen Bereichen wurden jedoch weitere Anstrengungen verlangt, so unter anderem bei der Landreform. Im Bericht vom Oktober 1999 wurden zwar Fortschritte bei der Angleichung, insbesondere im Agrarsektor, festgestellt, aber die wirksame Anwendung der in nationales Recht umgesetzten Maßnahmen ist hauptsächlich auf Grund fehlender finanzieller Mittel weiterhin mit Schwierigkeiten verbunden. Es sind noch weitere Anstrengungen erforderlich, insbesondere im Bereich der Tierhaltung und des Bodenmarkts. In der Fischereipolitik hat Bulgarien seine ersten Maßnahmen erlassen. Es hat auch die meisten UN-Übereinkommen und Konventionen im Fischereisektor unterzeichnet und einige davon ratifiziert. Im Bericht vom November 2000 wurden die beträchtlichen Fortschritte dargestellt, die Bulgarien im Agrarsektor gemacht hat, vor allem im Wein- und im Getreidesektor. Jedoch gibt es nur wenig private Investitionen, gibt es noch keinen transparenten funktionierenden Grundstücksmarkt und entsprechen die Rechtsvorschriften über die gemeinsamen Marktorganisationen noch nicht der GAP. In den Bereichen Tier- und Pflanzengesundheit sind Rechtsvorschriften erlassen worden, aber die zu ihrer Durchführung eingesetzten technischen Mittel und Humanressourcen reichen noch nicht aus. Im Bereich der Fischerei wird Bulgarien noch erhebliche Anstrengungen unternehmen müssen, um den gemeinschaftlichen Besitzstand umzusetzen und die Gemeinsame Fischereipolitik anwenden zu können. Aus dem Bericht vom November 2001 geht hervor, dass Bulgarien zufrieden stellende Fortschritte bei der Rechtsangleichung erzielt hat. Die SAPARD- Stelle hat eine teilweise Akkreditierung von der Europäischen Kommission erhalten. Im Bereich der Fischerei ist mit der Verabschiedung des Fischerei- und Aquakulturgesetzes im April 2001 ein wesentlicher Fortschritt erzielt worden. Außerdem sind nicht zu vernachlässigende Verbesserungen auf institutioneller und operationeller Ebene festgestellt worden. So ist die Verwaltungskapazität im Bereich der Ressourcenbewirtschaftung, -inspektion und -kontrollen verstärkt worden und wurden gute Fortschritte bei der Schaffung des Fangflottenregisters gemacht. Im Bericht vom Oktober 2002 werden die von Bulgarien im Bereich der Rechtsangleichung und des Aufbaus des institutionellen Rahmens gemachten Fortschritte betont. Dagegen sind bei der Umsetzung der Rechtsvorschriften keine solchen Fortschritte verzeichnet worden. Im Bereich der Fischerei sind bei der Verabschiedung und Umsetzung der Gemeinsamen Fischereipolitik Bemühungen unternommen worden. Aus dem Bericht vom November 2003 geht hervor, dass Bulgarien bei den horizontalen Fragen gute Fortschritte erzielt hat, bei der Einführung der gemeinsamen Marktorganisationen jedoch in Verzug geraten ist. Bei der Entwicklung des ländlichen Raums sowie dem Veterinär- und dem Pflanzenschutzsektor sind Fortschritte gemacht worden. Im Bereich der Fischerei hat Bulgarien seit dem letzten Bericht zahlreiche Bemühungen unternommen. Dem Bericht vom Oktober 2004 zufolge waren Fortschritte bei der Angleichung an den gemeinschaftlichen Besitzstand und der Verstärkung der Verwaltungskapazitäten erzielt worden. Bulgarien hatte auch bei den horizontalen Fragen wie der Anwendung des EAGFL und der Entwicklung der gemeinsamen Marktorganisationen gute Fortschritte gemacht. Im Bericht vom Oktober 2005 wird festgestellt, dass Bulgarien die Verpflichtungen und Anforderungen erfüllt, die sich auch den Beitrittsverhandlungen hinsichtlich bestimmter horizontaler Fragen ergeben. Bei den operativen Systemen sowie im Veterinärbereich sind jedoch größere Anstrengungen notwendig. Im Bereich der Fischerei müssen bei der Bewirtschaftung der Ressourcen und dem Erlass der Rechtsvorschriften über die Marktpolitik noch Fortschritte gemacht werden. Der Beitrittsvertrag wurde am 25. April 2005 unterzeichnet und der Beitritt fand am 1. Januar 2007 statt. GEMEINSCHAFTLICHER BESITZSTAND Die Gemeinsame Agrarpolitik (GAP) zielt darauf ab, ein modernes Agrarsystem zu erhalten und zu entwickeln, der landwirtschaftlichen Bevölkerung eine angemessene Lebenshaltung zu gewährleisten, für eine Belieferung der Verbraucher zu angemessenen Preisen Sorge zu tragen und den freien Warenverkehr innerhalb der Europäischen Gemeinschaft zu verwirklichen. Das Europa-Abkommen bildet den Rechtsrahmen für den Handel mit Agrarerzeugnissen zwischen Bulgarien und der Europäischen Gemeinschaft und zielt auf eine verstärkte Zusammenarbeit bei der Modernisierung, Umstrukturierung und Privatisierung der bulgarischen Landwirtschaft und Agro- Nahrungsmittelindustrie sowie bei den Pflanzenschutznormen ab. Das Weißbuch über die Staaten Mittel- und Osteuropas und den Binnenmarkt (1995) deckt die Rechtsvorschriften in den Bereichen Veterinär-, Pflanzenschutz- und Futtermittelkontrollen sowie Bestimmungen für die Vermarktung der Erzeugnisse ab. Mit diesen Rechtsvorschriften sollen der Schutz der Verbraucher, der öffentlichen Gesundheit sowie der Tier- und Pflanzengesundheit gewährleistet werden. Die Gemeinsame Fischereipolitik umfasst die gemeinsame Marktorganisation, die Strukturpolitik, die Abkommen mit Drittländern, die Bewirtschaftung und Erhaltung der Fischereiressourcen und die wissenschaftliche Forschung auf diesem Gebiet. Das Europa-Abkommen enthält Bestimmungen über den Handel mit Fischereierzeugnissen zwischen der Gemeinschaft und Bulgarien. Das Weißbuch sieht keine Maßnahmen in diesem Sektor vor. BEWERTUNG DER LAGE Landwirtschaft Bulgarien hat die Verhandlungen über dieses Kapitel vorläufig abgeschlossen und ist in den Genuss einiger Übergangsregelungen gekommen. Im Allgemeinen hält es die Verpflichtungen ein und entspricht den Anforderungen, die sich aus den Beitrittsverhandlungen ergeben. Bei der Einrichtung der Zahlstelle ist kein Fortschritt gemacht worden: die diesbezüglichen Rechtsvorschriften müssen noch erlassen werden. Außerdem muss ein Integriertes Verwaltungs- und Kontrollsystem (INVEKOS) eingeführt werden. Die für die gemeinsamen Marktorganisationen (GMO) erforderlichen Strukturen sind noch nicht vorhanden. Bei den Rechtsvorschriften sind keine Fortschritte im Milchsektor zu verzeichnen, dagegen jedoch in den Sektoren Wein, Zucker sowie Obst und Gemüse. Im Bereich der Politik zur Entwicklung des ländlichen Raums wurde dem Land eine dreijährige Übergangszeit gewährt. Außerdem sind besondere Anstrengungen im Bereich der Rechtsangleichung im Veterinärbereich zu unternehmen, insbesondere durch den Erlass des Veterinärrahmengesetzes. Die Anstrengungen müssen dringend auf die Tierseuchenbekämpfung, den Handel mit lebenden Tieren und Erzeugnissen tierischen Ursprungs, die den Veterinärbereich betreffenden Gesundheitsvorschriften und das Wohlbefinden der Tiere konzentriert werden. Fischerei Insgesamt wird festgestellt, dass Bulgarien seinen Verpflichtungen nachkommt und den sich aus den Beitrittsverhandlungen im Bereich der staatlichen Beihilfen und der internationalen Fischereiabkommen ergebenden Anforderungen genügt. Verstärkte Anstrengungen sind jedoch in den Bereichen Bestandsbewirtschaftung und Flottenmanagement erforderlich, indem die Überwachungs- und Kontrollinstrumente konsolidiert werden, einschließlich der Einführung des Systems zur Überwachung der Schiffe. Es sind Fortschritte erforderlich, um die gemeinsamen Marktorganisationen (GMO) und die für die Verwaltung und Überwachung dieses Sektors erforderlichen Verwaltungskapazitäten zu entwickeln. Die Verhandlungen über dieses Kapitel sind vorläufig abgeschlossen und Bulgarien hat keine


So let's call the umlaut function and see if it makes any difference here


```python
new_text = umlauts(text)
```


```python
new_text = ' '.join((new_text.strip('\n').split()))
print(new_text)
```

    In ihrer Stellungnahme vom Juli 1997 war die Europaeische Kommission zu der Auffassung gelangt, dass Bulgarien mit der Angleichung seiner Rechtsvorschriften an den gemeinschaftlichen Besitzstand noch sehr wenig vorangekommen ist. Bevor Bulgarien die aus einer Mitgliedschaft erwachsenden Pflichten uebernehmen koenne, seien grundlegende Reformen in seiner Agrarpolitik erforderlich. Der Bericht vom November 1998 hielt gewisse Fortschritte im Hinblick auf die kurzfristigen Prioritaeten der Beitrittspartnerschaft fest. Auch die Liberalisierung der Agrarpreise und die Beseitigung der Ausfuhrabgaben und nichtmengenmaessigen Ausfuhrbeschraenkungen wurde vorangetrieben. In vielen Bereichen wurden jedoch weitere Anstrengungen verlangt, so unter anderem bei der Landreform. Im Bericht vom Oktober 1999 wurden zwar Fortschritte bei der Angleichung, insbesondere im Agrarsektor, festgestellt, aber die wirksame Anwendung der in nationales Recht umgesetzten Massnahmen ist hauptsaechlich auf Grund fehlender finanzieller Mittel weiterhin mit Schwierigkeiten verbunden. Es sind noch weitere Anstrengungen erforderlich, insbesondere im Bereich der Tierhaltung und des Bodenmarkts. In der Fischereipolitik hat Bulgarien seine ersten Massnahmen erlassen. Es hat auch die meisten UN-Uebereinkommen und Konventionen im Fischereisektor unterzeichnet und einige davon ratifiziert. Im Bericht vom November 2000 wurden die betraechtlichen Fortschritte dargestellt, die Bulgarien im Agrarsektor gemacht hat, vor allem im Wein- und im Getreidesektor. Jedoch gibt es nur wenig private Investitionen, gibt es noch keinen transparenten funktionierenden Grundstuecksmarkt und entsprechen die Rechtsvorschriften ueber die gemeinsamen Marktorganisationen noch nicht der GAP. In den Bereichen Tier- und Pflanzengesundheit sind Rechtsvorschriften erlassen worden, aber die zu ihrer Durchfuehrung eingesetzten technischen Mittel und Humanressourcen reichen noch nicht aus. Im Bereich der Fischerei wird Bulgarien noch erhebliche Anstrengungen unternehmen muessen, um den gemeinschaftlichen Besitzstand umzusetzen und die Gemeinsame Fischereipolitik anwenden zu koennen. Aus dem Bericht vom November 2001 geht hervor, dass Bulgarien zufrieden stellende Fortschritte bei der Rechtsangleichung erzielt hat. Die SAPARD- Stelle hat eine teilweise Akkreditierung von der Europaeischen Kommission erhalten. Im Bereich der Fischerei ist mit der Verabschiedung des Fischerei- und Aquakulturgesetzes im April 2001 ein wesentlicher Fortschritt erzielt worden. Ausserdem sind nicht zu vernachlaessigende Verbesserungen auf institutioneller und operationeller Ebene festgestellt worden. So ist die Verwaltungskapazitaet im Bereich der Ressourcenbewirtschaftung, -inspektion und -kontrollen verstaerkt worden und wurden gute Fortschritte bei der Schaffung des Fangflottenregisters gemacht. Im Bericht vom Oktober 2002 werden die von Bulgarien im Bereich der Rechtsangleichung und des Aufbaus des institutionellen Rahmens gemachten Fortschritte betont. Dagegen sind bei der Umsetzung der Rechtsvorschriften keine solchen Fortschritte verzeichnet worden. Im Bereich der Fischerei sind bei der Verabschiedung und Umsetzung der Gemeinsamen Fischereipolitik Bemuehungen unternommen worden. Aus dem Bericht vom November 2003 geht hervor, dass Bulgarien bei den horizontalen Fragen gute Fortschritte erzielt hat, bei der Einfuehrung der gemeinsamen Marktorganisationen jedoch in Verzug geraten ist. Bei der Entwicklung des laendlichen Raums sowie dem Veterinaer- und dem Pflanzenschutzsektor sind Fortschritte gemacht worden. Im Bereich der Fischerei hat Bulgarien seit dem letzten Bericht zahlreiche Bemuehungen unternommen. Dem Bericht vom Oktober 2004 zufolge waren Fortschritte bei der Angleichung an den gemeinschaftlichen Besitzstand und der Verstaerkung der Verwaltungskapazitaeten erzielt worden. Bulgarien hatte auch bei den horizontalen Fragen wie der Anwendung des EAGFL und der Entwicklung der gemeinsamen Marktorganisationen gute Fortschritte gemacht. Im Bericht vom Oktober 2005 wird festgestellt, dass Bulgarien die Verpflichtungen und Anforderungen erfuellt, die sich auch den Beitrittsverhandlungen hinsichtlich bestimmter horizontaler Fragen ergeben. Bei den operativen Systemen sowie im Veterinaerbereich sind jedoch groessere Anstrengungen notwendig. Im Bereich der Fischerei muessen bei der Bewirtschaftung der Ressourcen und dem Erlass der Rechtsvorschriften ueber die Marktpolitik noch Fortschritte gemacht werden. Der Beitrittsvertrag wurde am 25. April 2005 unterzeichnet und der Beitritt fand am 1. Januar 2007 statt. GEMEINSCHAFTLICHER BESITZSTAND Die Gemeinsame Agrarpolitik (GAP) zielt darauf ab, ein modernes Agrarsystem zu erhalten und zu entwickeln, der landwirtschaftlichen Bevoelkerung eine angemessene Lebenshaltung zu gewaehrleisten, fuer eine Belieferung der Verbraucher zu angemessenen Preisen Sorge zu tragen und den freien Warenverkehr innerhalb der Europaeischen Gemeinschaft zu verwirklichen. Das Europa-Abkommen bildet den Rechtsrahmen fuer den Handel mit Agrarerzeugnissen zwischen Bulgarien und der Europaeischen Gemeinschaft und zielt auf eine verstaerkte Zusammenarbeit bei der Modernisierung, Umstrukturierung und Privatisierung der bulgarischen Landwirtschaft und Agro- Nahrungsmittelindustrie sowie bei den Pflanzenschutznormen ab. Das Weissbuch ueber die Staaten Mittel- und Osteuropas und den Binnenmarkt (1995) deckt die Rechtsvorschriften in den Bereichen Veterinaer-, Pflanzenschutz- und Futtermittelkontrollen sowie Bestimmungen fuer die Vermarktung der Erzeugnisse ab. Mit diesen Rechtsvorschriften sollen der Schutz der Verbraucher, der oeffentlichen Gesundheit sowie der Tier- und Pflanzengesundheit gewaehrleistet werden. Die Gemeinsame Fischereipolitik umfasst die gemeinsame Marktorganisation, die Strukturpolitik, die Abkommen mit Drittlaendern, die Bewirtschaftung und Erhaltung der Fischereiressourcen und die wissenschaftliche Forschung auf diesem Gebiet. Das Europa-Abkommen enthaelt Bestimmungen ueber den Handel mit Fischereierzeugnissen zwischen der Gemeinschaft und Bulgarien. Das Weissbuch sieht keine Massnahmen in diesem Sektor vor. BEWERTUNG DER LAGE Landwirtschaft Bulgarien hat die Verhandlungen ueber dieses Kapitel vorlaeufig abgeschlossen und ist in den Genuss einiger Uebergangsregelungen gekommen. Im Allgemeinen haelt es die Verpflichtungen ein und entspricht den Anforderungen, die sich aus den Beitrittsverhandlungen ergeben. Bei der Einrichtung der Zahlstelle ist kein Fortschritt gemacht worden: die diesbezueglichen Rechtsvorschriften muessen noch erlassen werden. Ausserdem muss ein Integriertes Verwaltungs- und Kontrollsystem (INVEKOS) eingefuehrt werden. Die fuer die gemeinsamen Marktorganisationen (GMO) erforderlichen Strukturen sind noch nicht vorhanden. Bei den Rechtsvorschriften sind keine Fortschritte im Milchsektor zu verzeichnen, dagegen jedoch in den Sektoren Wein, Zucker sowie Obst und Gemuese. Im Bereich der Politik zur Entwicklung des laendlichen Raums wurde dem Land eine dreijaehrige Uebergangszeit gewaehrt. Ausserdem sind besondere Anstrengungen im Bereich der Rechtsangleichung im Veterinaerbereich zu unternehmen, insbesondere durch den Erlass des Veterinaerrahmengesetzes. Die Anstrengungen muessen dringend auf die Tierseuchenbekaempfung, den Handel mit lebenden Tieren und Erzeugnissen tierischen Ursprungs, die den Veterinaerbereich betreffenden Gesundheitsvorschriften und das Wohlbefinden der Tiere konzentriert werden. Fischerei Insgesamt wird festgestellt, dass Bulgarien seinen Verpflichtungen nachkommt und den sich aus den Beitrittsverhandlungen im Bereich der staatlichen Beihilfen und der internationalen Fischereiabkommen ergebenden Anforderungen genuegt. Verstaerkte Anstrengungen sind jedoch in den Bereichen Bestandsbewirtschaftung und Flottenmanagement erforderlich, indem die Ueberwachungs- und Kontrollinstrumente konsolidiert werden, einschliesslich der Einfuehrung des Systems zur Ueberwachung der Schiffe. Es sind Fortschritte erforderlich, um die gemeinsamen Marktorganisationen (GMO) und die fuer die Verwaltung und Ueberwachung dieses Sektors erforderlichen Verwaltungskapazitaeten zu entwickeln. Die Verhandlungen ueber dieses Kapitel sind vorlaeufig abgeschlossen und Bulgarien hat keine


And it did, so we know know that it works and its ok.
Now we need to remove all the punctuations and numbers from the text. Lets do that


```python
from string import punctuation
```


```python
punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'



It looks like it covers all the punctuation, Lets remove them


```python
remove_pun = str.maketrans('', '', punctuation)
text_wo_pun = new_text.translate(remove_pun) # we are using native string functions as they are fast
```


```python
text_wo_pun
```




    'In ihrer Stellungnahme vom Juli 1997 war die Europaeische Kommission zu der Auffassung gelangt dass Bulgarien mit der Angleichung seiner Rechtsvorschriften an den gemeinschaftlichen Besitzstand noch sehr wenig vorangekommen ist Bevor Bulgarien die aus einer Mitgliedschaft erwachsenden Pflichten uebernehmen koenne seien grundlegende Reformen in seiner Agrarpolitik erforderlich Der Bericht vom November 1998 hielt gewisse Fortschritte im Hinblick auf die kurzfristigen Prioritaeten der Beitrittspartnerschaft fest Auch die Liberalisierung der Agrarpreise und die Beseitigung der Ausfuhrabgaben und nichtmengenmaessigen Ausfuhrbeschraenkungen wurde vorangetrieben In vielen Bereichen wurden jedoch weitere Anstrengungen verlangt so unter anderem bei der Landreform Im Bericht vom Oktober 1999 wurden zwar Fortschritte bei der Angleichung insbesondere im Agrarsektor festgestellt aber die wirksame Anwendung der in nationales Recht umgesetzten Massnahmen ist hauptsaechlich auf Grund fehlender finanzieller Mittel weiterhin mit Schwierigkeiten verbunden Es sind noch weitere Anstrengungen erforderlich insbesondere im Bereich der Tierhaltung und des Bodenmarkts In der Fischereipolitik hat Bulgarien seine ersten Massnahmen erlassen Es hat auch die meisten UNUebereinkommen und Konventionen im Fischereisektor unterzeichnet und einige davon ratifiziert Im Bericht vom November 2000 wurden die betraechtlichen Fortschritte dargestellt die Bulgarien im Agrarsektor gemacht hat vor allem im Wein und im Getreidesektor Jedoch gibt es nur wenig private Investitionen gibt es noch keinen transparenten funktionierenden Grundstuecksmarkt und entsprechen die Rechtsvorschriften ueber die gemeinsamen Marktorganisationen noch nicht der GAP In den Bereichen Tier und Pflanzengesundheit sind Rechtsvorschriften erlassen worden aber die zu ihrer Durchfuehrung eingesetzten technischen Mittel und Humanressourcen reichen noch nicht aus Im Bereich der Fischerei wird Bulgarien noch erhebliche Anstrengungen unternehmen muessen um den gemeinschaftlichen Besitzstand umzusetzen und die Gemeinsame Fischereipolitik anwenden zu koennen Aus dem Bericht vom November 2001 geht hervor dass Bulgarien zufrieden stellende Fortschritte bei der Rechtsangleichung erzielt hat Die SAPARD Stelle hat eine teilweise Akkreditierung von der Europaeischen Kommission erhalten Im Bereich der Fischerei ist mit der Verabschiedung des Fischerei und Aquakulturgesetzes im April 2001 ein wesentlicher Fortschritt erzielt worden Ausserdem sind nicht zu vernachlaessigende Verbesserungen auf institutioneller und operationeller Ebene festgestellt worden So ist die Verwaltungskapazitaet im Bereich der Ressourcenbewirtschaftung inspektion und kontrollen verstaerkt worden und wurden gute Fortschritte bei der Schaffung des Fangflottenregisters gemacht Im Bericht vom Oktober 2002 werden die von Bulgarien im Bereich der Rechtsangleichung und des Aufbaus des institutionellen Rahmens gemachten Fortschritte betont Dagegen sind bei der Umsetzung der Rechtsvorschriften keine solchen Fortschritte verzeichnet worden Im Bereich der Fischerei sind bei der Verabschiedung und Umsetzung der Gemeinsamen Fischereipolitik Bemuehungen unternommen worden Aus dem Bericht vom November 2003 geht hervor dass Bulgarien bei den horizontalen Fragen gute Fortschritte erzielt hat bei der Einfuehrung der gemeinsamen Marktorganisationen jedoch in Verzug geraten ist Bei der Entwicklung des laendlichen Raums sowie dem Veterinaer und dem Pflanzenschutzsektor sind Fortschritte gemacht worden Im Bereich der Fischerei hat Bulgarien seit dem letzten Bericht zahlreiche Bemuehungen unternommen Dem Bericht vom Oktober 2004 zufolge waren Fortschritte bei der Angleichung an den gemeinschaftlichen Besitzstand und der Verstaerkung der Verwaltungskapazitaeten erzielt worden Bulgarien hatte auch bei den horizontalen Fragen wie der Anwendung des EAGFL und der Entwicklung der gemeinsamen Marktorganisationen gute Fortschritte gemacht Im Bericht vom Oktober 2005 wird festgestellt dass Bulgarien die Verpflichtungen und Anforderungen erfuellt die sich auch den Beitrittsverhandlungen hinsichtlich bestimmter horizontaler Fragen ergeben Bei den operativen Systemen sowie im Veterinaerbereich sind jedoch groessere Anstrengungen notwendig Im Bereich der Fischerei muessen bei der Bewirtschaftung der Ressourcen und dem Erlass der Rechtsvorschriften ueber die Marktpolitik noch Fortschritte gemacht werden Der Beitrittsvertrag wurde am 25 April 2005 unterzeichnet und der Beitritt fand am 1 Januar 2007 statt GEMEINSCHAFTLICHER BESITZSTAND Die Gemeinsame Agrarpolitik GAP zielt darauf ab ein modernes Agrarsystem zu erhalten und zu entwickeln der landwirtschaftlichen Bevoelkerung eine angemessene Lebenshaltung zu gewaehrleisten fuer eine Belieferung der Verbraucher zu angemessenen Preisen Sorge zu tragen und den freien Warenverkehr innerhalb der Europaeischen Gemeinschaft zu verwirklichen Das EuropaAbkommen bildet den Rechtsrahmen fuer den Handel mit Agrarerzeugnissen zwischen Bulgarien und der Europaeischen Gemeinschaft und zielt auf eine verstaerkte Zusammenarbeit bei der Modernisierung Umstrukturierung und Privatisierung der bulgarischen Landwirtschaft und Agro Nahrungsmittelindustrie sowie bei den Pflanzenschutznormen ab Das Weissbuch ueber die Staaten Mittel und Osteuropas und den Binnenmarkt 1995 deckt die Rechtsvorschriften in den Bereichen Veterinaer Pflanzenschutz und Futtermittelkontrollen sowie Bestimmungen fuer die Vermarktung der Erzeugnisse ab Mit diesen Rechtsvorschriften sollen der Schutz der Verbraucher der oeffentlichen Gesundheit sowie der Tier und Pflanzengesundheit gewaehrleistet werden Die Gemeinsame Fischereipolitik umfasst die gemeinsame Marktorganisation die Strukturpolitik die Abkommen mit Drittlaendern die Bewirtschaftung und Erhaltung der Fischereiressourcen und die wissenschaftliche Forschung auf diesem Gebiet Das EuropaAbkommen enthaelt Bestimmungen ueber den Handel mit Fischereierzeugnissen zwischen der Gemeinschaft und Bulgarien Das Weissbuch sieht keine Massnahmen in diesem Sektor vor BEWERTUNG DER LAGE Landwirtschaft Bulgarien hat die Verhandlungen ueber dieses Kapitel vorlaeufig abgeschlossen und ist in den Genuss einiger Uebergangsregelungen gekommen Im Allgemeinen haelt es die Verpflichtungen ein und entspricht den Anforderungen die sich aus den Beitrittsverhandlungen ergeben Bei der Einrichtung der Zahlstelle ist kein Fortschritt gemacht worden die diesbezueglichen Rechtsvorschriften muessen noch erlassen werden Ausserdem muss ein Integriertes Verwaltungs und Kontrollsystem INVEKOS eingefuehrt werden Die fuer die gemeinsamen Marktorganisationen GMO erforderlichen Strukturen sind noch nicht vorhanden Bei den Rechtsvorschriften sind keine Fortschritte im Milchsektor zu verzeichnen dagegen jedoch in den Sektoren Wein Zucker sowie Obst und Gemuese Im Bereich der Politik zur Entwicklung des laendlichen Raums wurde dem Land eine dreijaehrige Uebergangszeit gewaehrt Ausserdem sind besondere Anstrengungen im Bereich der Rechtsangleichung im Veterinaerbereich zu unternehmen insbesondere durch den Erlass des Veterinaerrahmengesetzes Die Anstrengungen muessen dringend auf die Tierseuchenbekaempfung den Handel mit lebenden Tieren und Erzeugnissen tierischen Ursprungs die den Veterinaerbereich betreffenden Gesundheitsvorschriften und das Wohlbefinden der Tiere konzentriert werden Fischerei Insgesamt wird festgestellt dass Bulgarien seinen Verpflichtungen nachkommt und den sich aus den Beitrittsverhandlungen im Bereich der staatlichen Beihilfen und der internationalen Fischereiabkommen ergebenden Anforderungen genuegt Verstaerkte Anstrengungen sind jedoch in den Bereichen Bestandsbewirtschaftung und Flottenmanagement erforderlich indem die Ueberwachungs und Kontrollinstrumente konsolidiert werden einschliesslich der Einfuehrung des Systems zur Ueberwachung der Schiffe Es sind Fortschritte erforderlich um die gemeinsamen Marktorganisationen GMO und die fuer die Verwaltung und Ueberwachung dieses Sektors erforderlichen Verwaltungskapazitaeten zu entwickeln Die Verhandlungen ueber dieses Kapitel sind vorlaeufig abgeschlossen und Bulgarien hat keine'



As we can see that all the punctuation have been removed lets go on to remove numbers


```python
from string import digits

remove_digits = str.maketrans('', '', digits)
 
```

Again we are using string function to do all the heavy lifting


```python
text_wo_num = text_wo_pun.translate(remove_digits)
```


```python
text_wo_num
```




    'In ihrer Stellungnahme vom Juli  war die Europaeische Kommission zu der Auffassung gelangt dass Bulgarien mit der Angleichung seiner Rechtsvorschriften an den gemeinschaftlichen Besitzstand noch sehr wenig vorangekommen ist Bevor Bulgarien die aus einer Mitgliedschaft erwachsenden Pflichten uebernehmen koenne seien grundlegende Reformen in seiner Agrarpolitik erforderlich Der Bericht vom November  hielt gewisse Fortschritte im Hinblick auf die kurzfristigen Prioritaeten der Beitrittspartnerschaft fest Auch die Liberalisierung der Agrarpreise und die Beseitigung der Ausfuhrabgaben und nichtmengenmaessigen Ausfuhrbeschraenkungen wurde vorangetrieben In vielen Bereichen wurden jedoch weitere Anstrengungen verlangt so unter anderem bei der Landreform Im Bericht vom Oktober  wurden zwar Fortschritte bei der Angleichung insbesondere im Agrarsektor festgestellt aber die wirksame Anwendung der in nationales Recht umgesetzten Massnahmen ist hauptsaechlich auf Grund fehlender finanzieller Mittel weiterhin mit Schwierigkeiten verbunden Es sind noch weitere Anstrengungen erforderlich insbesondere im Bereich der Tierhaltung und des Bodenmarkts In der Fischereipolitik hat Bulgarien seine ersten Massnahmen erlassen Es hat auch die meisten UNUebereinkommen und Konventionen im Fischereisektor unterzeichnet und einige davon ratifiziert Im Bericht vom November  wurden die betraechtlichen Fortschritte dargestellt die Bulgarien im Agrarsektor gemacht hat vor allem im Wein und im Getreidesektor Jedoch gibt es nur wenig private Investitionen gibt es noch keinen transparenten funktionierenden Grundstuecksmarkt und entsprechen die Rechtsvorschriften ueber die gemeinsamen Marktorganisationen noch nicht der GAP In den Bereichen Tier und Pflanzengesundheit sind Rechtsvorschriften erlassen worden aber die zu ihrer Durchfuehrung eingesetzten technischen Mittel und Humanressourcen reichen noch nicht aus Im Bereich der Fischerei wird Bulgarien noch erhebliche Anstrengungen unternehmen muessen um den gemeinschaftlichen Besitzstand umzusetzen und die Gemeinsame Fischereipolitik anwenden zu koennen Aus dem Bericht vom November  geht hervor dass Bulgarien zufrieden stellende Fortschritte bei der Rechtsangleichung erzielt hat Die SAPARD Stelle hat eine teilweise Akkreditierung von der Europaeischen Kommission erhalten Im Bereich der Fischerei ist mit der Verabschiedung des Fischerei und Aquakulturgesetzes im April  ein wesentlicher Fortschritt erzielt worden Ausserdem sind nicht zu vernachlaessigende Verbesserungen auf institutioneller und operationeller Ebene festgestellt worden So ist die Verwaltungskapazitaet im Bereich der Ressourcenbewirtschaftung inspektion und kontrollen verstaerkt worden und wurden gute Fortschritte bei der Schaffung des Fangflottenregisters gemacht Im Bericht vom Oktober  werden die von Bulgarien im Bereich der Rechtsangleichung und des Aufbaus des institutionellen Rahmens gemachten Fortschritte betont Dagegen sind bei der Umsetzung der Rechtsvorschriften keine solchen Fortschritte verzeichnet worden Im Bereich der Fischerei sind bei der Verabschiedung und Umsetzung der Gemeinsamen Fischereipolitik Bemuehungen unternommen worden Aus dem Bericht vom November  geht hervor dass Bulgarien bei den horizontalen Fragen gute Fortschritte erzielt hat bei der Einfuehrung der gemeinsamen Marktorganisationen jedoch in Verzug geraten ist Bei der Entwicklung des laendlichen Raums sowie dem Veterinaer und dem Pflanzenschutzsektor sind Fortschritte gemacht worden Im Bereich der Fischerei hat Bulgarien seit dem letzten Bericht zahlreiche Bemuehungen unternommen Dem Bericht vom Oktober  zufolge waren Fortschritte bei der Angleichung an den gemeinschaftlichen Besitzstand und der Verstaerkung der Verwaltungskapazitaeten erzielt worden Bulgarien hatte auch bei den horizontalen Fragen wie der Anwendung des EAGFL und der Entwicklung der gemeinsamen Marktorganisationen gute Fortschritte gemacht Im Bericht vom Oktober  wird festgestellt dass Bulgarien die Verpflichtungen und Anforderungen erfuellt die sich auch den Beitrittsverhandlungen hinsichtlich bestimmter horizontaler Fragen ergeben Bei den operativen Systemen sowie im Veterinaerbereich sind jedoch groessere Anstrengungen notwendig Im Bereich der Fischerei muessen bei der Bewirtschaftung der Ressourcen und dem Erlass der Rechtsvorschriften ueber die Marktpolitik noch Fortschritte gemacht werden Der Beitrittsvertrag wurde am  April  unterzeichnet und der Beitritt fand am  Januar  statt GEMEINSCHAFTLICHER BESITZSTAND Die Gemeinsame Agrarpolitik GAP zielt darauf ab ein modernes Agrarsystem zu erhalten und zu entwickeln der landwirtschaftlichen Bevoelkerung eine angemessene Lebenshaltung zu gewaehrleisten fuer eine Belieferung der Verbraucher zu angemessenen Preisen Sorge zu tragen und den freien Warenverkehr innerhalb der Europaeischen Gemeinschaft zu verwirklichen Das EuropaAbkommen bildet den Rechtsrahmen fuer den Handel mit Agrarerzeugnissen zwischen Bulgarien und der Europaeischen Gemeinschaft und zielt auf eine verstaerkte Zusammenarbeit bei der Modernisierung Umstrukturierung und Privatisierung der bulgarischen Landwirtschaft und Agro Nahrungsmittelindustrie sowie bei den Pflanzenschutznormen ab Das Weissbuch ueber die Staaten Mittel und Osteuropas und den Binnenmarkt  deckt die Rechtsvorschriften in den Bereichen Veterinaer Pflanzenschutz und Futtermittelkontrollen sowie Bestimmungen fuer die Vermarktung der Erzeugnisse ab Mit diesen Rechtsvorschriften sollen der Schutz der Verbraucher der oeffentlichen Gesundheit sowie der Tier und Pflanzengesundheit gewaehrleistet werden Die Gemeinsame Fischereipolitik umfasst die gemeinsame Marktorganisation die Strukturpolitik die Abkommen mit Drittlaendern die Bewirtschaftung und Erhaltung der Fischereiressourcen und die wissenschaftliche Forschung auf diesem Gebiet Das EuropaAbkommen enthaelt Bestimmungen ueber den Handel mit Fischereierzeugnissen zwischen der Gemeinschaft und Bulgarien Das Weissbuch sieht keine Massnahmen in diesem Sektor vor BEWERTUNG DER LAGE Landwirtschaft Bulgarien hat die Verhandlungen ueber dieses Kapitel vorlaeufig abgeschlossen und ist in den Genuss einiger Uebergangsregelungen gekommen Im Allgemeinen haelt es die Verpflichtungen ein und entspricht den Anforderungen die sich aus den Beitrittsverhandlungen ergeben Bei der Einrichtung der Zahlstelle ist kein Fortschritt gemacht worden die diesbezueglichen Rechtsvorschriften muessen noch erlassen werden Ausserdem muss ein Integriertes Verwaltungs und Kontrollsystem INVEKOS eingefuehrt werden Die fuer die gemeinsamen Marktorganisationen GMO erforderlichen Strukturen sind noch nicht vorhanden Bei den Rechtsvorschriften sind keine Fortschritte im Milchsektor zu verzeichnen dagegen jedoch in den Sektoren Wein Zucker sowie Obst und Gemuese Im Bereich der Politik zur Entwicklung des laendlichen Raums wurde dem Land eine dreijaehrige Uebergangszeit gewaehrt Ausserdem sind besondere Anstrengungen im Bereich der Rechtsangleichung im Veterinaerbereich zu unternehmen insbesondere durch den Erlass des Veterinaerrahmengesetzes Die Anstrengungen muessen dringend auf die Tierseuchenbekaempfung den Handel mit lebenden Tieren und Erzeugnissen tierischen Ursprungs die den Veterinaerbereich betreffenden Gesundheitsvorschriften und das Wohlbefinden der Tiere konzentriert werden Fischerei Insgesamt wird festgestellt dass Bulgarien seinen Verpflichtungen nachkommt und den sich aus den Beitrittsverhandlungen im Bereich der staatlichen Beihilfen und der internationalen Fischereiabkommen ergebenden Anforderungen genuegt Verstaerkte Anstrengungen sind jedoch in den Bereichen Bestandsbewirtschaftung und Flottenmanagement erforderlich indem die Ueberwachungs und Kontrollinstrumente konsolidiert werden einschliesslich der Einfuehrung des Systems zur Ueberwachung der Schiffe Es sind Fortschritte erforderlich um die gemeinsamen Marktorganisationen GMO und die fuer die Verwaltung und Ueberwachung dieses Sektors erforderlichen Verwaltungskapazitaeten zu entwickeln Die Verhandlungen ueber dieses Kapitel sind vorlaeufig abgeschlossen und Bulgarien hat keine'



Now lets remove german stop words from the text, So here we can not use string function, but we are going to use list comprihensions which are as fast as string functions


```python
text_wo_stop_words = [word for word in text_wo_num.split() if word.lower() not in german_stop_words_to_use]
```

Now lets see if it removed the stop words or not


```python
text_wo_stop_words = ' '.join(text_wo_stop_words)
```


```python
text_wo_stop_words
```




    'Stellungnahme Juli Europaeische Kommission Auffassung gelangt Bulgarien Angleichung Rechtsvorschriften gemeinschaftlichen Besitzstand wenig vorangekommen Bevor Bulgarien Mitgliedschaft erwachsenden Pflichten uebernehmen koenne seien grundlegende Reformen Agrarpolitik erforderlich Bericht November hielt gewisse Fortschritte Hinblick kurzfristigen Prioritaeten Beitrittspartnerschaft fest Liberalisierung Agrarpreise Beseitigung Ausfuhrabgaben nichtmengenmaessigen Ausfuhrbeschraenkungen wurde vorangetrieben vielen Bereichen wurden jedoch weitere Anstrengungen verlangt Landreform Bericht Oktober wurden Fortschritte Angleichung insbesondere Agrarsektor festgestellt wirksame Anwendung nationales Recht umgesetzten Massnahmen hauptsaechlich Grund fehlender finanzieller Mittel weiterhin Schwierigkeiten verbunden weitere Anstrengungen erforderlich insbesondere Bereich Tierhaltung Bodenmarkts Fischereipolitik Bulgarien ersten Massnahmen erlassen meisten UNUebereinkommen Konventionen Fischereisektor unterzeichnet davon ratifiziert Bericht November wurden betraechtlichen Fortschritte dargestellt Bulgarien Agrarsektor gemacht Wein Getreidesektor Jedoch gibt wenig private Investitionen gibt transparenten funktionierenden Grundstuecksmarkt entsprechen Rechtsvorschriften gemeinsamen Marktorganisationen GAP Bereichen Tier Pflanzengesundheit Rechtsvorschriften erlassen worden Durchfuehrung eingesetzten technischen Mittel Humanressourcen reichen Bereich Fischerei Bulgarien erhebliche Anstrengungen unternehmen muessen gemeinschaftlichen Besitzstand umzusetzen Gemeinsame Fischereipolitik anwenden Bericht November geht hervor Bulgarien zufrieden stellende Fortschritte Rechtsangleichung erzielt SAPARD Stelle teilweise Akkreditierung Europaeischen Kommission erhalten Bereich Fischerei Verabschiedung Fischerei Aquakulturgesetzes April wesentlicher Fortschritt erzielt worden Ausserdem vernachlaessigende Verbesserungen institutioneller operationeller Ebene festgestellt worden Verwaltungskapazitaet Bereich Ressourcenbewirtschaftung inspektion kontrollen verstaerkt worden wurden gute Fortschritte Schaffung Fangflottenregisters gemacht Bericht Oktober Bulgarien Bereich Rechtsangleichung Aufbaus institutionellen Rahmens gemachten Fortschritte betont Dagegen Umsetzung Rechtsvorschriften Fortschritte verzeichnet worden Bereich Fischerei Verabschiedung Umsetzung Gemeinsamen Fischereipolitik Bemuehungen unternommen worden Bericht November geht hervor Bulgarien horizontalen Fragen gute Fortschritte erzielt Einfuehrung gemeinsamen Marktorganisationen jedoch Verzug geraten Entwicklung laendlichen Raums sowie Veterinaer Pflanzenschutzsektor Fortschritte gemacht worden Bereich Fischerei Bulgarien seit letzten Bericht zahlreiche Bemuehungen unternommen Bericht Oktober zufolge Fortschritte Angleichung gemeinschaftlichen Besitzstand Verstaerkung Verwaltungskapazitaeten erzielt worden Bulgarien horizontalen Fragen Anwendung EAGFL Entwicklung gemeinsamen Marktorganisationen gute Fortschritte gemacht Bericht Oktober festgestellt Bulgarien Verpflichtungen Anforderungen erfuellt Beitrittsverhandlungen hinsichtlich bestimmter horizontaler Fragen ergeben operativen Systemen sowie Veterinaerbereich jedoch groessere Anstrengungen notwendig Bereich Fischerei muessen Bewirtschaftung Ressourcen Erlass Rechtsvorschriften Marktpolitik Fortschritte gemacht Beitrittsvertrag wurde April unterzeichnet Beitritt fand Januar statt GEMEINSCHAFTLICHER BESITZSTAND Gemeinsame Agrarpolitik GAP zielt darauf ab modernes Agrarsystem erhalten entwickeln landwirtschaftlichen Bevoelkerung angemessene Lebenshaltung gewaehrleisten Belieferung Verbraucher angemessenen Preisen Sorge tragen freien Warenverkehr innerhalb Europaeischen Gemeinschaft verwirklichen EuropaAbkommen bildet Rechtsrahmen Handel Agrarerzeugnissen Bulgarien Europaeischen Gemeinschaft zielt verstaerkte Zusammenarbeit Modernisierung Umstrukturierung Privatisierung bulgarischen Landwirtschaft Agro Nahrungsmittelindustrie sowie Pflanzenschutznormen ab Weissbuch Staaten Mittel Osteuropas Binnenmarkt deckt Rechtsvorschriften Bereichen Veterinaer Pflanzenschutz Futtermittelkontrollen sowie Bestimmungen Vermarktung Erzeugnisse ab Rechtsvorschriften sollen Schutz Verbraucher oeffentlichen Gesundheit sowie Tier Pflanzengesundheit gewaehrleistet Gemeinsame Fischereipolitik umfasst gemeinsame Marktorganisation Strukturpolitik Abkommen Drittlaendern Bewirtschaftung Erhaltung Fischereiressourcen wissenschaftliche Forschung Gebiet EuropaAbkommen enthaelt Bestimmungen Handel Fischereierzeugnissen Gemeinschaft Bulgarien Weissbuch sieht Massnahmen Sektor BEWERTUNG LAGE Landwirtschaft Bulgarien Verhandlungen Kapitel vorlaeufig abgeschlossen Genuss Uebergangsregelungen gekommen Allgemeinen haelt Verpflichtungen entspricht Anforderungen Beitrittsverhandlungen ergeben Einrichtung Zahlstelle Fortschritt gemacht worden diesbezueglichen Rechtsvorschriften muessen erlassen Ausserdem Integriertes Verwaltungs Kontrollsystem INVEKOS eingefuehrt gemeinsamen Marktorganisationen GMO erforderlichen Strukturen vorhanden Rechtsvorschriften Fortschritte Milchsektor verzeichnen dagegen jedoch Sektoren Wein Zucker sowie Obst Gemuese Bereich Politik Entwicklung laendlichen Raums wurde Land dreijaehrige Uebergangszeit gewaehrt Ausserdem besondere Anstrengungen Bereich Rechtsangleichung Veterinaerbereich unternehmen insbesondere Erlass Veterinaerrahmengesetzes Anstrengungen muessen dringend Tierseuchenbekaempfung Handel lebenden Tieren Erzeugnissen tierischen Ursprungs Veterinaerbereich betreffenden Gesundheitsvorschriften Wohlbefinden Tiere konzentriert Fischerei Insgesamt festgestellt Bulgarien Verpflichtungen nachkommt Beitrittsverhandlungen Bereich staatlichen Beihilfen internationalen Fischereiabkommen ergebenden Anforderungen genuegt Verstaerkte Anstrengungen jedoch Bereichen Bestandsbewirtschaftung Flottenmanagement erforderlich Ueberwachungs Kontrollinstrumente konsolidiert einschliesslich Einfuehrung Systems Ueberwachung Schiffe Fortschritte erforderlich gemeinsamen Marktorganisationen GMO Verwaltung Ueberwachung Sektors erforderlichen Verwaltungskapazitaeten entwickeln Verhandlungen Kapitel vorlaeufig abgeschlossen Bulgarien'



After the stop word removal lets do some more preprocessing that is specific to this corpus of data that is removing the currency symbols that are most common 
$|€|¥|₹|£


```python
def currency(text):
    """
    Removes the currency symbols from the text
    :param text: text as string
    :retrun: manipulated text as string
    """
    
    tempVar = text # local variable
    
    tempVar = tempVar.replace('$', '')
    tempVar = tempVar.replace('€', '')
    tempVar = tempVar.replace('¥', '')
    tempVar = tempVar.replace('₹', '')
    tempVar = tempVar.replace('£', '')
    
    return tempVar
```


```python
text_after_currency_removal = currency(text_wo_stop_words)
```


```python
text_after_currency_removal
```




    'Stellungnahme Juli Europaeische Kommission Auffassung gelangt Bulgarien Angleichung Rechtsvorschriften gemeinschaftlichen Besitzstand wenig vorangekommen Bevor Bulgarien Mitgliedschaft erwachsenden Pflichten uebernehmen koenne seien grundlegende Reformen Agrarpolitik erforderlich Bericht November hielt gewisse Fortschritte Hinblick kurzfristigen Prioritaeten Beitrittspartnerschaft fest Liberalisierung Agrarpreise Beseitigung Ausfuhrabgaben nichtmengenmaessigen Ausfuhrbeschraenkungen wurde vorangetrieben vielen Bereichen wurden jedoch weitere Anstrengungen verlangt Landreform Bericht Oktober wurden Fortschritte Angleichung insbesondere Agrarsektor festgestellt wirksame Anwendung nationales Recht umgesetzten Massnahmen hauptsaechlich Grund fehlender finanzieller Mittel weiterhin Schwierigkeiten verbunden weitere Anstrengungen erforderlich insbesondere Bereich Tierhaltung Bodenmarkts Fischereipolitik Bulgarien ersten Massnahmen erlassen meisten UNUebereinkommen Konventionen Fischereisektor unterzeichnet davon ratifiziert Bericht November wurden betraechtlichen Fortschritte dargestellt Bulgarien Agrarsektor gemacht Wein Getreidesektor Jedoch gibt wenig private Investitionen gibt transparenten funktionierenden Grundstuecksmarkt entsprechen Rechtsvorschriften gemeinsamen Marktorganisationen GAP Bereichen Tier Pflanzengesundheit Rechtsvorschriften erlassen worden Durchfuehrung eingesetzten technischen Mittel Humanressourcen reichen Bereich Fischerei Bulgarien erhebliche Anstrengungen unternehmen muessen gemeinschaftlichen Besitzstand umzusetzen Gemeinsame Fischereipolitik anwenden Bericht November geht hervor Bulgarien zufrieden stellende Fortschritte Rechtsangleichung erzielt SAPARD Stelle teilweise Akkreditierung Europaeischen Kommission erhalten Bereich Fischerei Verabschiedung Fischerei Aquakulturgesetzes April wesentlicher Fortschritt erzielt worden Ausserdem vernachlaessigende Verbesserungen institutioneller operationeller Ebene festgestellt worden Verwaltungskapazitaet Bereich Ressourcenbewirtschaftung inspektion kontrollen verstaerkt worden wurden gute Fortschritte Schaffung Fangflottenregisters gemacht Bericht Oktober Bulgarien Bereich Rechtsangleichung Aufbaus institutionellen Rahmens gemachten Fortschritte betont Dagegen Umsetzung Rechtsvorschriften Fortschritte verzeichnet worden Bereich Fischerei Verabschiedung Umsetzung Gemeinsamen Fischereipolitik Bemuehungen unternommen worden Bericht November geht hervor Bulgarien horizontalen Fragen gute Fortschritte erzielt Einfuehrung gemeinsamen Marktorganisationen jedoch Verzug geraten Entwicklung laendlichen Raums sowie Veterinaer Pflanzenschutzsektor Fortschritte gemacht worden Bereich Fischerei Bulgarien seit letzten Bericht zahlreiche Bemuehungen unternommen Bericht Oktober zufolge Fortschritte Angleichung gemeinschaftlichen Besitzstand Verstaerkung Verwaltungskapazitaeten erzielt worden Bulgarien horizontalen Fragen Anwendung EAGFL Entwicklung gemeinsamen Marktorganisationen gute Fortschritte gemacht Bericht Oktober festgestellt Bulgarien Verpflichtungen Anforderungen erfuellt Beitrittsverhandlungen hinsichtlich bestimmter horizontaler Fragen ergeben operativen Systemen sowie Veterinaerbereich jedoch groessere Anstrengungen notwendig Bereich Fischerei muessen Bewirtschaftung Ressourcen Erlass Rechtsvorschriften Marktpolitik Fortschritte gemacht Beitrittsvertrag wurde April unterzeichnet Beitritt fand Januar statt GEMEINSCHAFTLICHER BESITZSTAND Gemeinsame Agrarpolitik GAP zielt darauf ab modernes Agrarsystem erhalten entwickeln landwirtschaftlichen Bevoelkerung angemessene Lebenshaltung gewaehrleisten Belieferung Verbraucher angemessenen Preisen Sorge tragen freien Warenverkehr innerhalb Europaeischen Gemeinschaft verwirklichen EuropaAbkommen bildet Rechtsrahmen Handel Agrarerzeugnissen Bulgarien Europaeischen Gemeinschaft zielt verstaerkte Zusammenarbeit Modernisierung Umstrukturierung Privatisierung bulgarischen Landwirtschaft Agro Nahrungsmittelindustrie sowie Pflanzenschutznormen ab Weissbuch Staaten Mittel Osteuropas Binnenmarkt deckt Rechtsvorschriften Bereichen Veterinaer Pflanzenschutz Futtermittelkontrollen sowie Bestimmungen Vermarktung Erzeugnisse ab Rechtsvorschriften sollen Schutz Verbraucher oeffentlichen Gesundheit sowie Tier Pflanzengesundheit gewaehrleistet Gemeinsame Fischereipolitik umfasst gemeinsame Marktorganisation Strukturpolitik Abkommen Drittlaendern Bewirtschaftung Erhaltung Fischereiressourcen wissenschaftliche Forschung Gebiet EuropaAbkommen enthaelt Bestimmungen Handel Fischereierzeugnissen Gemeinschaft Bulgarien Weissbuch sieht Massnahmen Sektor BEWERTUNG LAGE Landwirtschaft Bulgarien Verhandlungen Kapitel vorlaeufig abgeschlossen Genuss Uebergangsregelungen gekommen Allgemeinen haelt Verpflichtungen entspricht Anforderungen Beitrittsverhandlungen ergeben Einrichtung Zahlstelle Fortschritt gemacht worden diesbezueglichen Rechtsvorschriften muessen erlassen Ausserdem Integriertes Verwaltungs Kontrollsystem INVEKOS eingefuehrt gemeinsamen Marktorganisationen GMO erforderlichen Strukturen vorhanden Rechtsvorschriften Fortschritte Milchsektor verzeichnen dagegen jedoch Sektoren Wein Zucker sowie Obst Gemuese Bereich Politik Entwicklung laendlichen Raums wurde Land dreijaehrige Uebergangszeit gewaehrt Ausserdem besondere Anstrengungen Bereich Rechtsangleichung Veterinaerbereich unternehmen insbesondere Erlass Veterinaerrahmengesetzes Anstrengungen muessen dringend Tierseuchenbekaempfung Handel lebenden Tieren Erzeugnissen tierischen Ursprungs Veterinaerbereich betreffenden Gesundheitsvorschriften Wohlbefinden Tiere konzentriert Fischerei Insgesamt festgestellt Bulgarien Verpflichtungen nachkommt Beitrittsverhandlungen Bereich staatlichen Beihilfen internationalen Fischereiabkommen ergebenden Anforderungen genuegt Verstaerkte Anstrengungen jedoch Bereichen Bestandsbewirtschaftung Flottenmanagement erforderlich Ueberwachungs Kontrollinstrumente konsolidiert einschliesslich Einfuehrung Systems Ueberwachung Schiffe Fortschritte erforderlich gemeinsamen Marktorganisationen GMO Verwaltung Ueberwachung Sektors erforderlichen Verwaltungskapazitaeten entwickeln Verhandlungen Kapitel vorlaeufig abgeschlossen Bulgarien'



After this we will perform lemmetization, for this we are going to use spacy library which has industry standard lemmetizer


```python
import spacy
model_de = spacy.load('de')
```

    /usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192, got 176
      return f(*args, **kwds)
    /usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192, got 176
      return f(*args, **kwds)



```python
def lemmatizer(text): 
    """
    Lemmetize words using spacy
    :param: text as string
    :return: lemmetized text as string
    """
    sent = []
    doc = model_de(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

```


```python
text_after_lema = lemmatizer(text_after_currency_removal.lower())
```


```python
text_after_lema
```




    'stellungnahme juli europaeische kommission auffassung langen bulgarien angleichung rechtsvorschriften gemeinschaftlich besitzstand wenig vorangekommen bevor bulgarien mitgliedschaft erwachsend pflichten uebernehmen koenne sein grundlegende reformen agrarpolitik erforderlich bericht november halten gewiß fortschritte hinblick kurzfristig prioritaeten beitrittspartnerschaft fest liberalisierung agrarpreise beseitigung ausfuhrabgaben nichtmengenmaessigen ausfuhrbeschraenkungen werden vorangetrieben viel bereichen werden jedoch weit anstrengungen verlangen landreform bericht oktober werden fortschritte angleichung insbesondere agrarsektor feststellen wirksam anwendung national recht umgesetzt massnahmen hauptsaechlich grund fehlend finanziell mittel weiterhin schwierigkeiten verbinden weit anstrengungen erforderlich insbesondere bereich tierhaltung bodenmarkts fischereipolitik bulgarien erst massnahmen erlassen meist unuebereinkommen konventionen fischereisektor unterzeichnen davon ratifizieren bericht november werden betraechtlichen fortschritte darstellen bulgarien agrarsektor machen wein getreidesektor jedoch geben wenig privat investitionen geben transparent funktionierend grundstuecksmarkt entsprechen rechtsvorschriften gemeinsam marktorganisationen gap bereichen tier pflanzengesundheit rechtsvorschriften erlassen werden durchfuehrung eingesetzt technisch mittel humanressourcen reich bereich fischerei bulgarien erheblich anstrengungen unternehmen muessen gemeinschaftlich besitzstand umsetzen gemeinsam fischereipolitik anwenden bericht november gehen hervor bulgarien zufrieden stellend fortschritte rechtsangleichung erzielen sapard stelle teilweise akkreditierung europaeischen kommission erhalten bereich fischerei verabschiedung fischerei aquakulturgesetzes april wesentlich fortschritt erzielen werden ausserdem vernachlaessigende verbesserungen institutionell operationeller eben feststellen werden verwaltungskapazitaet bereich ressourcenbewirtschaftung inspektion kontrollen verstaerkt werden werden gut fortschritte schaffung fangflottenregisters machen bericht oktober bulgarien bereich rechtsangleichung aufbaus institutionell rahmens gemacht fortschritte betonen dagegen umsetzung rechtsvorschriften fortschritte verzeichnen werden bereich fischerei verabschiedung umsetzung gemeinsam fischereipolitik bemuehungen unternehmen werden bericht november gehen hervor bulgarien horizontal fragen gut fortschritte erzielen einfuehrung gemeinsam marktorganisationen jedoch verzug geraten entwicklung laendlichen raums sowie veterinaer pflanzenschutzsektor fortschritte machen werden bereich fischerei bulgarien seit letzt bericht zahlreiche bemuehungen unternehmen bericht oktober zufolge fortschritte angleichung gemeinschaftlich besitzstand verstaerkung verwaltungskapazitaeten erzielen werden bulgarien horizontal fragen anwendung eagfl entwicklung gemeinsam marktorganisationen gut fortschritte machen bericht oktober feststellen bulgarien verpflichtungen anforderungen erfuellt beitrittsverhandlungen hinsichtlich bestimmt horizontal fragen ergeben operativ systemen sowie veterinaerbereich jedoch groessere anstrengungen notwendig bereich fischerei muessen bewirtschaftung ressourcen erlass rechtsvorschriften marktpolitik fortschritte machen beitrittsvertrag werden april unterzeichnen beitreten finden januar statt gemeinschaftlich besitzstand gemeinsam agrarpolitik gap zielen darauf ab modern agrarsystem erhalten entwickeln landwirtschaftlich bevoelkerung angemessen lebenshaltung gewaehrleisten belieferung verbraucher angemessen preisen sorge tragen frei warenverkehr innerhalb europaeischen gemeinschaft verwirklichen europaabkommen bilden rechtsrahmen handel agrarerzeugnissen bulgarien europaeischen gemeinschaft zielen verstaerkte zusammenarbeit modernisierung umstrukturierung privatisierung bulgarisch landwirtschaft agro nahrungsmittelindustrie sowie pflanzenschutznormen ab weissbuch staaten mittel osteuropas binnenmarkt decken rechtsvorschriften bereichen veterinaer pflanzenschutz futtermittelkontrollen sowie bestimmungen vermarktung erzeugnisse ab rechtsvorschriften sollen schutz verbraucher oeffentlichen gesundheit sowie tier pflanzengesundheit gewaehrleistet gemeinsam fischereipolitik umfasst gemeinsam marktorganisation strukturpolitik abkommen drittlaendern bewirtschaftung erhaltung fischereiressourcen wissenschaftliche forschung gebiet europaabkommen enthaelt bestimmungen handel fischereierzeugnissen gemeinschaft bulgarien weissbuch sehen massnahmen sektor bewertung lage landwirtschaft bulgarien verhandlungen kapitel vorlaeufig abschließen genuss uebergangsregelungen kommen allgemein haelt verpflichtungen entsprechen anforderungen beitrittsverhandlungen ergeben einrichtung zahlstelle fortschritt machen werden diesbezueglichen rechtsvorschriften muessen erlassen ausserdem integriert verwaltungs kontrollsystem invekos eingefuehrt gemeinsam marktorganisationen gmo erforderlich strukturen vorhanden rechtsvorschriften fortschritte milchsektor verzeichnen dagegen jedoch sektoren wein zucker sowie obst gemuese bereich politik entwicklung laendlichen raums werden land dreijaehrige uebergangszeit gewaehrt ausserdem besonder anstrengungen bereich rechtsangleichung veterinaerbereich unternehmen insbesondere erlass veterinaerrahmengesetzes anstrengungen muessen dringen tierseuchenbekaempfung handel lebend tieren erzeugnissen tierisch ursprungs veterinaerbereich betreffend gesundheitsvorschriften wohlbefinden tiere konzentrieren fischerei insgesamt feststellen bulgarien verpflichtungen nachkommen beitrittsverhandlungen bereich staatlich beihilfen international fischereiabkommen ergebend anforderungen genuegt verstaerkte anstrengungen jedoch bereichen bestandsbewirtschaftung flottenmanagement erforderlich ueberwachungs kontrollinstrumente konsolidieren einschliesslich einfuehrung systems ueberwachung schiffe fortschritte erforderlich gemeinsam marktorganisationen gmo verwaltung ueberwachung sektors erforderlich verwaltungskapazitaeten entwickeln verhandlungen kapitel vorlaeufig abschließen bulgarien'



Ok I dont know german so I can't say if it did its job or not but I we will check it


```python
text_after_lema == text_after_currency_removal.lower()
```




    False



This is a good indication as its a proof that it did something for making sure lets just try some thing


```python
a = 'The quick brown fox jumps over the lazy dog'
b = 'The quick brown fox jumps over the lazy dog'
c = 'The quick brown fox jump over the lazy dog'   
```


```python
a == b
```




    True




```python
b == c
```




    False



Yep it works so I will assume that the lemmetizer worked. This is probably all the preprocessig required for preprocessing German Text

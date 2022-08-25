+++
title =  "NLP and ML notes - Part 2"
description =  "Handy dandy notes"
date = "2018-11-02"
author = "Jay Vala"
tags = ["python", "text", "nlp", "analysis", "bayes", "conditional-porbability"]
+++

## Bayes Theorem

In conditional probability we have a order of probabilities. That is 

$$P(A|B) =\frac{P(A\cap B)}{P(B)}$$

But it may not happen that we have the probability in this order or may happen that we may not have the probability of $P(B)$, thats when Bayes Theorm comes into play.

Bayes Theorem lets you swap the order of dependence between events. So we can calculate $P(A\|B)$ in terms of $P(B\|A)$. This is useful as mentioned above when it is difficult to calculate any one of the term in conditinal probability.

So,

$$P(B|A) = \frac{P(A\cap B)}{P(A)} = \frac{P(A|B)P(B)}{P(A)}$$

Here, the denominator $P(A)$ can be viewed as *Normalizing Constant* something that ensures we always have a probability function.

So, to evaluate this denominator,

$$P(A\cap B) = P(A|B)P(B)$$

$$P(A|\cap \bar{B}) = P(A| \bar{B})P(\bar{B})$$

Therefore, by additivity rule, 

$$P(A) = P(A\cap B) + P(A\cap \bar{B})$$  

$$P(A)= P(A|B)P(B) + P(A| \bar{B})P(\bar{B})$$

Hence, the **Bayes Theorem** becomes,


$$P(B|A) = \frac{P(A|B)P(B)}{P(A|B)P(B) + P(A| \bar{B})P(\bar{B})}$$

>Example:Treasure chest A contains *20 Gold* and *30 Silver* coins, Chest B contains *50 Gold* and *50 Silver* coins. You randomly choose a treasure chest to open, and then randomly choose a coin from that treasure chest. If the coin you choose is gold, then what is the probability that you chose chest A?

Solution: 

>Chest A = 20 Gold, 30 Silver

>Chest B = 50 Gold, 50 Silver

So let $P(A)$ be the probability of selecting Chest A, and $P(B)$ be the probability of selecting Chest B, and let $P(G\|A)$ be the probability of selecting a Gold coint from chest A.

$$P(A) = \frac{1}{2}$$

$$P(B) = \frac{1}{2}$$

$$P(G|A) = \frac{20}{50} = \frac{2}{5}$$

$$P(G|B) = \frac{50}{100} = \frac{1}{2}$$

So, according to **Bayes Theorem**,Probability of chest being A given that the coin was gold is 

$$P(A|G) = \frac{P(G|A)P(A)}{P(G)}$$

Now, we can clearly see the problem here, we don't have $P(G)$ but we can express it using the additive method explained above. So,

 $$P(G) = P(A)P(G|A) + P(\bar{A})P(G|\bar{A})$$

But, $\bar{A}$ in this case is $B$, hence the above equation becomes

 $$P(G) = P(A)P(G|A) + P(B)P(G|B)$$

Putting this in above Bayes formula

$$P(A|G) = \frac{P(G|A)P(A)}{P(G)}$$

$$P(A|G) = \frac{P(G|A)P(A)}{P(A)P(G|A) + P(B)P(G|B)}$$

Putting the respective values in we get,

$$P(A|G) = \frac{\frac{1}{2} . \frac{2}{5}}{\frac{1}{2} . \frac{2}{5} + \frac{1}{2} .\frac{1}{2}}$$

$$P(A|G) =\frac{\frac{1}{5}}{\frac{1}{5}+\frac{1}{4}}$$

$$P(A|G) = \frac{1}{5} .\frac{20}{9}$$

$$P(A|G) = \frac{20}{45}$$

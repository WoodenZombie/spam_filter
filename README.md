# Spam Filter using Multinomial Naive Bayes Algorithm
Spam Filter using Multinomial Naive Bayes algorithm

# Dataset 

The file contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

-> A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. 
-> A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available.

Link for an dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

# Algorithm

First, text being processed by removing all the stopwords, punctuation and then being tokenized
Then, I split data into train and test subsets, turn it into a document-term matrix and apply a Multinomial Naive Bayes algorithm

# Multinomial Naive Bayes

Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.

To understand Naive Bayes theorem’s working, it is important to understand the Bayes theorem concept first as it is based on the latter.

Bayes theorem, formulated by Thomas Bayes, calculates the probability of an event occurring based on the prior knowledge of conditions related to an event. It is based on the following formula:

P(A|B) = P(A) * P(B|A)/P(B)

Where we are calculating the probability of class A when predictor B is already provided.

P(B) = prior probability of B

P(A) = prior probability of class A




P(B|A) = occurrence of predictor B given class A probability




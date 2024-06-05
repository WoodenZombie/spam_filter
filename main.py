import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import string
from nltk.corpus import stopwords


def text_process(msg):
    """
    Takes in a string of text, then removes all punctuation, stopwords and returns a cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

    # Check characters to see if they are in punctuation
    no_punct = [char for char in msg if char not in string.punctuation]

    # Join the characters again to form the string.
    no_punct = ''.join(no_punct)

    # Now just remove any stopwords
    return ' '.join([word for word in no_punct.split() if word.lower() not in STOPWORDS])


# Try to read a CSV file
try:
    data = pd.read_csv('./spam.csv', encoding='latin1')
    print("Successfully loaded.")
    data.dropna(how="any", inplace=True, axis=1)
    data.columns = ['label', 'message']
    print(data.head())
except Exception as EXC:
    print(f"File read error: {EXC}")

# Maps ham and spam messages as ones and two's
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
# Creates a new column with message length
data['message_len'] = data.message.apply(len)

# Processes the data
data['clean_msg'] = data.message.apply(text_process)

# how to define X and y (from the SMS data) for use with vectorizer
X = data.clean_msg
y = data.label_num

# splits data into random test and train subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# instantiate the vectorizer
"""
Vectorization is a technique used to improve the performance of Python code by eliminating the use of loops.
 This feature can significantly reduce the execution time of code.
"""
vect = CountVectorizer()
vect.fit(X_train)

# learn training data vocabulary, then use it to create a document-term matrix
"""
    A document-term matrix is a mathematical matrix that describes
    the frequency of terms that occur in each document in a collection.
"""
X_train_dtm = vect.transform(X_train)

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)

# instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
print("Accuracy Score:")
print(metrics.accuracy_score(y_test, y_pred_class))

# print the confusion matrix
"""
    A confusion matrix represents the prediction summary in matrix form.
    It shows how many prediction are correct and incorrect per class.
"""
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_class))

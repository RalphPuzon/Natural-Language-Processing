#NLP:

#LIBRARIES:
import os
os.chdir('C:\\Users\\Ralph\\Desktop\\Courses\\ML\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#quoting = 3 igonres double quotes embedded in text

#CLEANING TEXT:
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) # replace nonchar with space
review = review.lower()

#remove nonkey words
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
review = review.split()
#review = [word for word in review if word not in set(stopwords.words('english'))]
#set() <- treat input as a set, faster algorithm to go through set than list in py?
 
#STEMMING:
import nltk.stem.porter
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#return review to string:
review = ' '.join(review)



#FULL RUN OF CLEANUP CODE:
import re
import nltk
from nltk.corpus import stopwords
import nltk.stem.porter
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range (0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


#BAG OF WORDS MODEL:
#works on a sparse matrix utilizing classification convention:
    
#create matrix of features + target

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #x, since it is a matrix of features
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#want more test than predict, tf test_size = 0.20

#no need for feature scaling

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

































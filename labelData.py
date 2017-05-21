## from file named check_model_accuracy.py conculde that Random Frorest and Decision Tree classifier are most suitable classifier for given dataset because it gives higher accuracy. #

# import necessary packages
import pandas
import os
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

os.chdir(r"/Users/kaushalpurohit/GitRepository/LabelData/")

# Read text file from the local and create dataframe
data = []
label = []

with open(r"../Label/LabelledData.txt",encoding = "utf-8") as f:
    for i in f:
        a = i.split(" ,,, ")
        data.append(a[0])
        label.append(a[1])
        
train = pandas.DataFrame()
train["data"] = data
train["label"] = label

#strip \n from tail
train.label = train.label.str.strip("\n")

## text cleaning
#Stopword removing#
train["data"] = [word for word in train["data"] if word not in stopwords.words('english')]

#lowercase#
train.data = train.data.str.lower()

#remove punctuation#
train.data = train.data.str.strip("?")

# create training and testing set from main data for cross validation (here training size is 60% of main data and testing size is 40% of main data)
X_train,X_test,Y_train,Y_test = train_test_split(train.data,train.label,test_size=0.4)

X_test.shape

# Classification model for Random Forest
text_clf_RF = Pipeline([("vect",CountVectorizer()),("tfidf",TfidfTransformer()),("clf",RandomForestClassifier())])

# Classification model for Decision Tree
text_clf_CART = Pipeline([("vect",CountVectorizer()),("tfidf",TfidfTransformer()),("clf",DecisionTreeClassifier())])

# Fit model on training set
classification_RF = text_clf_RF.fit(X_train,Y_train)
classification_CART = text_clf_CART.fit(X_train,Y_train)

# Predict model on Test set
text_clf_RF=classification_RF.predict(X_test)
text_clf_CART=classification_CART.predict(X_test)

# Check accuracy of model

print("Random Forest model Accuracy is: "+ "" + np.mean(text_clf_RF == Y_test))     
print("Decision Tree model Accuracy is: "+ "" + np.mean(text_clf_CART == Y_test))

# import package
import pandas
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

# Read text data from local and  create Dataframe
data = []
label = []

with open(r"/Users/kaushalpurohit/Documents/Himani/Label/LabelledData.txt",encoding = "utf-8") as f:
    for i in f:
        a = i.split(" ,,, ")
        data.append(a[0])
        label.append(a[1])
        

labeldata1 = pandas.DataFrame()
labeldata1["data"] = data
labeldata1["label"] = label
#strip \n from tail
labeldata1.label = labeldata1.label.str.strip("\n")

## text cleaning
#Stopword removing#
labeldata1["data"] = [word for word in labeldata1["data"] if word not in stopwords.words('english')]

#lowercase#
labeldata1.data = labeldata1.data.str.lower()

#remove punctuation#
labeldata1.data = labeldata1.data.str.strip("?")

# create count vector 
count_vect = CountVectorizer()
counts = count_vect.fit_transform(labeldata1.data)

# create tfidf matrix 
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(counts)


# To check which model is more suitable to data,initialize different classification models.
model=[]

model.append(('SVM', SVC()))
model.append(('LDA', LinearDiscriminantAnalysis()))
model.append(('CART', DecisionTreeClassifier()))
model.append(('NB', GaussianNB()))
model.append(('RF', RandomForestClassifier()))

# By cross validation using Kfold analysis identify model accuracy
results = []
names = []
scoring = 'accuracy'
for name, model in model:
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, tfidf.toarray(), labeldata1.label, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Generate boxplot
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# By from the code we get high accuracy of Random Forest and Decision Tree model so we will do prediction using these two models.(Filename: labelData.py)
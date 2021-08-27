
import pandas as pd

df=pd.read_csv(r'Data.csv', encoding = "ISO-8859-1")

print(df.head())

#if label is 1 it means stock price will increase with respect to particular headlines,if 0 it means it will either constant or decrease

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data=train.iloc[:,2:27]

print(data.columns)

# Removing punctuations
#apart from a-z ,remove all by blank 
data.replace("[^a-zA-Z]"," ", inplace=True)

print(data.head())

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
print(new_Index)

data.columns= new_Index



### accessing data of 1st column
print(data['0'])



# Convertng headlines to lower case
#why str.lower on 0,1,2---- bcz we can apply function on a column only
for index in new_Index:
    data[index]=data[index].str.lower()
print(data.head(1))

##print entire data of 2nd row
print(data.iloc[1,0:25])

#create a list of all headlines for row 2,as 25 headlines in row 2
#' '.join([str(x) for x in data.iloc[1,0:25]])

## create a list headlines to insert all the headlines of all rows
headlines=[]
for row in range(0,len(data)):
    headlines.append(' '.join([str(i) for i in data.iloc[row,0:25]]))

#headlines[0:3]



"""#### Implement Bag of Words"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

## implement BAG OF WORDS
cv=CountVectorizer(ngram_range=(2,2))
traindata_x=cv.fit_transform(headlines)

# implement RandomForest Classifier,almost takes 2 mins
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindata_x,train['Label'])

## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))

test_data = cv.transform(test_transform)

randomclassifier.fit(traindata_x,train['Label'])

predictions = randomclassifier.predict(test_data)





"""### plot_confusion_matrix"""

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=["positive", "negative"]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
cm=confusion_matrix(test['Label'],predictions)
print(cm)
plot_confusion_matrix(cm)    
plt.show()

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)



"""### Implement MultinomialNB"""

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(traindata_x,train['Label'])

predictions = nb.predict(test_data)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)

plot_confusion_matrix(matrix)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.read_csv(r'Data.csv', encoding="ISO-8859-1")

print(df.head())


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data = train.iloc[:, 2:27]

print(data.columns)


data.replace("[^a-zA-Z]", " ", inplace=True)

print(data.head())

list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
print(new_Index)

data.columns = new_Index


for index in new_Index:
    data[index] = data[index].str.lower()
print(data.head(1))


headlines = []
for row in range(0, len(data)):
    headlines.append(' '.join([str(i) for i in data.iloc[row, 0:25]]))


"""#### Implement Bag of Words"""


cv = CountVectorizer(ngram_range=(2, 2))
traindata_x = cv.fit_transform(headlines)


randomclassifier = RandomForestClassifier(
    n_estimators=200, criterion='entropy')
randomclassifier.fit(traindata_x, train['Label'])


test_transform = []
for row in range(0, len(test)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

test_data = cv.transform(test_transform)

randomclassifier.fit(traindata_x, train['Label'])

predictions = randomclassifier.predict(test_data)


"""### plot_confusion_matrix"""


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
cm = metrics.confusion_matrix(test['Label'], predictions)
print(cm)
plot_confusion_matrix(cm)
plt.show()

# Import library to check accuracy

score = accuracy_score(test['Label'], predictions)
print(score)
report = classification_report(test['Label'], predictions)
print(report)


"""### Implement MultinomialNB"""

nb = MultinomialNB()
nb.fit(traindata_x, train['Label'])

predictions = nb.predict(test_data)
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
score = accuracy_score(test['Label'], predictions)
print(score)
report = classification_report(test['Label'], predictions)
print(report)

plot_confusion_matrix(matrix)

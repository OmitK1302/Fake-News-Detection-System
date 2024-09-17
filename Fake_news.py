import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


fake_data = pd.read_csv('./Fake.csv')
true_data = pd.read_csv('./True.csv')

fake_data["class"] = 0
true_data["class"] = 1

fake_data_manual_testing = fake_data.tail(10)
for i in range(23480, 23470, -1):
    fake_data.drop([i], axis=0, inplace=True)

true_data_manual_testing = true_data.tail(10)
for i in range(21416, 21406, -1):
    true_data.drop([i], axis=0, inplace=True)

# fake_data_manual_testing['class'] = 0
# true_data_manual_testing['class'] = 1

fake_data_manual_testing.loc[:, 'class'] = 0
true_data_manual_testing.loc[:, 'class'] = 1


data_merge = pd.concat([fake_data, true_data], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)
data['text'] = data['text'].apply(lambda x: re.sub('\[.*?\]', '', x))
data['text'] = data['text'].apply(lambda x: re.sub("\\W", " ", x))
data['text'] = data['text'].apply(lambda x: re.sub('https?://\S+|www\.\S+', '', x))
data['text'] = data['text'].apply(lambda x: re.sub('<.*?>+', '', x))
data['text'] = data['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
data['text'] = data['text'].apply(lambda x: re.sub('\n', '', x))
data['text'] = data['text'].apply(lambda x: re.sub('\w*\d\w*', '', x))
data['text'] = data['text'].apply(lambda x: x.lower())

data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)

# Random Forest
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)

print("Logistic Regression:")
print(classification_report(y_test, pred_lr))
print("Random Forest:")
print(classification_report(y_test, pred_rf))

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(lambda x: re.sub('\W', ' ', x.lower()))
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    print("\nLogistic Regression Prediction: {}".format(output_label(pred_LR[0])))
    print("Random Forest Prediction: {}".format(output_label(pred_RF[0])))

while True:
    print("What do you want to do:")
    print("1. Fake news Detection")
    print("2. Exit")
    
    try:
        choice = int(input("Enter your choice: "))
        if choice == 1:
            news = input("Enter the news to check: \n")
            testing(news)
        elif choice == 2:
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter a number.")
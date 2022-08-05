import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  

url = 'https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv'
df = pd.read_csv(url, header=0, sep=",")
#Step 2:
df.head()

df.info()

df.describe()
df.sample(5)
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == 'True' else 0)
df.sample(5)

df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['url','is_spam']]
df.shape

import regex as re

clean_desc = []

for w in range(len(df.url)):
    desc = df['url'][w].lower()
    
    #remove punctuation
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    #remove tags
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    #remove digits and special chars
    desc=re.sub("(\\d|\\W)+"," ",desc)
    
    clean_desc.append(desc)

#assign the cleaned descriptions to the data frame
df['url'] = clean_desc
        
df.head()

stop_words = ['is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get', 'not', 'here', 'in', 'im', 'have', 'on', 
're', 'https', 'com']

from sklearn.feature_extraction.text import CountVectorizer

message_vectorizer = CountVectorizer().fit_transform(df['url'])

X_train, X_test, y_train, y_test = train_test_split(message_vectorizer, df['is_spam'], test_size = 0.25, random_state = 1234, shuffle = True)

from sklearn import model_selection, svm

classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

from sklearn.metrics import classification_report, accuracy_score

from sklearn import datasets
from sklearn.svm import SVC

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import re
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('/Users/finsup/Downloads/NLP-Heroku/spam.csv', encoding = "latin-1")

df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

processed_messages = []

for message in range(0, len(X)):
	processed_message = re.sub(r'\W', ' ', str(X[message]))
	processed_message = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_message)
	processed_message = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_message)
	processed_message = re.sub(r'\s+', ' ', processed_message, flags=re.I)
	processed_message = processed_message.lower()
	processed_messages.append(processed_message)

cv = CountVectorizer(min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

predictions = clf.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))




joblib.dump(clf, '/Users/finsup/Downloads/NLP-Heroku/NB_spam_model.pkl')
joblib.dump(cv, '/Users/finsup/Downloads/NLP-Heroku/cv.pkl')


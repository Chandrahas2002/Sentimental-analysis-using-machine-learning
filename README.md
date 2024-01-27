# Sentimental-analysis-using-machine-learning

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisp

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisp

df = pd.read_csv('minitwitter.csv')

df.head()

df.info()


df.isnull().sum()

df.columns


text_df = df.drop(['Target', 'ID', 'DATE', 'FLAG', 'USER'], axis=1)
text_df.head()

print(text_df['TEXT'].iloc[0],"\n")
print(text_df['TEXT'].iloc[1,],"\n")
print(text_df['TEXT'].iloc[2],"\n")
print(text_df['TEXT'].iloc[3],"\n")
print(text_df['TEXT'].iloc[4],"\n")
print(text_df['TEXT'].iloc[5],"\n")
print(text_df['TEXT'].iloc[6],"\n")
print(text_df['TEXT'].iloc[7],"\n")
print(text_df['TEXT'].iloc[8],"\n")
print(text_df['TEXT'].iloc[9],"\n")



text_df.info()

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)
    

import nltk
nltk.download('punkt')
text_df.text = text_df['TEXT'].apply(data_processing)

text_df = text_df.drop_duplicates('TEXT')

from nltk.stem import PorterStemmer
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

from nltk.stem import PorterStemmer

text_df['TEXT'] = text_df['TEXT'].apply(lambda x: stemming(x))


text_df.head()

print(text_df['TEXT'].iloc[0],"\n")
print(text_df['TEXT'].iloc[1,],"\n")
print(text_df['TEXT'].iloc[2],"\n")
print(text_df['TEXT'].iloc[3],"\n")
print(text_df['TEXT'].iloc[4],"\n")
print(text_df['TEXT'].iloc[5],"\n")
print(text_df['TEXT'].iloc[6],"\n")
print(text_df['TEXT'].iloc[7],"\n")
print(text_df['TEXT'].iloc[8],"\n")
print(text_df['TEXT'].iloc[9],"\n")

text_df.info()

def polarity(text):
    return TextBlob(text).sentiment.polarity

text_df['polarity'] = text_df['TEXT'].apply(polarity)

text_df.info()

text_df.head(11)

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

text_df['sentiment'] = text_df['polarity'].apply(sentiment)

text_df.head()










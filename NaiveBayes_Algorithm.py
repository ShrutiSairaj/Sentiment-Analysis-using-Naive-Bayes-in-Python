
##########Sentiment Analysis

#Data Import
import pandas as pd
input_data = pd.read_csv("sentiment_checkout.csv",encoding='latin1')



#Basic Details of the data
input_data.shape
input_data.columns
input_data.head(10)

#Frequency of sentiment col
input_data['Type_comment'].value_counts()


##########
#Creating Document Term Matrix

from sklearn.feature_extraction.text import CountVectorizer

countvec1 = CountVectorizer()
dtm_v1 = pd.DataFrame(countvec1.fit_transform(input_data['openend']).toarray(), columns=countvec1.get_feature_names(), index=None)
dtm_v1['Type_comment'] = input_data['Type_comment']
dtm_v1.head()

#############################################
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#####Writing a Custom Tokenizer
stemmer = PorterStemmer()
def tokenize(text):
    text = stemmer.stem(text)               #stemming
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations and Underscores
    tokens = nltk.word_tokenize(text)       #tokenizing
    return tokens

countvec = CountVectorizer(min_df= 5, tokenizer=tokenize, stop_words=stopwords.words('english'))
dtm = pd.DataFrame(countvec.fit_transform(input_data['openend']).toarray(), columns=countvec.get_feature_names(), index=None)
#Adding label Column
dtm['Type_comment'] = input_data['Type_comment']
dtm.head()

###Building training and testing sets
df_train = dtm[:10000]
df_test = dtm[10000:13871]
df_pred=dtm[13871:]


################# Building Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
X_train= df_train.drop(['Type_comment'], axis=1)
#Fitting model to our data
clf.fit(X_train, df_train['Type_comment'])

#Accuracy
X_test= df_test.drop(['Type_comment'], axis=1)
clf.score(X_test,df_test['Type_comment'])

#Prediction




pred_sentiment=clf.predict(df_pred.drop('Type_comment', axis=1))
print(pred_sentiment)



input_data['pred']=''


input_data.iloc[13871:,18]=pred_sentiment=clf.predict(df_pred.drop('Type_comment', axis=1))


input_data.to_csv('sentiment_checkout_pred.csv')



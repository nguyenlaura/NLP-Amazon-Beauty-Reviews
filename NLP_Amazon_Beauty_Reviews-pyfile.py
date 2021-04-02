#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser

import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[9]:


df = pd.read_json('data/Luxury_Beauty_5.json', lines=True)


# In[10]:


# remove dupes starting at line 10947 to 12147
df.drop(df.index[10947:12148], axis=0, inplace=True)


# In[11]:


df.head()


# In[17]:


# in case of non-text, changing the reviewText column to text 
df['reviewText'] = df['reviewText'].apply(lambda x: str(x))


# In[18]:


# change time from obj to datetime
# vote of review usefulness is abysmally low -- Only 19% of reviews have a helpfulness score


# In[19]:


df['reviewTime'] = pd.to_datetime(df['reviewTime'])
print(df['reviewTime'].min())
print(df['reviewTime'].max())
df.info()


# ### EDA
# * count unique ASIN -- show num unique beauty items
# * count unique reviewerID -- show num unique users
# * time frame -- min and max
# * distribution of ratings
# * distribution of verified vs unverified purchases

# In[20]:


# df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'],unit='s')
# df['unixReviewTime'] = df['unixReviewTime'].dt.year


# In[253]:


df.describe()


# In[21]:


df['asin'].value_counts()


# In[22]:


# how many asins have only 1 review ---> 19
(df['asin'].value_counts() == 1).value_counts()


# In[23]:


df['reviewerID'].value_counts()


# In[24]:


import seaborn as sns
sns.pairplot(df);
#unixreviewtime not converted


# In[ ]:


sns.pairplot(df)
#unixreviewtime converted to year

#is it possible to plot date? last cell throws error. i think it's bc of the dashes in dates


# In[25]:


df['overall'].value_counts()


# In[242]:


#distribution of ratings
fig, ax = plt.subplots()
ax.hist(df['overall'], bins=10, color = '#00A8E1')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1', '2', '3', '4', '5'])
ax.set_ylabel('Frequency of Reviews')
ax.set_title('Ratings Distribution');
# more 5 stars given -- skewed
# fig.savefig('ratings_distribution.png')


# In[243]:


#frequency of reviews by month
fig, ax = plt.subplots()
ax.hist(pd.to_datetime(df['unixReviewTime'],unit='s').dt.month, bins=24, color= 'orange', width= 0.55)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax.set_xticklabels(['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], rotation=45)
ax.set_ylabel('Frequency of Reviews')
ax.set_title('Reviews by Month \n (2005 to 2018)');
# fig.savefig('reviews_by_month.png')

# more beauty product reviews in the spring around March, summer around June and July, 
# and in the fall around black friday/thanksgiving in November


# In[244]:


#frequency of reviews by year
fig, ax = plt.subplots()
ax.hist(pd.to_datetime(df['unixReviewTime'],unit='s').dt.year, bins=28, color = 'orange', width =0.75)
# ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# ax.set_xticklabels(['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'], rotation=45)
ax.set_ylabel('Frequency of Reviews')
ax.set_title('Reviews by Year \n (2005 to 2018)');
# fig.savefig('reviews_by_year.png')

# more reviews starting 2013 and beyond
# 2018 looks low but that's because we stopped collection in september 2018 -- not full year


# In[190]:


df['verified'].value_counts()


# In[214]:


# verified by rating
grouped = df.groupby(['overall'])
rating = []
avg_ratings = []
# for category_name,category_df in grouped:
    print(category_df.loc[df['verified'] == True, 'overall'])
    print(category_df.loc[df['verified'] == False, 'overall'])


# In[247]:


# set width of bars
barWidth = 0.25
 
# set heights of bars
bars1 = [659, 668, 1322, 2174, 11737]
bars2 = [436, 828, 2442, 5419, 7392]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='#00A8E1', width=barWidth, edgecolor='white', label='Verified')
plt.bar(r2, bars2, color='orange', width=barWidth, edgecolor='white', label='Not Verifed')

 
# Add xticks on the middle of the group bars
plt.xlabel('Ratings', fontweight='bold')
plt.ylabel('Frequency of Ratings')
plt.xticks([r + barWidth for r in range(len(bars1))], ['1', '2', '3', '4', '5'])
plt.title('Frequency of Ratings by \n Verified & Unverified Buyers')
# Create legend & Show graphic
plt.legend()
plt.show()

plt.savefig('verified_unverified.png')


# ## Text Processing

# In[40]:


import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

#apply remove_accents fxn
#gave a lot of errors until i applied to_string on reveiwText above!
df['processed_text'] = df['reviewText'].transform(remove_accents)
#remove ellipses
df['processed_text'] = df['processed_text'].replace('\.+','.',regex=True)
df['processed_text'].head()


# In[79]:


docs = []
for row in df['processed_text']:
    docs.append(row)
df.head()
# docs


# In[41]:


tokens = [sent for sent in map(word_tokenize, df['processed_text'])]

list(enumerate(tokens))


# In[42]:


tokens_lower = [[word.lower() for word in sent]
                 for sent in tokens]


# In[43]:


stopwords_ = set(stopwords.words('english'))


# In[44]:


punctuation_ = set(string.punctuation)
print("--- punctuation: {}".format(string.punctuation))


# In[61]:


def filter_tokens(sent):
    return([w for w in sent if not w in stopwords_ and not w in punctuation_])

tokens_filtered = list(map(filter_tokens, tokens_lower))

for sent in tokens_filtered:
    print("--- sentence tokens: {}".format(sent))


# ### Bag of Words

# In[231]:


stemmer_snowball = SnowballStemmer('english')
tokens_stemsnowball = [list(map(stemmer_snowball.stem, sent)) for sent in tokens_filtered]
print("--- sentence tokens (snowball): {}".format(tokens_stemsnowball[0]))
#'data-hook=', "''", 'product-link-link', "''", 
#'class=', "''", 'a-link-norm', "''", 'href=', "''", 
#'/angela-s-garden-7120-921-kids-garden-glove-honey-bee/dp/b000p8dhqg/ref=cm_cr_arp_d_rvw_txt',
#'ie=utf8'

#^^^ info from images was left in... remove href?
# There are 479 instances -- not much so may not be important


# In[170]:


#append lists to an arr which will contain all tokens -- bag of words
#pass bag of words through tfidf vectorizer
bows = [] # list of list
bow = [] # all the words
for sent in tokens_filtered:
    for word in sent:
        bow.append(word)
    bows.append(sent)
# bow


# In[251]:


from wordcloud import WordCloud, STOPWORDS
comment_words = ' '
stopwords = set(STOPWORDS) 

for word in bow: 
    comment_words += word + ' '

wordcloud = WordCloud(width = 1200, height = 800, 
            background_color ='white',
            max_words=100,
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)

# plt.savefig('top_200_words.png')


# In[274]:


# column of text with 5 stars only
five_star_text = df.loc[df['overall'] == 5, 'reviewText']
from wordcloud import WordCloud, STOPWORDS
comment_words = ' '
stopwords = set(STOPWORDS) 

# iterate through the csv file 
for val in five_star_text: 
# typecaste each val to string 
    val = str(val) 
   # split the value 
    tokens = val.split()

# # Converts each token into lowercase 
# for i in range(len(tokens)): 
#     tokens[i] = tokens[i].lower() 

    for words in tokens: 
        comment_words += words + ' '

wordcloud = WordCloud(width = 1200, height = 800, 
            background_color ='white',
            max_words=100,
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)

# plt.savefig('5_star_words.png')


# In[105]:


bows_lst = list()
for lst in bows:
    new_bow = ' '.join(lst)
    bows_lst.append(new_bow)
# bows_lst[0]


# In[83]:


# displaying bows
for i in range(len(docs)):
    print("\n--- review: {}".format(docs[i]))
    print("--- bow: {}".format(bows[i]))


# In[ ]:


# n-grams: bigrams and trigrams


# ## TFIDF and SVM
# ### Term Frequency

# In[89]:


from collections import Counter
#go through bag of words and count occurrence of each keyword
# term occurence = counting distinct words in each bag
term_occ = list(map(lambda bow : Counter(bow), bows))

# term frequency = occurences over length of bag
term_freq = list()
for i in range(len(docs)):
    term_freq.append( {k: (v / float(len(bows[i])))
                       for k, v in term_occ[i].items()} )

# displaying occurences
for i in range(len(docs)):
    print("\n--- review: {}".format(docs[i]))
    print("--- bow: {}".format(bows[i]))
    print("--- term_occ: {}".format(term_occ[i]))
    print("--- term_freq: {}".format(term_freq[i]))


# ### Document Frequency

# In[87]:


# document occurence = number of documents having this word
# term frequency = occurences over length of bag

doc_occ = Counter( [word for bow in bows for word in set(bow)] )

# document frequency = occurences over length of corpus
doc_freq = {k: (v / float(len(docs)))
            for k, v in doc_occ.items()}

# displaying vocabulary
print("\n--- full vocabulary: {}".format(doc_occ))
print("\n--- doc freq: {}".format(doc_freq))


# ### TF-IDF Matrix

# In[106]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
document_tfidf_matrix = tfidf.fit_transform(bows_lst)


# In[107]:


document_tfidf_matrix


# In[74]:


features = tfidf.vocabulary_
tfidf_matrix = pd.DataFrame(data=document_tfidf_matrix.todense(), columns=features)
# features
# now we have 26,000 features after processing the text


# In[76]:


tfidf_matrix.sample()


# In[114]:


# next steps, reduce number of features
# model
# use VADER to test n-grams
df['review_type'] = np.where(df['overall'] <= 4, 0, 1)


# In[115]:


df['review_type'].value_counts()


# In[90]:


from sklearn.model_selection import train_test_split


# In[118]:


document_tfidf_matrix
len(y)


# ## Modeling

# In[162]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

y = np.array(df['review_type'])
X_train, X_test, y_train, y_test = train_test_split(document_tfidf_matrix, y)
#fit on train
#use predict
#evaluation metric -- f1, confusion matrix, precision/recall
y.shape


# #### Base Model

# In[135]:


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.predict(X_test)
dummy_clf.score(X_test, y_test)


# In[141]:


dummy_clf.score(X_train, y_train)


# #### Logistic Regression Model

# In[136]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)


# In[137]:


lr_y_pred = logistic_regression.predict(X_test)


# In[138]:


from sklearn.metrics import accuracy_score
lr_accuracy = accuracy_score(y_test,lr_y_pred)
lr_accuracy_perc = 100*lr_accuracy
lr_accuracy_perc
#testing accuracy


# In[140]:


logistic_regression.score(X_train, y_train)


# In[139]:


logistic_regression.score(X_test, y_test)


# In[252]:


print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_y_pred))
print("Logistic Regression Classification Report")
print(classification_report(y_test, lr_y_pred))


# #### Gradient Boosting Model

# In[182]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=300, learning_rate=learning_rate, max_features=2, max_depth=60, random_state=0)
    gb_clf.fit(X_train, y_train)
    
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


# In[183]:


gb_clf2 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.5, max_features=2, max_depth=60, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))
gb_clf2.score(X_test, y_test)


# #### Random Forest Model

# In[179]:


rfc = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)


# In[180]:


print(confusion_matrix(y_test,rfc_y_pred))
print(classification_report(y_test,rfc_y_pred))
print(accuracy_score(y_test, rfc_y_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[2]:


Corpus = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/NLP/Training.csv")
Corpus.head()


# In[3]:


len(Corpus)


# In[4]:


Corpus.shape


# In[5]:


Corpus.isnull().any()


# In[12]:


Corpus.sample(5)


# In[14]:


Corpus.columns


# In[17]:


sns.countplot(Corpus.Label)
plt.xlabel("Label")
plt.title("countplot")


# In[ ]:





# In[18]:



# 1. Removing Blank Spaces
Corpus['reviews'].dropna(inplace=True)
# 2. Changing all text to lowercase
Corpus['text_original'] = Corpus['reviews']
Corpus['text'] = [entry.lower() for entry in Corpus['reviews']]
# 3. Tokenization-In this each entry in the corpus will be broken into set of words
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
# 4. Remove Stop words, Non-Numeric and perfoming Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

Corpus.head()


# In[19]:


for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)


# In[20]:


Corpus.drop(['text'], axis=1)
output_path = 'preprocessed_data_MS.csv'
Corpus.to_csv(output_path, index=False)


# In[22]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Label'],test_size=0.1)


# In[23]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[28]:


Tfidf_vect = TfidfVectorizer(max_features=3000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Tfidf_vect.vocabulary_)
#print(len(Tfidf_vect.vocabulary_))


# In[29]:


print(len(Tfidf_vect.vocabulary_))


# In[30]:


print(Train_X_Tfidf)


# In[31]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[33]:


#print(classification_report(Test_Y,predictions_SVM))


# In[34]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# In[ ]:





# In[37]:


Corpus.head()


# In[ ]:





# In[ ]:





# In[ ]:


preprocessed_data_MS


# In[38]:


data = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/NLP/preprocessed_data_MS.csv")
data.head()


# In[39]:


Corpus.drop(['text_final'], axis=1)
output_path = 'preprocessed_Text.csv'
Corpus.to_csv(output_path, index=False)


# In[41]:


pre = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/NLP/preprocessed_Text.csv")
pre.head()


# In[47]:


da = pre.drop(["reviews", "text_original", "text"], axis = 1)

da.head()


# In[48]:


output_path = 'Preprocess.csv'
da.to_csv(output_path, index=False)


# In[ ]:





# In[49]:


df = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/NLP/preprocess.csv")
df.head()


# In[ ]:





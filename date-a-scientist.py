#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

profiles = pd.read_csv('profiles.csv')

#for i in range(31):
#    print(profiles.columns[i], profiles.iloc[:,i].nunique())
    
profiles["sex_code"] = profiles["sex"].map({"m":0, "f":1})
profiles["orientation_code"] = profiles["orientation"].map({"straight":0, "bisexual":1, "gay":2})
profiles["status_code"] = profiles["status"].map({"single":0, "available":1, "seeing someone":2, "married":3, "unknown":4})
profiles['age_group'] = profiles["age"].apply(lambda x: 0 if x > 18 and x <= 30 else 1 if x > 30 and x <= 60 else 2)
profiles['diet'] = profiles["diet"].apply(lambda x: "anything" if "anything" in str(x) else "vegetarian" if "vegetarian" in str(x) else "vegan" if "vegan" in str(x) else "halal" if "halal" in str(x) else "kosher" if "kosher" in str(x) else "other")
profiles['num_lang'] = profiles["speaks"].apply(lambda x: "1" if "," not in str(x) else "2" if str(x).count(",") == 1 else "3" if str(x).count(",") == 2 else "4" if str(x).count(",") == 3 else "4+")
profiles['offspring'] = profiles["offspring"].apply(lambda x: "no info" if pd.isna(x) == True else "has kid(s)" if "has" in str(x) else "doesn't have kid(s)")
profiles['religion'] = profiles["religion"].apply(lambda x: "no info" if pd.isna(x) == True else "christian" if "catholicism" in str(x) or "christianity" in str(x) else "atheist" if "atheism" in str(x) or "agnosticism" in str(x) else "muslim" if "islam" in str(x) else "hindu" if "hinduism" in str(x) else "buddhist" if "buddhism" in str(x) else "jewish" if "judaism" in str(x) else "other")
profiles['education_status'] = profiles["education"].apply(lambda x: "no info" if pd.isna(x) == True or "space" in str(x) else "graduate" if "graduated" in str(x) else "drop out" if "dropped" in str(x) else "high schooler" if "high" in str(x) else "immatriculated")


# In[4]:


from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

#Clustering based on sex, orientation, status, age
classifier = KMeans(n_clusters=6)
classifier.fit(profiles[["sex_code", "orientation_code", "status_code", "age_group"]])

cluster_0_indices = []
cluster_1_indices = []
cluster_2_indices = []
cluster_3_indices = []
cluster_4_indices = []
cluster_5_indices = []
for i in range(len(classifier.labels_)):
    if classifier.labels_[i] == 0:
        cluster_0_indices.append(i)
    if classifier.labels_[i] == 1:
        cluster_1_indices.append(i)
    if classifier.labels_[i] == 2:
        cluster_2_indices.append(i)
    if classifier.labels_[i] == 3:
        cluster_3_indices.append(i)
    if classifier.labels_[i] == 4:
        cluster_4_indices.append(i)
    if classifier.labels_[i] == 5:
        cluster_5_indices.append(i)

profiles.loc[cluster_0_indices, 'cluster'] = 0
profiles.loc[cluster_1_indices, 'cluster'] = 1
profiles.loc[cluster_2_indices, 'cluster'] = 2
profiles.loc[cluster_3_indices, 'cluster'] = 3
profiles.loc[cluster_4_indices, 'cluster'] = 4
profiles.loc[cluster_5_indices, 'cluster'] = 5
profiles.cluster = profiles.cluster.astype(int)


# In[5]:


import seaborn as sns

tab_age = pd.crosstab(profiles.age_group, profiles.cluster, normalize='columns')
tab_sex = pd.crosstab(profiles.sex, profiles.cluster, normalize='columns')
tab_orientation = pd.crosstab(profiles.orientation, profiles.cluster, normalize='columns')
tab_status = pd.crosstab(profiles.status, profiles.cluster, normalize='columns')
tab_diet = pd.crosstab(profiles.diet, profiles.cluster, normalize='columns')
tab_drinks = pd.crosstab(profiles.drinks, profiles.cluster, normalize='columns')
tab_drugs = pd.crosstab(profiles.drugs, profiles.cluster, normalize='columns')
tab_job = pd.crosstab(profiles.job, profiles.cluster, normalize='columns')
tab_smokes = pd.crosstab(profiles.smokes, profiles.cluster, normalize='columns')
tab_lang = pd.crosstab(profiles.num_lang, profiles.cluster, normalize='columns')
tab_body = pd.crosstab(profiles.body_type, profiles.cluster, normalize='columns')
tab_kids = pd.crosstab(profiles.offspring, profiles.cluster, normalize='columns')
tab_rel = pd.crosstab(profiles.religion, profiles.cluster, normalize='columns')
tab_edu = pd.crosstab(profiles.education_status, profiles.cluster, normalize='columns')

sns.heatmap(tab_age, annot=True, yticklabels=["18 - 30", "31 - 60", "60+"], cmap="Blues")
plt.show()
sns.heatmap(tab_sex, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_orientation, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_status, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_diet, annot=True, cmap="Blues")
plt.show()
sns.heatmap(tab_drinks, annot=True, cmap="Blues")
plt.show()
sns.heatmap(tab_drugs, annot=True, cmap="Blues")
plt.show()
sns.heatmap(tab_job, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_smokes, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_lang, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_body, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_kids, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_rel, annot=True, yticklabels=True, cmap="Blues")
plt.show()
sns.heatmap(tab_edu, annot=True, yticklabels=True, cmap="Blues")
plt.show()


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer

cities = []
for i in range(len(profiles)):
    cities.append(profiles.location[i].split(",")[0])

essays = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]
corpus = []
for i in range(len(profiles)):
    for essay in essays:
        corpus.append(str(profiles.iloc[i][essay]))


# In[3]:


words = []

for i in range(0, 599460, 10):
    slice = corpus[i:i+10]
    words.append(" ".join(slice).replace('<br />\n', ' ').replace('\n', ' '))
    
vectorizer = CountVectorizer(ngram_range=(2, 2))
user_word_counts = vectorizer.fit_transform(words)


# In[4]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(user_word_counts, cities, test_size=0.2, random_state=1)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[5]:


print(classifier.score(X_test, y_test))
# This model has a 50% accuracy when classifying a user to a city (in California) by the essays they have written for their online dating profile.


#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import seaborn as sns
#%%

#%%
train = pd.read_csv('../datasets/csvs/train.csv')
test = pd.read_csv('../datasets/csvs/test.csv')
#%%

#%%
print(type(train))
train.head(5)
#%%

#%%
y = train.target.value_counts()

plt.bar([0, 1], y, width=0.8, color=['#28C9FF', '#FF3818'])
plt.title('#Normal tweets vs #Emergency tweets')
plt.show()
#%%

#%%
figure, (plt1, plt2) = plt.subplots(1, 2, figsize=(20, 10))
nodis_len = train[train['target']==0]['text'].str.len()
dis_len = train[train['target']==1]['text'].str.len()

plt1.set_title('normal tweet')
plt2.set_title('disaster tweet')
plt1.hist(nodis_len, color='blue')
plt2.hist(dis_len, color='red')

figure.suptitle('tweet character count')

plt.show()
#%%

#%%
figure, (plt1, plt2) = plt.subplots(1, 2, figsize=(20, 10))
nodis_len = train[train['target']==0]['text'].str.split().map(lambda x: len(x))
dis_len = train[train['target']==1]['text'].str.split().map(lambda x: len(x))

plt1.set_title('normal tweet')
plt2.set_title('disaster tweet')
plt1.hist(nodis_len, color='blue')
plt2.hist(dis_len, color='red')

figure.suptitle('tweet word count')

plt.show()
#%%

#%%
figure, (plt1, plt2) = plt.subplots(1, 2, figsize=(20, 10))
nodis_len = train[train['target']==0]['text'].str.split().apply(lambda x: [len(i) for i in x])
dis_len = train[train['target']==1]['text'].str.split().apply(lambda x: [len(i) for i in x])

plt1.set_title('normal tweet')
plt2.set_title('disaster tweet')

sns.distplot(nodis_len.map(lambda x: np.mean(x)), ax=plt1, color='green')
sns.distplot(dis_len.map(lambda x: np.mean(x)), ax=plt2, color='red')

figure.suptitle('word length average')
#%%

#%%
train.head(50)
#%%
## **NLP PROJECT**
**Note** was built using vscode interactive mode; comments are left in for convenience if vscode interactive mode is available to user

### **Overview**
______________________________

Building a machine learning model to predict which Tweets are about real disasters and which ones are false alarms.

Perform data analysis on 'train.csv' dataset
Perform general cleaning and further cleaning according to observations from data analysis
Remove irrelevant columns
Create model
Find validation split
Feed training data to model
Apply weights to 'test.csv' dataset

### **datavisual.py**
______________________________

Use this file for exploratory data analysis
ie. making charts and shit to get insights

Objective: to figure out a way to process Tweets ~~without the bullshit~~ and discover trends to assist in cleaning

### **model.py**
______________________________

The actual Tensorflow learning model built on BERT

### **tools.py**
______________________________

Any kind of modular function to be imported

Contains primary cleaning functions

**Note** Might split this up later into model tools and EDA tools

SIDE NOTE: contains dictionary of acronyms/shortened words to be replaced, if you can come up with any acronym or shortened word that isn't in the list you can add it to the dictionary directly. still incapable of dividing separate but connected words (ex. suchas -> such as) without a very brute forcey and specific approach

### **csvs**
______________________________

CSV files provided by Kaggle

### **test_clean.pkl & train_clean.pkl**
______________________________

Pickle file from which cleaned train and test csvs are located (test csv is not cleaned, just has some extraneous columns dropped)

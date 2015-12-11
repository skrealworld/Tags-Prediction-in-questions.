import time
tos = time.clock() #Get the start time log. 


import nltk
import pickle
import data_handleing as dh   #Another script to import data and get most frequent tags. 
import numpy as np
import random


from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from random import randint
from colorama import Fore, Back, Style   # library to color commoand line outputs. 
from bs4 import BeautifulSoup  #library to clean HTML data
#from sklearn.linear_model import SGDClassifier 
#from sklearn.feature_extraction.text import HashingVectorizer
#from joblib import Parallel, delayed


no_of_training = 10000
no_of_testing = 300


trainData = dh.getTrainData(no_of_training)

print("Number of training samples is :- ", no_of_training)


train_TitleData_list = []


#Store features as list. 
for x in range(no_of_training):

	 #Preprocessing part

	 temp_list = 3*trainData[x][1] + 4*(BeautifulSoup(trainData[x][2],"html.parser").text)# + 2*trainData[x][3] 

	 #temp_list = 4*list1[x][2] + 3*list1[x][1] + 2*list1[x][3] 

	 train_TitleData_list.append(temp_list)



#Convert list to numpy array.

X_train = np.array(train_TitleData_list)

train_TitleLabel_list = []

for x in range(no_of_training):
	temp = trainData[x][3]

	temp_tokens = word_tokenize(temp)

	train_TitleLabel_list.append(temp_tokens)


y_train_text = train_TitleLabel_list


del trainData

#Testing file contains 299,999 samples
testData = dh.getTestData(no_of_testing)

print("Number of testing samples is :- ", no_of_testing)

test_TitleData_list = []

for x in range(no_of_testing):

	 temp_list =  testData[x][1] + testData[x][2] 

	 test_TitleData_list.append(temp_list)


X_test = np.array(test_TitleData_list)


#Get the target tags. 
test_TitleLabel_list = []
for x in range(no_of_testing):

	temp = testData[x][3]
	temp_tokens = word_tokenize(temp)

	test_TitleLabel_list.append(temp_tokens)

target_tags = test_TitleLabel_list

del testData

#This line can give you top N most frequent tags in the given data. 
#target_names = dh.getTags(2000)

#Represent the Y data also as numbers
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(y_train_text)



#Define pipeline for classifier,
classifier = Pipeline([
    ('tfidf_vec', TfidfVectorizer(min_df=50, ngram_range =(1,3))),
    ('tfidf_trans', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])


# Baseline - 2 
'''
classifier = Pipeline([
    ('tfidf_vec', TfidfVectorizer(min_df=50,max_features =450)),
    ('tfidf_trans', TfidfTransformer()),
    ('clf', OneVsRestClassifier(SGDClassifier(loss='log',n_jobs = -1)))])

'''



#Baseline - 1
'''
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])


'''

#Train the model, we can save the trained model and load it further to test on other data. 
classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)


for item,targets, labels in zip(X_test, target_tags, all_labels):
      print(" \n \n ------------------------------Next Test Sample-------------------------\
      \n Test Sample => \n %s \n \n \033[31m \n TARGETED TAGS => %s \n \n \033[32m \n PREDICTED TAGS => %s" %(item, targets, ', '.join(labels)))
      print(Style.RESET_ALL)



#Calculate accuracies. 

arr1 = [] 
arr2 = []

for x in range(no_of_testing):
	if len(set(target_tags[x])&set(list(all_labels[x])))>=1:
         arr1.append(1)
	else:
         arr1.append(0)


for x in range(no_of_testing):
	if len(set(target_tags[x])&set(list(all_labels[x])))>=2:
         arr2.append(1)
	else:
         arr2.append(0)


print("Overall accuray to predict 1 or more tags is :- \033[31m ", sum(arr1)*100/len(arr1), "% \033[37m \n")

print("Overall accuray to predict 2 or more tags is :- \033[31m ", sum(arr2)*100/len(arr2), "%  \033[37m ")

toe = time.clock()

print("\nTook \033[31m ", (toe-tos) , " \033[37m seconds to execute.")

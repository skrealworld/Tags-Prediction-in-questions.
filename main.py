import nltk
import pickle
import data_handleing as dh
import numpy as np
import random

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from random import randint
from colorama import Fore, Back, Style



def titleLengthFeature(data):

	
	length = len(data)

	titleLength = []

	for x in range(0,length-1):
		#Get the tokens 
		tokens = nltk.word_tokenize(data[x][1])
		titleLength.append(len(tokens)) 		
		
	
	return titleLength



'''
def titleTfidfFeatures(data):

	wholeData = data['trainData'] + data['testData']

	wholeTfidf = {}

def tokenize(text):
	tokens = nltk.word_tokenize(text)


return tokens
          
from sklearn.feature_extraction.text import TfidfVectorizer
         tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
         tfs = tfidf.fit_transform(data['trainData'])

          wholeTfidf['trainData'] = tfs


          tfs = tfidf.fit_transform(data['testData'])


          wholeTfidf['testData'] = tfs
	
 	return wholeTfidf


'''


# import the training data
list1 = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

random.shuffle(list1)

#decide the amount of data you want to use for training. 
train_Length = len(list1)//1000

print("Number of training samples is :- ", train_Length)


train_TitleData_list = []

#Store features as list. 
for x in range(train_Length):

	 #temp = list1[x][1]+list1[x][]

	 
	 temp_list = 3*list1[x][1] + list1[x][2] + 10*list1[x][3] 


	 train_TitleData_list.append(temp_list)



#Convert list to numpy array.

X_train = np.array(train_TitleData_list)

train_TitleLabel_list = []

for x in range(train_Length):
	temp = list1[x][3]

	temp_tokens = word_tokenize(temp)

	train_TitleLabel_list.append(temp_tokens)


y_train_text = train_TitleLabel_list


#Testing file contains 299,999 samples
list2 = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/testSubData.p","rb"))

random.shuffle(list2)

test_Length = len(list2)//1000

print("Number of testing samples is :- ", test_Length)

test_TitleData_list = []

for x in range(test_Length):

	 temp_list = list2[x][1] + list2[x][2] 

	 test_TitleData_list.append(temp_list)


X_test = np.array(test_TitleData_list)


#Get the target tags. 
test_TitleLabel_list = []
for x in range(test_Length):

	temp = list2[x][3]
	temp_tokens = word_tokenize(temp)

	test_TitleLabel_list.append(temp_tokens)

target_tags = test_TitleLabel_list

#target_names = dh.getTags(2000)


lb = preprocessing.LabelBinarizer()


Y = lb.fit_transform(y_train_text)


classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])


classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)



for item,targets, labels in zip(X_test, target_tags, all_labels):
      print(" \n \n ------------------------------Next Test Sample-------------------------\
      \n Test Sample => \n %s \n \n \033[31m \n TARGETED TAGS => %s \n \n \033[32m \n PREDICTED TAGS => %s" %(item, targets, ', '.join(labels)))
      print(Style.RESET_ALL)


arr1 = [] 
arr2 = []

for x in range(test_Length):
	if len(set(target_tags[x])&set(list(all_labels[x])))>=1:
         arr1.append(1)
	else:
         arr1.append(0)


for x in range(test_Length):
	if len(set(target_tags[x])&set(list(all_labels[x])))>=2:
         arr2.append(1)
	else:
         arr2.append(0)




print("Overall accuray to predict 1 or more tags is :- \033[31m ", sum(arr1)*100/len(arr1), "%" "\033[37m \n")

print("Overall accuray to predict 2 or more tags is :- \033[31m ", sum(arr2)*100/len(arr2), "%" "\033[37m ")



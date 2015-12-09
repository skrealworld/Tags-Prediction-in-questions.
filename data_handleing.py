'''
This script is to deal with huge data of questions and related tags. 

Author - Sourabh Kulhare
'''

#This is the data handeling script for project, 
#function = getTrainData(nSample) = return = list of traning samples. 
#function = getTestData(nSample) = return = list of testing samples. 
#all of these functions includes preprocessing of data also. 

#1,000,000 training samlpes. 
#30,000 testing examples.
#30,000 totally clean samples to test at the end. 
#load & save as pickle file pickle.dump(trainSubData,open("trainSubData.p","wb"))

#                           Import training .csv file. 
#csvfileTr = open('/Users/SK_Mac/Documents/Github/NLP_data/Train.csv', 'r')
#trainreader = csv.reader(csvfileTr)
#data = list(trainreader);





import csv
from itertools import islice
import pickle
import random

#Function to import whole data at once and return a dictionary. 

def getData():

    data = {}

    data['trainData'] = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

    data['testData']  = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/testSubData.p","rb"))

    return data



def getTrainData(N):

    """
    Convert reader object into list. 
    ID = [i][0]
    Title of question = [i][1]
    Body of question = [i][2]
    Tags = [i][3] 
    """

    trainData = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

    random.shuffle(trainData)

    data = []


    for x in range(N):
        data.append(trainData[x])




    #perform pre processing tasks and cleaning the data.
    
    return data



def getTestData(N):

    """
    Convert reader object into list. 
    ID = [i][0]
    Title of question = [i][1]
    Body of question = [i][2]
    Tags = [i][3] 
    """

    testData = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

    random.shuffle(testData)

    data = []


    for x in range(N):
        data.append(testData[x])


    #perform pre processing tasks and cleaning the data.
    
    return data




def getTags(N):
    
    import pickle
    from nltk import word_tokenize

    from nltk import FreqDist

    list1 = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

    list2 = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/testSubData.p","rb"))

    list3 = list1 + list2

    length = len(list3)

    tagText = ""

    for x in range(length):
        tagText = tagText + list3[x][3]
 

    tokens = word_tokenize(tagText)

    fdist = FreqDist(tokens)

    topNTags = fdist.most_common(N+1)

    l = len(topNTags)

    nTags = []

    for x in range(1,l):
        nTags.append(topNTags[x][0])

    return nTags




def titleLengthFeature(data):
    
    length = len(data)

    titleLength = []

    for x in range(0,length-1):
        #Get the tokens 
        tokens = nltk.word_tokenize(data[x][1])
        titleLength.append(len(tokens))         
        
    
    return titleLength












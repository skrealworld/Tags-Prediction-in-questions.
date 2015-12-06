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


#Function to import whole data at once and return a dictionary. 

def getData():

    data = {}

    data['trainData'] = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

    data['testData']  = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/testSubData.p","rb"))

    return data
 



def getTrainData():


    trainData = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

    """
    Convert reader object into list. 
    ID = [i][0]
    Title of question = [i][1]
    Body of question = [i][2]
    Tags = [i][3] 
    """

    #perform pre processing tasks and cleaning the data.
    
    return trainData



def getTestData():
    
    
    testData = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/testSubData.p","rb"))

    """
    Convert reader object into list. 
    ID = [i][0]
    Title of question = [i][1]
    Body of question = [i][2]
    Tags = [i][3]  // Only for training data, for testing data we don't have tags. 

    """


    #perform pre processing tasks and cleaning the data.
    return testData




data = {}

list1 = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/trainSubData.p","rb"))

list2 = pickle.load(open("/Users/SK_Mac/Documents/Github/NLP_data/testSubData.p","rb"))

list3 = list1 + list2

length = len(list3)








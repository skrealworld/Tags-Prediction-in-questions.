'''
This script is to deal with huge data of questions and related tags. 

Author - Sourabh Kulhare
'''


#This is the data handeling script for project, 
#function = getTrainData(nSample) = return = list of traning samples. 
#function = getTestData(nSample) = return = list of testing samples. 
#all of these functions includes preprocessing of data also. 

import csv
from itertools import islice




def getTrainData():


    trainData = []

    # Import training .csv file. 
    csvfileTr = open('/Users/SK_Mac/Documents/Github/NLP_data/Train.csv', 'r')
    trainreader = csv.reader(csvfileTr)

    """
    Convert reader object into list. 
    ID = [i][0]
    Title of question = [i][1]
    Body of question = [i][2]
    Tags = [i][3]  // Only for training data, for testing data we don't have tags. 
    """

    data = list(trainreader);

    #divide data into training and testing

    trainData = []
    testData = []


    for x in range(0,20000):
        trainData.appand(data[x])



return trainData



def getTestData():
    testData = []
    
    # Import testing .csv file. 
    csvfileTs = open('/Users/SK_Mac/Documents/Github/NLP_data/Test.csv', 'r')

    testreader = csv.reader(csvfileTs)

    """
    Convert reader object into list. 
    ID = [i][0]
    Title of question = [i][1]
    Body of question = [i][2]
    Tags = [i][3]  // Only for training data, for testing data we don't have tags. 

    """


    for x in range(0,1000):
        testData.appand(data[x])



return testData




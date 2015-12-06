#import data_handleing as data
import nltk





def titleLengthFeature(data):

	
	length = len(data)

	titleLength = []

	for x in range(0,length-1):
		#Get the tokens 
		tokens = nltk.word_tokenize(data[x][1])
		titleLength.append(len(tokens)) 		
		
	
	return titleLength






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









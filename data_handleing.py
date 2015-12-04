'''
This script is to deal with huge data of questions and related tags. 

Author - Sourabh Kulhare
'''

import csv
from itertools import islice


csvfile = open('/Users/SK_Mac/Documents/Github/NLP_data/Test.csv', 'r')

testreader = csv.reader(csvfile)

for x in range(1,3):
	print(testreader[x])




def line_stream(filename, stop=None):
    """Streams `filename` one line at a time, optionally stopping at `stop`"""
    with open(filename) as csvfile:
        next(csvfile, None)  # skip header
        for line in islice(csv.reader(csvfile, delimiter=',', quotechar='"'), stop):
            yield line
            
def title_tokenize(s, stopwords=STOPWORDS):
    """Extract valid SO tags style tokens, from string `s` excluding tokens in `STOPWORDS`"""
    return [token for token in re.findall(r'\b\w[\w#+.-]*(?<!\.$)', s.lower())
                    if token not in stopwords]






from collections import Counter

def basic_recommender(title, usefulness=usefulness):
    return Counter({word: usefulness.get(word, 0)
                  for word in title_tokenize(title)})
    









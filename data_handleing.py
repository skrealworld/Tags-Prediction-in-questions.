'''
This script is to deal with huge data of questions and related tags. 

Author - Sourabh Kulhare
'''

import csv
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


'''
Generic script for classifier:
Classifier = SVM linear model
It takes four parameters as input:
    a(list) = training data which is dictionary of features
    b(np.array) = ground truth of features(associated class)
    c(list) = test data which is dictionay of features
    d(np.array) = ground truth of test data
'''

def svm_Classifier(wholeData):

	a = wholeData['trainData'][]


    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #define test data
    X_data_ts = v1.transform(c)

    #import linear SVM
    from sklearn.svm import LinearSVC

    #generate model
    svm_Classifier = LinearSVC().fit(X_data_tr, Y_data_tr)

    #Use trained model to classify test data
    Y_pred = svm_Classifier.predict(X_data_ts)

    return Y_pred


def get_Naivebayes_Acc(a,b,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #define test data
    X_data_ts = v1.transform(c)

    #import Naive bayes classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_data_tr,Y_data_tr)

    #Use trained model to classify test data
    Y_pred = clf.predict(X_data_ts)

    return Y_pred
    #print(len(clf.classes_))
    #print(clf.classes_)
    #most_informative_feature_for_class(v1,clf, clf.classes_[0])
    #most_informative_feature_for_class(v1,clf, clf.classes_[1])

    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(Y_data_ts,Y_pred))


def get_LinearRegression_Acc(a,b,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #define test data
    X_data_ts = v1.transform(c)

    #import Linear Regression classifier
    import numpy as np
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X_data_tr,Y_data_tr)

    #Use trained model to classify test data
    Y_pred = regr.predict(X_data_ts)
    # Convert into nearest integer
    Y_pred = np.rint(Y_pred)

    return Y_pred

    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(Y_data_ts,Y_pred))


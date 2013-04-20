import datas
from sklearn.svm import SVC
from sklearn import cross_validation

def main():
    training = datas.training
    testing  = datas.testing
    classifier = SVC(kernel='linear',C=1.0)
    classifier.fit(training.ix[:,'pclass':],training.ix[:,'survived'])
    result = classifier.predict(testing)
    print result

def test():
    training = datas.training
    svc = SVC(kernel='linear',C=1.0)
    training = training.drop(['embarked'],axis=1)
    kfold    = cross_validation.KFold(len(training), 3)
    result   = cross_validation.cross_val_score(svc, training.ix[:,'pclass':], training.ix[:,'survived'], cv=kfold,n_jobs=1)
    print result


if __name__ == "__main__":
    test()
    #main()

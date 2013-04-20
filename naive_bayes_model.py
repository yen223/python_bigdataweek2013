import datas
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

def main():
    '''
    Blah blah blah
    '''
    training = datas.training
    training = training.drop(['embarked'],axis=1)
    testing  = datas.testing
    testing = testing.drop(['embarked'],axis=1)
    gnb = GaussianNB()
    gnb = gnb.fit(training.ix[:,'pclass':],training.ix[:,'survived'])
    result = gnb.predict(testing)
    print result

def test():
    training = datas.training
    gnb = GaussianNB()
    training = training.drop(['embarked'],axis=1)
    kfold    = cross_validation.KFold(len(training), 3)
    result   = cross_validation.cross_val_score(gnb, training.ix[:,'pclass':], training.ix[:,'survived'], cv=kfold,n_jobs=1)
    print result


if __name__ == "__main__":
    test()

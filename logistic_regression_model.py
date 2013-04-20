import datas
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

def main():
    training = datas.training
    training = training.drop(['embarked'],axis=1)
    testing  = datas.testing
    testing = testing.drop(['embarked'],axis=1)
    logreg = LogisticRegression()
    logreg = logreg.fit(training.ix[:,'pclass':],training.ix[:,'survived'])
    result = logreg.predict(testing)
    print result

def test():
    logreg = LogisticRegression()
    training = datas.training
    training = training.drop(['embarked','sibsp','fare'],axis=1)
    kfold    = cross_validation.KFold(len(training), 3)
    result   = cross_validation.cross_val_score(logreg, training.ix[:,'pclass':], training.ix[:,'survived'], cv=kfold,n_jobs=1)
    print result

if __name__ == "__main__":
    test()

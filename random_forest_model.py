import datas
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def main():
    training = datas.training
    testing  = datas.testing
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest = random_forest.fit(training.ix[:,'pclass':],training.ix[:,'survived'])
    result = random_forest.predict(testing)
    print result

def test():
    training = datas.training
    random_forest = RandomForestClassifier(n_estimators=100)
    training = training.drop(['embarked','sibsp','fare' ],axis=1)
    kfold    = cross_validation.KFold(len(training), 3)
    result   = cross_validation.cross_val_score(random_forest, training.ix[:,'pclass':], training.ix[:,'survived'], cv=kfold,n_jobs=1)
    print result


if __name__ == "__main__":
    test()

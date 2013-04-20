import pandas


training = pandas.read_csv("data/train.csv")
testing  = pandas.read_csv("data/test.csv")

def clean_data(data_frame):
    data_frame = data_frame.drop(
        ['name','cabin','ticket','parch'],axis=1)
    data_frame['sex'] = data_frame['sex'].replace(
        ['male','female'], [1,0])
    data_frame['embarked'] = data_frame['embarked'].replace(
       ['C','S','Q'], [0,1,2])
    
    data_frame['age'] = data_frame['age'].fillna(data_frame['age'].mean())
    data_frame['embarked'] = data_frame['embarked'].fillna(
        data_frame['embarked'].mean())

    data_frame['fare'] = data_frame['fare'].fillna(
        data_frame['fare'].mean())

    return data_frame

training = clean_data(training)
testing = clean_data(testing)

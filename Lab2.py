import hashlib
import pandas as pd
import numpy as np
from User import UserModel
from flask import Flask
from flask import jsonify
from flask import request
from datetime import datetime

IDToUserModel = {}

def splitByID(train, test):
    userIDs = train['userId'].unique().tolist()
    users = []
    for userID in userIDs:
        userTrainTransactions = train.loc[train.userId == userID]
        userTestTransactions = test.loc[test.userId == userID]
        trainUserModel = UserModel(userID, userTrainTransactions)
        testUserModel = UserModel(userID, userTestTransactions)
        users.append((trainUserModel, testUserModel))

    return users


def calculateCategoriesHash(row):
    category = row['category']
    sha1 = hashlib.sha1()
    sha1.update(str(category))
    return sha1.hexdigest()

def splitDataset(allTransactions):
    # randomly select 80% of the dataset to be train dataset and the rest to be test dataset
    df = pd.DataFrame(np.random.randn(len(allTransactions), 2))
    mask = np.random.rand(len(df)) < 0.8
    train = allTransactions[mask]
    test = allTransactions[~mask]
    return train, test, allTransactions

if __name__ == '__main__':
    print "running models creation using the file transactions_clean.txt that was published in moodle"
    allTransactions = pd.read_json("transactions_clean.txt")
    allTransactions['categoryHash'] = allTransactions.apply(calculateCategoriesHash, axis=1)
    train, test, allTransactions = splitDataset(allTransactions)

    users = splitByID(train, test)
    print "detected ", len(users), " users."
    i = 0
    for (trainUserModel, testUserModel) in users:
        i += 1
        print i, ") training model for user ", trainUserModel._userID
        trainUserModel.extractFeatures()
        testUserModel.extractFeatures()
        trainUserModel.trainModels(testUserModel)
        IDToUserModel[trainUserModel._userID] = trainUserModel

app = Flask(__name__)


@app.route('/', methods=['POST'])
def analyze():
    transaction = request.get_json()['trans']
    transaction = pd.DataFrame.from_dict([transaction]).iloc[0]
    userID = transaction['userId']
    category = transaction['category']
    sha1 = hashlib.sha1()
    sha1.update(str(category))
    transaction['categoryHash'] = sha1.hexdigest()
    dateType = type(transaction['date'])
    if dateType == str or dateType == unicode:
        transaction['date'] = datetime.strptime(transaction['date'], "%Y-%m-%d")
    if IDToUserModel.has_key(userID):
        userModel = IDToUserModel[userID]
    else:
        userID = IDToUserModel.keys()[0]
        userModel = IDToUserModel[userID]

    result = userModel.predict(transaction)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False)



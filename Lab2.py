import hashlib
import pandas as pd
import numpy as np
from User import UserModel


# TODO test query for missing user
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
    df = pd.DataFrame(np.random.RandomState(seed=2018).randn(len(allTransactions), 2)) #TODO remove seed
    mask = np.random.rand(len(df)) < 0.8
    train = allTransactions[mask]
    test = allTransactions[~mask]
    return train, test, allTransactions



allTransactions = pd.read_json("transactions_clean.txt")
allTransactions['categoryHash'] = allTransactions.apply(calculateCategoriesHash, axis=1)
train, test, allTransactions = splitDataset(allTransactions)

users = splitByID(train, test)
for (trainUserModel, testUserModel) in users:
    # print "training model for "
    trainUserModel.extractFeatures()
    testUserModel.extractFeatures()
    trainUserModel.trainModels(testUserModel)

print ""



import hashlib
import pandas as pd
from User import User


def splitByID(transactions):
    userIDs = transactions['userId'].unique().tolist()
    users = []
    for userID in userIDs:
        userTransactions = transactions.loc[transactions.userId == userID]
        newUser = User(userID, userTransactions)
        users.append(newUser)

    return users


def calculateCategoriesHash(row):
    category = row['category']
    sha1 = hashlib.sha1()
    sha1.update(str(category))
    return sha1.hexdigest()


transactions = pd.read_json("transactions_clean.txt")
transactions['categoryHash'] = transactions.apply(calculateCategoriesHash, axis=1)
users = splitByID(transactions)
for user in users:
    user.addIncomeFeatures()
    user.labelSubscriptionTransactions()
    user.addSubscriptionFeatures()

print ""



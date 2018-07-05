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


transactions = pd.read_json("transactions_clean.txt")
users = splitByID(transactions)
for user in users:
    user.addFeatures()

print ""



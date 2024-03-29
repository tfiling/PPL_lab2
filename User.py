from datetime import timedelta
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.svm import SVC
from sklearn import tree

###############################################
# general porpuse
CACHE_FILE = 'CACHE_FILE.csv'
DAY_OF_MONTH = 'dayOfMonth'
MONTH = 'month'
YEAR = 'year'


###############################################
# income feature column names

IS_INCOME = 'isIncome'
PAST_MONTH_INCOME_COUNT = 'pastMonthIncomeCount'
PAST_WEEK_INCOME_COUNT = 'pastWeekIncomeCount'
PAST_MONTH_INCOME = 'pastMonthIncome'
PAST_WEEK_INCOME = 'pastWeekIncome'
PAST_MONTH_MIN = 'pastMonthMin'
PAST_WEEK_MIN = 'pastWeekMin'
PAST_MONTH_MAX = 'pastMonthMax'
PAST_WEEK_MAX = 'pastWeekMax'
PAST_MONTH_MEAN = 'pastMonthMean'
PAST_WEEK_MEAN = 'pastWeekMean'
PAST_MONTH_STD = 'pastMonthStd'
PAST_WEEK_STD = 'pastWeekStd'
CURRENT_MONTH_INCOME_COUNT = 'currentMonthIncomeCount'
CURRENT_WEEK_INCOME_COUNT = 'currentWeekIncomeCount'
CURRENT_MONTH_INCOME = 'currentMonthIncome'
CURRENT_WEEK_INCOME = 'currentWeekIncome'
CURRENT_MONTH_MIN = 'currentMonthMin'
CURRENT_WEEK_MIN = 'currentWeekMin'
CURRENT_MONTH_MAX = 'currentMonthMax'
CURRENT_WEEK_MAX = 'currentWeekMax'
CURRENT_MONTH_MEAN = 'currentMonthMean'
CURRENT_WEEK_MEAN = 'currentWeekMean'
CURRENT_MONTH_STD = 'currentMonthStd'
CURRENT_WEEK_STD = 'currentWeekStd'
TOTAL_MONTH_INCOME = 'totalIncomeByEndOfMonth'
TOTAL_WEEK_INCOME = 'totalIncomeByEndOfWeek'

###############################################
# subscription feature column names

IS_PAST_WEEK_SIMILAR_AMOUNT = 'isPastWeekSimilarAmount'
IS_PAST_MONTH_SIMILAR_AMOUNT = 'isPastMonthSimilarAmount'
IS_PAST_WEEK_SIMILAR_CATEGORY = 'isPastWeekSimilarCategory'
IS_PAST_MONTH_SIMILAR_CATEGORY = 'isPastMonthSimilarCategory'
WEEK_INTERVAL_NAME_SIMILARITY = 'weekIntervalNameSimilarity'
MONTH_INTERVAL_NAME_SIMILARITY = 'monthIntervalNameSimilarity'
WEEK_INTERVAL_AMOUNT_MEAN = 'weekIntervalAmountMean'
MONTH_INTERVAL_AMOUNT_MEAN = 'monthIntervalAmountMean'
WEEK_INTERVAL_AMOUNT_STD = 'weekIntervalAmountStd'
MONTH_INTERVAL_AMOUNT_STD = 'monthIntervalAmountStd'

###############################################
# original data column names

AMOUNT = 'amount'
IS_SUBSCRIPTION = 'subscription'
DATE = 'date'
CATEGORY = 'category'
ID = 'id'
CATEGORY_HASH = 'categoryHash'
NAME = 'name'

class UserModel:

    _userID = None
    _transactions = None
    _cach_file = ''
    _pastMonthFeatures = {}
    _pastWeekFeatures = {}
    _monthToIncome = {}
    _weekToIncome = {}


    def __init__(self, userID, transactions):
        self._userID = userID
        self._originalTransactions = pd.DataFrame(transactions)
        self._transactions = transactions.sort_values(by=DATE)
        self.addUtilityColumns()
        self._monthlyIncomeFeatures = pd.DataFrame(self._transactions)
        self._weeklyIncomeFeatures = pd.DataFrame(self._transactions)
        self._subscriptionFeatures = pd.DataFrame(self._transactions)
        self._useCache = True

    def extractFeatures(self):
        self.addIncomeFeatures()
        self.addSubscriptionFeatures()

################################################################################################
## train model
################################################################################################

    def getIrrelevantColumns(self, additional):
        return ['__v', '_id', CATEGORY_HASH, ID, 'userId', DATE, 'createdAt', 'updatedAt', CATEGORY,
                             'location', 'paymentMeta', IS_SUBSCRIPTION, 'type', 'name', 'accountId', additional]


    def trainModels(self, testUserModel):
        # subscription model
        irrelevantColumns = self.getIrrelevantColumns(IS_SUBSCRIPTION)
        self._subscriptionModel = tree.DecisionTreeClassifier()
        x = self._subscriptionFeatures.drop(irrelevantColumns, axis=1)
        y = self._subscriptionFeatures[IS_SUBSCRIPTION]
        self._subscriptionModel.fit(x, y)

        x1 = testUserModel._subscriptionFeatures.drop(irrelevantColumns, axis=1)
        y1 = testUserModel._subscriptionFeatures[IS_SUBSCRIPTION]
        print "test score for subscription labeling: ", self._subscriptionModel.score(x1, y1)

        # month model
        irrelevantColumns = self.getIrrelevantColumns(TOTAL_MONTH_INCOME)
        self._monthModel = SVC(kernel='rbf')
        x = self._monthlyIncomeFeatures.drop(irrelevantColumns, axis=1)
        y = self._monthlyIncomeFeatures[TOTAL_MONTH_INCOME].apply(lambda x: int(x))
        try:
            self._monthModel.fit(x, y)
        except:
            y.ix[0] = y.ix[0] - 1
            self._monthModel.fit(x, y)

        x1 = testUserModel._monthlyIncomeFeatures.drop(irrelevantColumns, axis=1)
        y1 = testUserModel._monthlyIncomeFeatures[TOTAL_MONTH_INCOME].apply(lambda x: int(x))
        print "test score for monthly income model:", self._monthModel.score(x1, y1)

        # week
        irrelevantColumns = self.getIrrelevantColumns(TOTAL_WEEK_INCOME)
        self._weekModel = SVC(kernel='rbf')
        x = self._weeklyIncomeFeatures.drop(irrelevantColumns, axis=1)
        y = self._weeklyIncomeFeatures[TOTAL_WEEK_INCOME].apply(lambda x: int(x))
        try:
            self._weekModel.fit(x, y)
        except ValueError:
            y.ix[0] = y.ix[0] - 1
            self._weekModel.fit(x, y)

        x1 = testUserModel._weeklyIncomeFeatures.drop(irrelevantColumns, axis=1)
        y1 = testUserModel._weeklyIncomeFeatures[TOTAL_WEEK_INCOME].apply(lambda x: int(x))
        print "test score for weekly income model:", self._weekModel.score(x1, y1)

    def addUtilityColumns(self):
        for index, row in self._transactions.iterrows():
            date = row[DATE]
            date = date.isocalendar()
            self._transactions.ix[index, DAY_OF_MONTH] = date[2]
            self._transactions.ix[index, MONTH] = date[1]
            self._transactions.ix[index, YEAR] = date[0]

################################################################################################
## predict
################################################################################################

    def predict(self, transaction):
        transactionID = transaction['id']
        transactions = self._originalTransactions
        if len(transactions[transactions['id'] == transactionID]) == 0:
            transactions = transactions.append(pd.Series(transaction), ignore_index=True)
        user = UserModel(self._userID, transactions)
        user._useCache = False
        user.extractFeatures()
        subscriptionFeatures = user._subscriptionFeatures[user._subscriptionFeatures[ID] == transactionID]
        irrelevantColumns = self.getIrrelevantColumns(IS_SUBSCRIPTION)
        subscriptionFeatures = subscriptionFeatures.drop(irrelevantColumns, axis=1)
        monthlyIncomeFeatures = user._monthlyIncomeFeatures[user._monthlyIncomeFeatures[ID] == transactionID]
        irrelevantColumns = self.getIrrelevantColumns(TOTAL_MONTH_INCOME)
        monthlyIncomeFeatures = monthlyIncomeFeatures.drop(irrelevantColumns, axis=1)
        weeklyIncomeFeatures = user._weeklyIncomeFeatures[user._weeklyIncomeFeatures[ID] == transactionID]
        irrelevantColumns = self.getIrrelevantColumns(TOTAL_WEEK_INCOME)
        weeklyIncomeFeatures = weeklyIncomeFeatures.drop(irrelevantColumns, axis=1)

        try:
            predictedIsSubscription = bool(self._subscriptionModel.predict(subscriptionFeatures)[0])
        except:
            predictedIsSubscription = False
        try:
            predictedMonthlyIncome = int(abs(self._monthModel.predict(monthlyIncomeFeatures)[0]))
        except:
            predictedMonthlyIncome = 504
        try:
            predictedWeeklyIncome = int(abs(self._weekModel.predict(weeklyIncomeFeatures)[0]))
        except:
            predictedWeeklyIncome = 10

        return {
            "subscription": predictedIsSubscription,
            "weeklyIncome": predictedWeeklyIncome,
            "monthlyIncome": predictedMonthlyIncome
        }



################################################################################################
## income features
################################################################################################

    @staticmethod
    def zeroAllFeatures(keys):
        result = {}
        for key in keys:
            result[key] = 0

        return pd.Series(result)

#######################################
## past month / week features
#######################################

    def calculatePastMonthIncome(self, row):
        date = row[DATE]
        lastDay = date.replace(day=1)
        lastDay = lastDay - timedelta(days=1)
        firstDay = lastDay.replace(day=1)
        if not self._pastMonthFeatures.has_key(firstDay):
            lastMonthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            if len(lastMonthIncome) > 0:
                lastMonthIncome = lastMonthIncome[lastMonthIncome[IS_INCOME]][AMOUNT]
                series = pd.Series(
                    {
                        PAST_MONTH_INCOME: lastMonthIncome.sum(),
                        PAST_MONTH_INCOME_COUNT: len(lastMonthIncome),
                        # the income is negative therefore min reslted from max and vice versa
                        PAST_MONTH_MAX: lastMonthIncome.min(),
                        PAST_MONTH_MIN: lastMonthIncome.max(),
                        PAST_MONTH_MEAN: lastMonthIncome.mean(),
                        PAST_MONTH_STD: lastMonthIncome.std()
                    })
                series = series.fillna(0)
            else:
                series = self.zeroAllFeatures([
                        PAST_MONTH_INCOME,
                        PAST_MONTH_INCOME_COUNT,
                        PAST_MONTH_MAX,
                        PAST_MONTH_MIN,
                        PAST_MONTH_MEAN,
                        PAST_MONTH_STD
                    ])

            self._pastMonthFeatures[firstDay] = series

        result = pd.Series(self._pastMonthFeatures[firstDay])
        result[ID] = row[ID]
        return result

    def calculatePastWeekIncome(self, row):
        date = row[DATE]
        day = date.isocalendar()[2]
        firstDay = date - timedelta(days=day + 7)
        lastDay = date - timedelta(days=day)
        if not self._pastWeekFeatures.has_key(firstDay):
            lastWeekIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            if len(lastWeekIncome) > 0 and len(lastWeekIncome[lastWeekIncome[IS_INCOME]]):
                lastWeekIncome = lastWeekIncome[lastWeekIncome[IS_INCOME]][AMOUNT]
                series = pd.Series(
                    {
                        PAST_WEEK_INCOME: lastWeekIncome.sum(),
                        PAST_WEEK_INCOME_COUNT: len(lastWeekIncome),
                        # the income is negative therefore min reslted from max and vice versa
                        PAST_WEEK_MAX: lastWeekIncome.min(),
                        PAST_WEEK_MIN: lastWeekIncome.max(),
                        PAST_WEEK_MEAN: lastWeekIncome.mean(),
                        PAST_WEEK_STD: lastWeekIncome.std()

                    })
                series = series.fillna(0)
            else:
                series = self.zeroAllFeatures([
                        PAST_WEEK_INCOME,
                        PAST_WEEK_INCOME_COUNT,
                        PAST_WEEK_MAX,
                        PAST_WEEK_MIN,
                        PAST_WEEK_MEAN,
                        PAST_WEEK_STD
                    ])
            self._pastWeekFeatures[firstDay] = series

        result = self._pastWeekFeatures[firstDay]
        result[ID] = row[ID]
        return result

#######################################
## current month / week features
#######################################

    def calculateCurrentMonthIncome(self, row):
        date = row[DATE]
        firstDay = date.replace(day=1)
        monthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= date]
        monthIncome = monthIncome[monthIncome[IS_INCOME]][AMOUNT]
        result = pd.Series(
            {
                CURRENT_MONTH_INCOME: monthIncome.sum(),
                CURRENT_MONTH_INCOME_COUNT: len(monthIncome),
                # the income is negative therefore min reslted from max and vice versa
                CURRENT_MONTH_MAX: monthIncome.min(),
                CURRENT_MONTH_MIN: monthIncome.max(),
                CURRENT_MONTH_MEAN: monthIncome.mean(),
                CURRENT_MONTH_STD: monthIncome.std()
            })
        result = result.fillna(0)

        if not self._monthToIncome.has_key(firstDay):
            lastDay = firstDay + timedelta(days=31)
            lastDay = lastDay.replace(day=1)
            monthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            monthIncome = monthIncome[monthIncome[IS_INCOME]][AMOUNT]
            self._monthToIncome[firstDay] = monthIncome.sum()

        result[TOTAL_MONTH_INCOME] = self._monthToIncome[firstDay]
        result[ID] = row[ID]
        return result

    def calculateCurrentWeekIncome(self, row):
        date = row[DATE]
        day = date.isocalendar()[2]
        firstDay = date - timedelta(days=day - 1)
        weekIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= date]
        weekIncome = weekIncome[weekIncome[IS_INCOME]][AMOUNT]
        series = pd.Series(
            {
                CURRENT_WEEK_INCOME: weekIncome.sum(),
                CURRENT_WEEK_INCOME_COUNT: len(weekIncome),
                # the income is negative therefore min reslted from max and vice versa
                CURRENT_WEEK_MAX: weekIncome.min(),
                CURRENT_WEEK_MIN: weekIncome.max(),
                CURRENT_WEEK_MEAN: weekIncome.mean(),
                CURRENT_WEEK_STD: weekIncome.std()

            })
        result = series.fillna(0)

        if not self._weekToIncome.has_key(firstDay):
            lastDay = date + timedelta(days=7 - day)
            weekIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= date]
            weekIncome = weekIncome[weekIncome[IS_INCOME]][AMOUNT]
            self._weekToIncome[firstDay] = weekIncome.sum()

        result[TOTAL_WEEK_INCOME] = self._weekToIncome[firstDay]
        result[ID] = row[ID]
        return result

    def addIncomeFeatures(self):
        self._transactions[IS_INCOME] = self._transactions.apply(lambda row: row[AMOUNT] < 0, axis=1)

        pastMonthIncomeFeatures = self._transactions.apply(self.calculatePastMonthIncome, axis=1)
        currentMonthIncomeFeatures = self._transactions.apply(self.calculateCurrentMonthIncome, axis=1)
        self._monthlyIncomeFeatures = pd.merge(self._monthlyIncomeFeatures, pastMonthIncomeFeatures, how='inner', on=ID)
        self._monthlyIncomeFeatures = pd.merge(self._monthlyIncomeFeatures, currentMonthIncomeFeatures, how='inner', on=ID)

        pastWeekIncomeFeatures = self._transactions.apply(self.calculatePastWeekIncome, axis=1)
        currentWeekIncomeFeatures = self._transactions.apply(self.calculateCurrentWeekIncome, axis=1)
        self._weeklyIncomeFeatures = pd.merge(self._weeklyIncomeFeatures, pastWeekIncomeFeatures, how='inner', on=ID)
        self._weeklyIncomeFeatures = pd.merge(self._weeklyIncomeFeatures, currentWeekIncomeFeatures, how='inner', on=ID)


################################################################################################
## label subscriptions
################################################################################################

    def getAverageCostAndDateDiff(self, relatedTransactions):

        dateDiffDf = pd.DataFrame([], columns=[ID, DATE])
        costsDf = pd.DataFrame([], columns=[ID, AMOUNT])
        # costsDf.append(relatedTransactions.iloc[0][ID, AMOUNT])
        costsDf.loc[0] = pd.Series({ID: relatedTransactions.iloc[0][ID], AMOUNT: relatedTransactions.iloc[0][AMOUNT]})

        for i in range(1, len(relatedTransactions)):
            currentRow = relatedTransactions.iloc[i]
            prevRow = relatedTransactions.iloc[i - 1]
            costsDf.loc[i] = pd.Series({ID: relatedTransactions.iloc[0][ID], AMOUNT: currentRow[AMOUNT]})
            dateDiff = (currentRow[DATE] - prevRow[DATE]).days
            # dateDiffDf.append({ID: relatedTransactions.iloc[0][ID], DATE: dateDiff})
            dateDiffDf.loc[i] = pd.Series({ID: relatedTransactions.iloc[0][ID], DATE: dateDiff})

        averageCost = costsDf[AMOUNT].mean()
        averageDateDiff = dateDiffDf[DATE].mean()
        dateDiffStd = dateDiffDf[DATE].std()

        return averageCost, averageDateDiff, dateDiffStd


    def isSubscription(self, row):
        if row[IS_INCOME]:
            # incomes are not subscription
            return False

        if not np.isnan(row[IS_SUBSCRIPTION]):
            # preseve the previous calculation from a related transaction
            return row[IS_SUBSCRIPTION]

        categories = row[CATEGORY_HASH]
        relatedTransactions = self._subscriptionFeatures[self._subscriptionFeatures[CATEGORY_HASH] == categories]
        relatedTransactions = relatedTransactions.sort_values(by=DATE)

        if len(relatedTransactions) == 1:
            # one transaction of the category will hold the subscription requirements
            # but is surely not a subscription
            return False

        averageCost, averageDateDiff, dateDiffStd = self.getAverageCostAndDateDiff(relatedTransactions)

        # first verify the transactions has a proper interval
        categoryIsSubscription = False
        if dateDiffStd <= 0.1:
            # the date diffs must be consistent
            if abs(averageDateDiff - 7) <= 0.1 or abs(averageDateDiff - 30) <= 0.1:
                # and must imply an interval of once a month / week
                categoryIsSubscription = True

        for i, currRow in relatedTransactions.iterrows():
            rowIsSubscription = categoryIsSubscription and abs(currRow[AMOUNT] - averageCost) <= 20
            self._subscriptionFeatures[IS_SUBSCRIPTION] = rowIsSubscription

        currentRowIsSubscription = categoryIsSubscription and abs(row[AMOUNT] - averageCost) <= 20
        return currentRowIsSubscription


################################################################################################
## subscription features
################################################################################################

    @staticmethod
    def isSimilarAmount(transactions, amount):
        result = False
        for index, row in transactions.iterrows():
            if row[AMOUNT] == amount:
                result = True
                break
        return result


    def getIsSimilarAmountPastWeek(self, row):
        weekAgo = row[DATE] - timedelta(days=7)
        return self.isSimilarAmount(self._subscriptionFeatures[self._subscriptionFeatures[DATE] == weekAgo], row[AMOUNT])

    def getIsSimilarAmountPastMonth(self, row):
        monthAgo = row[DATE] - timedelta(days=30)
        return self.isSimilarAmount(self._subscriptionFeatures[self._subscriptionFeatures[DATE] == monthAgo], row[AMOUNT])


    @staticmethod
    def isSimilarCategory(transactions, category):
        result = False
        for index, row in transactions.iterrows():
            if row[CATEGORY_HASH] == category:
                result = True
                break
        return result

    def getIsSimilarCategoryPastWeek(self, row):
        weekAgo = row[DATE] - timedelta(days=7)
        return self.isSimilarCategory(self._subscriptionFeatures[self._subscriptionFeatures[DATE] == weekAgo], row[CATEGORY_HASH])

    def getIsSimilarCategoryPastMonth(self, row):
        monthAgo = row[DATE] - timedelta(days=30)
        return self.isSimilarCategory(self._subscriptionFeatures[self._subscriptionFeatures[DATE] == monthAgo], row[CATEGORY_HASH])

    # returns the highest similarity in range [0,1] from past week/month transactions
    @staticmethod
    def getTransactionNameSimilarity(transactions, name):
        maxSimilarity = 0
        for index, row in transactions.iterrows():
            currentSimilarity = SequenceMatcher(None, name, row[NAME]).ratio()
            if currentSimilarity > maxSimilarity:
                maxSimilarity = currentSimilarity
                if maxSimilarity == 1:
                    break
        return maxSimilarity

    def getNameSimilarityPastWeek(self, row):
        weekAgo = row[DATE] - timedelta(days=7)
        return self.getTransactionNameSimilarity(self._subscriptionFeatures[self._subscriptionFeatures[DATE] == weekAgo], row[NAME])

    def getNameSimilarityPastMonth(self, row):
        monthAgo = row[DATE] - timedelta(days=30)
        return self.getTransactionNameSimilarity(self._subscriptionFeatures[self._subscriptionFeatures[DATE] == monthAgo], row[NAME])

    def getRelatedTransactionByInterval(self, row, sortedTransactions, interval):
        date = row[DATE]
        sortedTransactions = sortedTransactions[sortedTransactions[DATE] < date]
        return sortedTransactions[sortedTransactions.apply(lambda row: (row[DATE] - row[DATE]).days % interval == 0)]


    def addSubscriptionFeatures(self):
        self._subscriptionFeatures[IS_SUBSCRIPTION] = np.nan
        self._subscriptionFeatures[IS_SUBSCRIPTION] = self._subscriptionFeatures.apply(self.isSubscription, axis=1)
        self._subscriptionFeatures[IS_PAST_WEEK_SIMILAR_AMOUNT] = self._subscriptionFeatures.apply(self.getIsSimilarAmountPastWeek, axis=1)
        self._subscriptionFeatures[IS_PAST_MONTH_SIMILAR_AMOUNT] = self._subscriptionFeatures.apply(self.getIsSimilarAmountPastMonth, axis=1)
        self._subscriptionFeatures[IS_PAST_WEEK_SIMILAR_CATEGORY] = self._subscriptionFeatures.apply(self.getIsSimilarCategoryPastWeek, axis=1)
        self._subscriptionFeatures[IS_PAST_MONTH_SIMILAR_CATEGORY] = self._subscriptionFeatures.apply(self.getIsSimilarCategoryPastMonth, axis=1)
        self._subscriptionFeatures[WEEK_INTERVAL_NAME_SIMILARITY] = self._subscriptionFeatures.apply(self.getNameSimilarityPastWeek, axis=1)
        self._subscriptionFeatures[MONTH_INTERVAL_NAME_SIMILARITY] = self._subscriptionFeatures.apply(self.getNameSimilarityPastMonth, axis=1)

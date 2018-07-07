from datetime import datetime, timedelta
import pandas as pd
import numpy
import os.path as path
from difflib import SequenceMatcher

###############################################
# general porpuse
CACHE_FILE = 'CACHE_FILE.csv'


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

class User:

    _userID = None
    _transactions = None
    _cach_file = ''
    _pastMonthFeatures = {}
    _pastWeekFeatures = {}
    _currentMonthFeatures = {}
    _currentWeekFeatures = {}


    def __init__(self, userID, transactions):
        self._userID = userID
        self._transactions = transactions.sort_values(by=DATE)
        self._cach_file = '/home/gal/development/PycharmProjects/PPL_lab2/cache/' + userID + CACHE_FILE
        print '######################################'
        print userID
        print self._cach_file
        print '######################################'

################################################################################################
## income features
################################################################################################

#######################################
## past month / week features
#######################################

    def calculatePastMonthIncome(self, row):
        date = row[DATE]
        date = date.replace(day=1)
        lastDay = date - timedelta(days=1)
        firstDay = lastDay.replace(day=1)
        if not self._pastMonthFeatures.has_key(firstDay):
            lastMonthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
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
            self._pastMonthFeatures[firstDay] = series.fillna(0)

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
            self._pastWeekFeatures[firstDay] = series.fillna(0)

        result = pd.Series(self._pastWeekFeatures[firstDay])
        result[ID] = row[ID]
        return result

#######################################
## current month / week features
#######################################

    def calculateCurrentMonthIncome(self, row):
        date = row[DATE]
        firstDay = date.replace(day=1)
        if not self._currentMonthFeatures.has_key(firstDay):
            monthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= date]
            monthIncome = monthIncome[monthIncome[IS_INCOME]][AMOUNT]
            series = pd.Series(
                {
                    CURRENT_MONTH_INCOME: monthIncome.sum(),
                    CURRENT_MONTH_INCOME_COUNT: len(monthIncome),
                    # the income is negative therefore min reslted from max and vice versa
                    CURRENT_MONTH_MAX: monthIncome.min(),
                    CURRENT_MONTH_MIN: monthIncome.max(),
                    CURRENT_MONTH_MEAN: monthIncome.mean(),
                    CURRENT_WEEK_STD: monthIncome.std()
                })
            self._currentMonthFeatures[firstDay] = series.fillna(0)

        result = pd.Series(self._currentMonthFeatures[firstDay])
        result[ID] = row[ID]
        return result

    def calculateCurrentWeekIncome(self, row):
        date = row[DATE]
        day = date.isocalendar()[2]
        firstDay = date - timedelta(days=day - 1)
        if not self._currentWeekFeatures.has_key(firstDay):
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
            self._currentWeekFeatures[firstDay] = series.fillna(0)

        result = pd.Series(self._currentWeekFeatures[firstDay])
        result[ID] = row[ID]
        return result

    def addIncomeFeatures(self):
        self._transactions[IS_INCOME] = self._transactions.apply(lambda row: row[AMOUNT] < 0, axis=1)
        print ""
        # TODO uncomment
        # pastMonthIncomeFeatures = self._transactions.apply(self.calculatePastMonthIncome, axis=1)
        # pastWeekIncomeFeatures = self._transactions.apply(self.calculatePastWeekIncome, axis=1)
        # currentMonthIncomeFeatures = self._transactions.apply(self.calculateCurrentMonthIncome, axis=1)
        # currentWeekIncomeFeatures = self._transactions.apply(self.calculateCurrentWeekIncome, axis=1)
        #
        # dfs = [pastMonthIncomeFeatures, pastWeekIncomeFeatures, currentMonthIncomeFeatures, currentWeekIncomeFeatures]
        #
        # for df in dfs:
        #     self._transactions = pd.merge(self._transactions, df, how='inner', on=ID)


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

        if not numpy.isnan(row[IS_SUBSCRIPTION]):
            # preseve the previous calculation from a related transaction
            return row[IS_SUBSCRIPTION]

        categories = row[CATEGORY_HASH]
        relatedTransactions = self._transactions[self._transactions[CATEGORY_HASH] == categories]
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
            self._transactions[IS_SUBSCRIPTION] = rowIsSubscription

        currentRowIsSubscription = categoryIsSubscription and abs(currRow[AMOUNT] - averageCost) <= 20
        return currentRowIsSubscription


    def labelSubscriptionTransactions(self):
        if path.isfile(self._cach_file):
            self._transactions = pd.read_csv(self._cach_file, parse_dates=[DATE])
            return

        self._transactions[IS_SUBSCRIPTION] = numpy.nan
        self._transactions[IS_SUBSCRIPTION] = self._transactions.apply(self.isSubscription, axis=1)
        self._transactions.to_csv(self._cach_file)

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
        return self.isSimilarAmount(self._transactions[self._transactions[DATE] == weekAgo], row[AMOUNT])

    def getIsSimilarAmountPastMonth(self, row):
        monthAgo = row[DATE] - timedelta(days=30)
        return self.isSimilarAmount(self._transactions[self._transactions[DATE] == monthAgo], row[AMOUNT])


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
        return self.isSimilarCategory(self._transactions[self._transactions[DATE] == weekAgo], row[CATEGORY_HASH])

    def getIsSimilarCategoryPastMonth(self, row):
        monthAgo = row[DATE] - timedelta(days=30)
        return self.isSimilarCategory(self._transactions[self._transactions[DATE] == monthAgo], row[CATEGORY_HASH])

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
        return self.getTransactionNameSimilarity(self._transactions[self._transactions[DATE] == weekAgo], row[NAME])

    def getNameSimilarityPastMonth(self, row):
        monthAgo = row[DATE] - timedelta(days=30)
        return self.getTransactionNameSimilarity(self._transactions[self._transactions[DATE] == monthAgo], row[NAME])

    def getRelatedTransactionByInterval(self, row, sortedTransactions, interval):
        date = row[DATE]
        sortedTransactions = sortedTransactions[sortedTransactions[DATE] < date]
        return sortedTransactions[sortedTransactions.apply(lambda row: (row[DATE] - row[DATE]).days % interval == 0)]






    def addSubscriptionFeatures(self):
        transactions = pd.DataFrame(self._transactions)
        transactions[IS_PAST_WEEK_SIMILAR_AMOUNT] = self._transactions.apply(self.getIsSimilarAmountPastWeek, axis=1)
        transactions[IS_PAST_MONTH_SIMILAR_AMOUNT] = self._transactions.apply(self.getIsSimilarAmountPastMonth, axis=1)
        transactions[IS_PAST_WEEK_SIMILAR_CATEGORY] = self._transactions.apply(self.getIsSimilarCategoryPastWeek, axis=1)
        transactions[IS_PAST_MONTH_SIMILAR_CATEGORY] = self._transactions.apply(self.getIsSimilarCategoryPastMonth, axis=1)
        transactions[WEEK_INTERVAL_NAME_SIMILARITY] = self._transactions.apply(self.getNameSimilarityPastWeek, axis=1)
        transactions[MONTH_INTERVAL_NAME_SIMILARITY] = self._transactions.apply(self.getNameSimilarityPastMonth, axis=1)
        print ""

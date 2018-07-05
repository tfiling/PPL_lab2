from datetime import datetime, timedelta
import pandas as pd

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
AMOUNT = 'amount'
DATE = 'date'
SUM = 'sum'
COUNT = 'count'

class User:

    _userID = None
    _transactions = None
    _incomeSumByMonth = {}
    _incomeSumByWeek = {}


    def __init__(self, userID, transactions):
        self._userID = userID
        self._transactions = transactions

    def calculatePastMonthIncome(self, row):
        date = row[DATE]
        date = date.replace(day=1)
        lastDay = date - timedelta(days=1)
        firstDay = lastDay.replace(day=1)
        if not self._incomeSumByMonth.has_key(firstDay):
            lastMonthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            lastMonthIncome = lastMonthIncome[lastMonthIncome[IS_INCOME]][AMOUNT]
            series = pd.Series(
                {
                    PAST_MONTH_INCOME: lastMonthIncome.sum(),
                    PAST_MONTH_INCOME_COUNT: len(lastMonthIncome),
                    # the income is negative therefore min reslted from max and vice versa
                    PAST_MONTH_MAX: lastMonthIncome.min(),
                    PAST_MONTH_MIN: lastMonthIncome.max(),
                    PAST_MONTH_MEAN: lastMonthIncome.mean()
                })
            self._incomeSumByMonth[firstDay] = series.fillna(0)

        result = pd.Series(self._incomeSumByMonth[firstDay])
        result['id'] = row['id']
        return result

    def calculatePastWeekIncome(self, row):
        date = row[DATE]
        day = date.isocalendar()[2]
        firstDay = date - timedelta(days=day + 7)
        lastDay = date - timedelta(days=day)
        if not self._incomeSumByWeek.has_key(firstDay):
            lastWeekIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            lastWeekIncome = lastWeekIncome[lastWeekIncome[IS_INCOME]][AMOUNT]
            series = pd.Series(
                {
                    PAST_WEEK_INCOME: lastWeekIncome.sum(),
                    PAST_WEEK_INCOME_COUNT: len(lastWeekIncome),
                    # the income is negative therefore min reslted from max and vice versa
                    PAST_WEEK_MAX: lastWeekIncome.min(),
                    PAST_WEEK_MIN: lastWeekIncome.max(),
                    PAST_WEEK_MEAN: lastWeekIncome.mean()

                })
            self._incomeSumByWeek[firstDay] = series.fillna(0)

        result = pd.Series(self._incomeSumByWeek[firstDay])
        result['id'] = row['id']
        return result

    def addFeatures(self):
        self._transactions[IS_INCOME] = self._transactions.apply(lambda row: row[AMOUNT] < 0, axis=1)
        pastMonthIncomeFeatures = self._transactions.apply(self.calculatePastMonthIncome, axis=1)
        currentWeekIncomeFeatures = self._transactions.apply(self.calculatePastWeekIncome, axis=1)

        dfs = [pastMonthIncomeFeatures, currentWeekIncomeFeatures]

        for df in dfs:
            self._transactions = pd.merge(self._transactions, df, how='inner', on='id')

        print ""
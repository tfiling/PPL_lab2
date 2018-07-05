from datetime import datetime, timedelta
import pandas as pd

IS_INCOME = 'isIncome'
PAST_MONTH_INCOME_COUNT = 'pastMonthIncomeCount'
PAST_WEEK_INCOME_COUNT = 'pastWeekIncomeCount'
PAST_MONTH_INCOME = 'pastMonthIncome'
PAST_WEEK_INCOME = 'pastWeekIncome'
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

    def calculatePastMonthIncome(self, row, returnStat):
        date = row[DATE]
        date = date.replace(day=1)
        lastDay = date - timedelta(days=1)
        firstDay = lastDay.replace(day=1)
        if not self._incomeSumByMonth.has_key(firstDay):
            lastMonthIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            lastMonthIncome = lastMonthIncome[lastMonthIncome[IS_INCOME]][AMOUNT]
            sum = lastMonthIncome.sum()
            count = len(lastMonthIncome)
            self._incomeSumByMonth[firstDay] = {SUM: sum, COUNT: count}

        return self._incomeSumByMonth[firstDay][returnStat]

    def calculatePastWeekIncome(self, row, returnStat):
        date = row[DATE]
        day = date.isocalendar()[2]
        firstDay = date - timedelta(days=day + 7)
        lastDay = date - timedelta(days=day)
        if not self._incomeSumByWeek.has_key(firstDay):
            lastWeekIncome = self._transactions[self._transactions[DATE] >= firstDay][self._transactions[DATE] <= lastDay]
            lastWeekIncome = lastWeekIncome[lastWeekIncome[IS_INCOME]][AMOUNT]
            sum = lastWeekIncome.sum()
            count = len(lastWeekIncome)
            self._incomeSumByWeek[firstDay] = {SUM: sum, COUNT: count}

        return self._incomeSumByWeek[firstDay][returnStat]



    def addFeatures(self):
        self._transactions[IS_INCOME] = self._transactions.apply(lambda row: row[AMOUNT] < 0, axis=1)
        self._transactions[PAST_MONTH_INCOME] = self._transactions.apply(lambda row: self.calculatePastMonthIncome(row, SUM), axis=1)
        self._transactions[PAST_WEEK_INCOME] = pd.DataFrame(self._transactions.apply(lambda row: self.calculatePastWeekIncome(row, SUM), axis=1))
        self._transactions[PAST_MONTH_INCOME_COUNT] = self._transactions.apply(lambda row: self.calculatePastMonthIncome(row, COUNT), axis=1)
        self._transactions[PAST_WEEK_INCOME_COUNT] = pd.DataFrame(self._transactions.apply(lambda row: self.calculatePastWeekIncome(row, COUNT), axis=1))
        print ""
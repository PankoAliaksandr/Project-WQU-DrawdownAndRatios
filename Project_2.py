# Libraries

from pandas_datareader import data as pdr
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Class implementation
class SP500:

    # Constructor
    def __init__(self):
        self.__index_data = pd.DataFrame()
        self.__index_returns = pd.DataFrame()
        self.__df = pd.DataFrame()
        self.__df_sorted = pd.DataFrame()
        self.__calmar_ratio = None
        self.__strat_results = pd.DataFrame()
        self.__cumulative_returns = list()
        self.__gain_to_pain_ratio = None
        self.__lake_ratio = None

        self.__download_data()

    # Getters
    def get_index_data(self):
        return self.__index_data

    def get_draw_downs(self):
        return self.__df

    def get_calmar_ratio(self):
        return self.__calmar_ratio

    def get_strategy_results(self):
        return self.__strat_results

    def get_cumulative_returns(self):
        return self.__cumulative_returns

    def get_gain_to_pain_ratio(self):
        return self.__gain_to_pain_ratio

    def get_lake_ratio(self):
        return self.__lake_ratio

    def __download_data(self):
        # Determine the first and the last days of the last 10 years period
        end_date = datetime.date.today()
        start_date = datetime.date(end_date.year - 10, end_date.month,
                                   end_date.day)
        # Index data
        self.__index_data = pdr.get_data_yahoo('^GSPC', start_date, end_date)
        self.__index_data = self.__index_data['Adj Close']
        self.__index_data.dropna(inplace=True)

    # Visualization

    def __plot_price(self):
        plt.plot(self.__index_data,
                 label="SP500 price")
        plt.title("S&P 500 price over last 10 years")
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __plot_returns(self):
        plt.plot(self.__index_returns,
                 label="S&P 500 returns")
        plt.title("S&P 500 returns over last 10 years")
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __plot_cumulative_returns(self):
        # Plot cumulative returns
        plt.plot(self.__cumulative_returns)
        plt.title('Cumulative returns of the strategy')
        plt.show()

    def __plot_draw_downs(self):
        for i in range(5):
            drawdown = self.__index_data[
                np.logical_and(self.__index_data.index >=
                               self.__df_sorted['Peak Date'][i],
                               self.__index_data.index <=
                               self.__df_sorted['Recovery Date'][i])]
            plt.plot(drawdown)
            title = 'Worst drawdowns ' + str(i+1)
            plt.title(title)
            plt.gcf().autofmt_xdate()
            plt.show()

    # Calculations

    def __calculate_daily_returns(self):
        self.__index_returns = self.__index_data.pct_change(1)
        # Drop first line with NA
        self.__index_returns.dropna(inplace=True)

    def __find_draw_downs(self):
        # Resulting draw downs data frame
        self.__df = pd.DataFrame(columns=['Drawdown in %', 'Peak Date',
                                          'Trough Date', 'Recovery Date',
                                          'Duration'])
        # Mooving window size 30
        for i in range(30, len(self.__index_data)):
            # 30 days window
            window = self.__index_data[(i-30):i]

            # Initialize drop
            is_in_drop = False
            drop_start_date = None
            drop_start_value = None
            drop_min_value = None
            drop_min_date = None
            drop_value = None
            duration = None

            for j in range(len(window)):

                # Find next drop
                if j > 0 and window[j] < window[j-1]:
                    if is_in_drop is False:
                        # Start drop
                        is_in_drop = True
                        drop_start_date = window.index[j-1]
                        drop_start_value = window[j-1]
                        drop_min_value = window[j]
                        drop_min_date = window.index[j]
                        duration = 1
                        continue
                    else:
                        # Already in a drop
                        duration = duration + 1
                        if window[j] < drop_min_value:
                            drop_min_value = window[j]
                            drop_min_date = window.index[j]
                        continue

                if j > 0 and window[j] > window[j-1]:
                    if is_in_drop is True:
                        if window[j] >= drop_start_value:
                            # End of a drop
                            drop_end_date = window.index[j]
                            drop_value = drop_start_value - drop_min_value
                            if drop_end_date - drop_start_date == 41:
                                print "FUCK!"

                            self.__df.loc[len(self.__df)] = [drop_value /
                                                             drop_start_value,
                                                             drop_start_date,
                                                             drop_min_date,
                                                             drop_end_date,
                                                             np.busday_count(
                                                               drop_start_date,
                                                               drop_end_date)
                                                             ]
                            # Initialize drop
                            is_in_drop = False
                            drop_start_date = None
                            drop_start_value = None
                            drop_min_value = None
                            drop_min_date = None
                            drop_value = None

                        # else: Nothing
                    # else: Nothing

        # Sort, delete duplicates and reset index
        self.__df_sorted = self.__df.sort_values('Drawdown in %',
                                                 ascending=False)
        self.__df_sorted.drop_duplicates(inplace=True)
        self.__df_sorted = self.__df_sorted.reset_index(drop=True)

    def __compounded_annual_growth_rate(self):
        cagr = (self.__index_returns + 1).prod() **    \
            (252.0 / len(self.__index_returns)) - 1
        return cagr

    def __calculate_calmar_ratio(self):
        cagr = self.__compounded_annual_growth_rate()
        self.__calmar_ratio = (100 * cagr) /    \
            self.__df_sorted['Drawdown in %'][0]

    def __implement_strategy(self):
        self.__strat_results = pd.DataFrame(columns=['Index Start',
                                                     'Index End', 'Return'])
        daily_mva = pd.rolling_mean(self.__index_data, 30)
        investment_price = None
        investment_index = None
        for i in range(29, len(self.__index_data)):
            # Price greater or equal than moving average
            if self.__index_data[i] >= daily_mva[i]:
                if investment_price is None:  # Start of a new investment
                    investment_price = self.__index_data[i]
                    investment_index = i
            else:  # Price less than moving average
                if investment_price is not None:  # End of an investment
                    return_value = (self.__index_data[i] - investment_price) /\
                        investment_price
                    self.__strat_results.loc[len(self.__strat_results)] = \
                        [investment_index, i, return_value]
                    investment_price = None
                    investment_index = None

    def __calculate_cumulateve_returns(self):
        self.__cumulative_returns.append(1)
        cum_sum = 1
        for i in self.__strat_results['Return']:
            cum_sum = cum_sum*(1+i)
            self.__cumulative_returns.append(cum_sum)

    def __calculate_gain_to_pain_ratio(self):
        gain = np.sum(self.__strat_results['Return'])
        negative_results = (self.__strat_results[
                self.__strat_results['Return'] < 0])['Return']
        pain = np.abs(np.sum(negative_results))
        self.__gain_to_pain_ratio = gain / pain

        # Show gain and losses
        fig, ax = plt.subplots(nrows=1)
        self.__strat_results['Return'].plot(kind='bar')
        # show every Nth label
        locs, labels = plt.xticks()
        N = 10
        plt.xticks(locs[::N], self.__strat_results.index[::N])
        # autorotate the xlabels
        fig.autofmt_xdate()
        plt.title('Gain and Pain of the strategy')
        plt.xlabel('Trades')
        plt.ylabel('Return')
        plt.show()

    def __calculate_lake_ratio(self):
        water_total_square = 0
        earth_total_square = 0
        for i in range(len(self.__strat_results)):
            portfolio_values = list()
            water_values = list()
            cum_sum = 10
            portfolio_values.append(10)
            water_values.append(0)
            start_index = int(self.__strat_results['Index Start'][i])
            end_index = int(self.__strat_results['Index End'][i])

            for j in range(start_index, end_index):
                cum_sum = cum_sum*(1 + (self.__index_data[i + 1] - self.__index_data[i])/
                                       self.__index_data[i])
                portfolio_values.append(cum_sum)
                water_values.append(np.max(portfolio_values) - cum_sum)

            earth_total_square = earth_total_square + np.sum(portfolio_values)
            water_total_square = water_total_square + np.sum(water_values)

        self.__lake_ratio = water_total_square / earth_total_square

    def main(self):
        self.__calculate_daily_returns()
        self.__plot_price()
        self.__plot_returns()
        self.__find_draw_downs()
        self.__plot_draw_downs()
        self.__calculate_calmar_ratio()
        self.__implement_strategy()
        self.__calculate_cumulateve_returns()
        self.__plot_cumulative_returns()
        self.__calculate_gain_to_pain_ratio()
        self.__calculate_lake_ratio()


sp500 = SP500()
sp500.main()
result = sp500.get_draw_downs()
calmar = sp500.get_calmar_ratio()
strategy = sp500.get_strategy_results()
gp_ratio = sp500.get_gain_to_pain_ratio()
r = sp500.get_lake_ratio()




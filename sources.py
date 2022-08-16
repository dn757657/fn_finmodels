import pandas_datareader as pdr
import yfinance as yf
import datetime
import copy
import numpy as np
from forex_python.converter import CurrencyRates
from dn_bpl import Txn, Category, Account, BplModel
from sqlalchemy.orm import Session
import sqlalchemy as sa
from utils import sqlalch_2_df
import pandas as pd
import logging
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

#internal
from df_utils import incluloc
from df_utils import wrapped_date_range


class FSource:
    """ all sources inherit FSource """

    def __init__(self, name, ylbl, interp=None, forecast=None, **kwargs):
        """
        :args
        data                        dataframe containing source data
        ylbl                        y axis dataframe label - use if data is consistently accessed from a column
                                    always map ylbl to desired output within child objects
        forecast                    forecast object
        """

        self.name = name
        self.data = None
        self.ylbl = ylbl
        self.ylbl_base = self.ylbl + "_base"
        self.ylbl_fcst = self.ylbl + "_fcst"
        self.interp = interp

        self.forecast = forecast  # forecast object for predicting future source values

        # currency params
        self.ccy_out = None
        self.ccy_native = None

        # optional params and source modifiers
        self.__dict__.update(kwargs)

        pass

    def sample(self, start, end):
        """ start/end as datetime -> dataframe sample
        :args
        start                       start of xlbl slice, must match xlbl col type
        end                         end of xlbl slice, must match xlbl col type

        :notes
        - each source type assigns result to self.data to be returned by sample
        - always sliced by index, start, end
        """
        # get base data
        base_sample = self.get_base()
        # TODO could add an option dictating what to do with duplicate indices like sum, drop, etc
        # cannot deal with duplicate indices, must be removed
        base_sample = self._sum_duplicate_indices(base_sample, self.ylbl)

        # forecast individual source using source forecast model and add to self.data
        fcst_df = self.get_forecast(end=end, training_df=base_sample[self.ylbl])
        self.get_final(base=base_sample, fcst=fcst_df)

        # temp = incluloc(self.data[self.ylbl], start, end)  # only return data within requested dates
        # self.data.loc[~self.data.index.isin(temp.index), self.ylbl] = np.nan


        # ensure proper ccy
        if self.ccy_native and self.ccy_out:
            if self.ccy_out != self.ccy_native:
                self.convert_ccy()

        self._apply_options(self.data, self.ylbl)  # apply sample options

        # TODO make slicing a source option in the future
        # self.data[self.ylbl] = incluloc(self.data[self.ylbl], start, end)
        return self.data

    def get_final(self, base, fcst):
        """ get final dataset given base and fcst """
        self.data = base
        self.data[self.ylbl_base] = self.data[self.ylbl]  # copy base to ylbl_base, base stays in ylbl

        if not fcst.empty:
            # self.data[self.ylbl_base] = self.data[self.ylbl]  # copy base to ylbl_base, base stays in ylbl
            self.data = pd.merge(self.data, fcst, left_index=True, right_index=True, how='outer')
            self.data[self.ylbl].fillna(self.data[self.ylbl_fcst], inplace=True)
        # else:
        #     self.data[self.ylbl_fcst] = 0

        return

    def get_base(self):
        """ sample the source base data - each source has own methodology
        - base always returns dataframe with datetime index and ylbl data
        """
        base_sample = pd.DataFrame()

        base_sample.sort_index(inplace=True)  # sort data

        return base_sample

    def get_forecast(self, end, training_df=pd.DataFrame()):
        """ :return data frame with forecasted y within start -> end window as freq=D
        (could resolve freq with code later)
        df has x and y cols along with integer index

        can be used to forecast all?

        :arg
            start           datetime object where forecast starts
            end             datetime object where forecast ends
            training_df     dataframe with datetime index and single column of training data
        """
        if training_df.empty:
            if self.data.empty:
                self.get_base()

            training_df = self.data

        # get start and end bounds for forecasting
        # get max date in available data
        max_data_date = training_df.index.max()
        # get max sample date, cannot assume sample_at is sorted
        max_sample_date = end

        # if sample is future than forecast
        if self.forecast:
            if max_sample_date > max_data_date:
                self.forecast.get_forecast(start=max_data_date, end=max_sample_date, training_df=training_df)
                forecast_df = self.forecast.forecast

            else:
                forecast_df = pd.DataFrame()
        else:
            forecast_df = pd.DataFrame

        # since get_forecast returns the data in the same column name as the trianing data need to rename as ylbl_fcst
        if not forecast_df.empty:
            forecast_df = forecast_df.rename(columns={self.ylbl: self.ylbl_fcst})

        return forecast_df

    def convert_ccy(self):
        """ convert native asset currency to output ccy """
        rates = CurrencyRates()

        self.data['rate'] = np.nan  # blank col
        self.data['rate'].iloc[0] = rates.get_rate(self.ccy_native, self.ccy_out)  # get conversion rate
        self.data = self.data.fillna(method='ffill')  # fill rate forwards
        self.data[self.ylbl] = self.data[self.ylbl] * self.data['rate']  # convert Close to requested ccy

        return

    def interpolate(self, new_index):
        """ interpolate specified col to new axis using source interp_type """

        lbls = [self.ylbl, self.ylbl_base, self.ylbl_fcst]

        # create df from index
        new_idx_df = pd.DataFrame(new_index)
        # remove dates from requested sample that are already in sample data
        parsed_sample_at_df = new_idx_df[~new_idx_df[0].isin(self.data.index)]
        # parsed_sample_at_df = sample_at_df

        # remove requested samples not within the scope of existing index (cannot use interpolation as forecast)
        parsed_sample_at_df = parsed_sample_at_df.loc[parsed_sample_at_df[0] > self.data.index.min()]
        parsed_sample_at_df = parsed_sample_at_df.loc[parsed_sample_at_df[0] < self.data.index.max()]
        parsed_sample_at_df.set_index(0, drop=True, inplace=True)

        self.data = pd.merge(self.data, parsed_sample_at_df, 'outer', left_index=True, right_index=True)
        self.data = self.data.sort_index()

        # only want to apply interpolation on relevant data so we get min and max of existing sample
        # get all index where main ylbl is not nan
        value_indexes = self.data.index[~self.data[self.ylbl].isna()]
        min_index = value_indexes.min()
        max_index = value_indexes.max()

        for ylbl in lbls:
            if ylbl in self.data:
                # interpolate missing values from sample
                if self.interp == 'to_previous':
                    self.data = self.interp_2_prev(self.data, ylbl, start=min_index, end=max_index)
                elif self.interp == 'linear':
                    self.data = self.interp_linear(self.data, ylbl)
                elif self.interp == 'zero':
                    self.data = self.interp_zero(self.data, ylbl, start=min_index, end=max_index)
                # else:
                #     self.data[ylbl] = np.nan

        return

    def interp_2_prev(self, df, ylbl, start, end):
        """ set all nan in self.y to precceeding value """

        df.loc[start:end, ylbl] = df.loc[start:end, ylbl].fillna(method='ffill')

        return df

    def interp_zero(self, df, ylbl, start, end):
        """ all interpolations are always zero """

        df.loc[start:end, ylbl].fillna(value=0, inplace=True)

        return df

    def interp_linear(self, df, ylbl, interp_index=None):
        """ interpolate linearly and fill nan values in self.y_col
         Args:
             interp_index       str label of index to interp to
         """
        df_copy = df.copy()
        if interp_index:  # if not interpolating using index (some other column), set as index
            df_copy = df_copy.set_index(df[interp_index])

        if is_numeric_dtype(df[interp_index]):
            df[ylbl] = df_copy[ylbl].interpolate(method='linear').values
        else:
            df[ylbl] = df_copy[ylbl].interpolate(method='time').values
            # df[self.y] = df_copy[self.y].values  # values required due to misaligned index

        return df

    # source options
    def _apply_options(self, data, col):
        """ apply options to sample
        :param data             dataframe to apply option to
        :param col              col within dataframe where option is applied
        """
        if not col:
            col = self.ylbl  # default col is the ylbl col for source

        # apply asset modifiers to source sample
        # if hasattr(self, 'sum_indices'):
        #     if self.sum_indices:
        #         data = self._sum_duplicate_indices(data, col)
        if hasattr(self, 'cumulative'):
            if self.cumulative:
                data = self._to_cumulative(data, col)
        if hasattr(self, 'inverted'):
            if self.inverted:
                data = self._invert(data, col)

        self.data = data
        return

    def _to_cumulative(self, data, col):
        """ convert y to cumulative sum column """

        if is_numeric_dtype(data[col]):
            # need to add in cumulative data and adjust final return data
            data['base_temp'] = data[self.ylbl_base]
            data['base_temp'] = pd.DataFrame.cumsum(data['base_temp'])

            # get all index where main ylbl is not nan
            value_indexes = data.index[~data[self.ylbl].isna()]
            min_index = value_indexes.min()

            # get cumulative adjustment from base
            adjust = data.loc[min_index, 'base_temp'] - data.loc[min_index, self.ylbl]

            data.loc[:, col] = pd.DataFrame.cumsum(data.loc[:, col])
            data.loc[:, col] = data.loc[:, col] + adjust

        else:
            logging.error("cannot sum non-numeric dtype")

        return data

    def _invert(self, data, col):
        data[col] = data[col] * -1

        return data

    def _normalize(self, data, col, start, end):
        """ normalize cumulative data at the start date """

        # get minimum date in inclusive requested dataset
        inclu_sample = incluloc(data, start, end)
        inclu_sample_min = inclu_sample.index.min()

        # get integer indexes where minimum inclusive date is index
        inclu_sample_min_int = np.where(data.index == inclu_sample_min)

        # get the last value not included
        offset_int = np.amin(inclu_sample_min_int) - 1
        offset = data[col].iloc[offset_int]

        # offset dataset by last value not included
        data[col] = data[col] - offset

        return data

    def _sum_duplicate_indices(self, data, col):
        data[col] = data.groupby(data.index)[col].sum()

        data = data[~data.index.duplicated()]

        return data


class Bpl_Txns(FSource):
    def __init__(self,
                 name,
                 ccy_native,
                 table,
                 index,
                 ylbl='txn_amount',
                 interp=None,
                 forecast=None,
                 filters: list = None,
                 joins: list = None,
                 ccy_out=None,
                 **kwargs):
        """
        :param name:
        :param ccy_native:
        :param ccy_out:
        :param filters:             list of dicts continaing table: table to join and filter: filter to use
        :param acc                  must be str
        :param index                sql table column name to use as iundex
        """
        super().__init__(name=name, ylbl=ylbl, interp=interp, forecast=forecast, **kwargs)
        self.ccy_native = ccy_native
        self.ccy_out = ccy_out
        # self.ylbl = 'txn_amount'

        # db params
        # if not isinstance(categories, list) and categories:
        #     categories = [categories]
        # if not isinstance(accounts, list) and accounts:
        #     accounts = [accounts]

        self.table = table
        self.index = index
        self.filters = filters
        self.joins = joins

    """ source data from bpl txns table sqlite db"""

    def get_base(self):
        """ categories and accounts are always filtered as OR condition """

        db = BplModel()
        session = Session(db.engine)

        filters = list()
        ta_joins = list()

        # if self.category:
        #     ta_joins.append(Category)
        #     filters.append(sa.or_(Category.id == self.category, Category.cat_desc == self.category))

        # for group in self.filters:
        #     table = group['table']
        #     filts = group['filters']
        #
        #     ta_joins.append(table)
        #     for f in filts:
        #         filters.append(f)
        if self.joins:
            for j in self.joins:
                ta_joins.append(j)

        if self.filters:
            for f in self.filters:
                filters.append(f)

        # if self.categories:
        #     ta_joins.append(Category)
        #     for cat in self.categories:
        #         filters.append(sa.or_(Category.id == cat, Category.cat_desc == cat))
        #
        # if self.accounts:
        #     ta_joins.append(Account)
        #     for acc in self.accounts:
        #         filters.append(sa.or_(Account.id == acc, Account.acc_num == acc, Account.acc_desc == acc))

        o_txns = (session.query(self.table))

        if ta_joins:
            o_txns = o_txns.join(*ta_joins)
            # o_txns = o_txns.filter(sa.or_(*filters))
            o_txns = o_txns.filter(*filters)

        df_txns = sqlalch_2_df(o_txns)

        base_sample = df_txns
        base_sample.set_index(self.index, inplace=True)  # index must be desired x column

        base_sample.sort_index(inplace=True)  # sort data

        # super().sample(start, end)
        return base_sample


# ------------------------------------------GENERATED SOURCES------------------------------------------------
class Periodic(FSource):
    """ generate/model periodic source data """

    def __init__(self, name, amount, period_unit, period_size, transition_pairs=None, **kwargs):
        super().__init__(name=name, **kwargs)
        """
        :param transition_pairs             list of lists containing date/amount transition pairs
        :param amount                       amount excluding transition pairs (before or if there are none)
        """

        self.transition_pairs = transition_pairs
        self.amount = amount
        self.period_unit = period_unit
        self.period_size = period_size

        self.freq = str(self.period_size) + self.period_unit

        # internal
        self.ylbl = self.name + "_y"

    def sample(self, start, end):
        """
        :notes
            data is always sampled at native frequency
            """

        # get date range from requested dates
        # first get regular date range(not inclusive)
        temp_dr = pd.Series([start, end])
        # create wrapped date range from regular date range
        sample_dr = wrapped_date_range(temp_dr, self.freq)

        if self.transition_pairs:
            # create df with transition dates and amounts in x, and y cols
            trans_amounts = list()
            trans_dates = list()
            for date, amount in self.transition_pairs:
                trans_dates.append(date)
                trans_amounts.append(amount)

            transition_amounts_df = pd.DataFrame({'temp': trans_dates,
                                                 self.ylbl: trans_amounts})
            # set index and drop col
            transition_amounts_df = transition_amounts_df.set_index('temp', drop=True)
        else:
            transition_amounts_df = pd.DataFrame()

        # add native sampling dates to transition dates df
        # create df from native dates date range
        sample_df = pd.DataFrame(sample_dr)
        sample_df[self.ylbl] = pd.Series()  # add empty ylbl col
        # remove dates from native dates that are already in transition dates
        parsed_sample_df = sample_df[~sample_df[0].isin(transition_amounts_df.index)]
        parsed_sample_df.set_index(0, drop=True, inplace=True)
        sample_df.set_index(0, drop=True, inplace=True)  # for isin command later
        # merge native and transition date dataframes
        sample = pd.merge(transition_amounts_df,
                          parsed_sample_df,
                          'outer', left_index=True, right_index=True)
        sample = sample.sort_index()

        # fill in the values using transition dates and amounts
        final_df = pd.concat([transition_amounts_df, sample])
        final_df = final_df.sort_index()
        final_df.fillna(method='ffill', inplace=True)  # forward fill from all transition dates
        final_df.fillna(self.amount, inplace=True)  # fill remainder with default amount
        # pull only native date values from final_df (exclude transition dates) - may need to rethink
        final_df = final_df[final_df.index.isin(sample_df.index)]

        self.data = final_df[[self.ylbl]]

        self.data = self.data.sort_index()

        super().sample(start=start, end=end)
        return self.data


# ------------------------------------------MARKET SOURCES------------------------------------------------
class Market(FSource):
    """ source for stock market tickers

    :notes
    - yahoo tickers do not always match questrade tickers
    """

    def __init__(self, tckr, name, ccy='CAD', forecast=None, **kwargs):
        """
        :args

        ccy                 desired OUTPUT currency: str :def CAD
        tckr                ticker to source pricing data: str

        :notes
        - market sources assign native currency internally
        """

        super().__init__(name, forecast=forecast, **kwargs)

        # ticker/symbol
        self.tckr = tckr

        # currency out and native
        self.ccy_out = ccy
        self.ccy_native = self.get_native_ccy()

    def sample(self, start, end):

        super().sample(start=start, end=end)
        return

    def get_native_ccy(self):
        """ get native currency for market asset as :str """

        tckr_info = yf.Ticker(self.tckr)
        tckr_ccy = tckr_info.info['currency']

        return tckr_ccy


class YahooMarket(Market):
    """ class to source ticker pricing using the yahoo engine

    :notes
    - yahoo tickers do not always match questrade tickers
     """

    def __init__(self, tckr, name, ccy, forecast=None, **kwargs):
        super().__init__(ccy=ccy, tckr=tckr, name=name, forecast=forecast, **kwargs)

        # mapping columns of interest
        self.ylbl = 'Close'

    def sample(self, start, end):

        # get ticker data as dataframe
        sdf = pdr.get_data_yahoo(str(self.tckr))
        sdf['ds'] = sdf.index

        # eliminate weekends - trialing feature to work with fb prophet better?? might below in forecast func instead
        self.data = sdf[sdf['ds'].dt.dayofweek < 5]

        super().sample(start, end)
        return self.data


class QuestradeMarket(Market):
    """ small variation of yahoo market since tickers from both systems do not always match """

    def __init__(self, tckr, name, ccy, forecast=None, **kwargs):
        super().__init__(ccy=ccy, tckr=tckr, name=name, forecast=forecast, **kwargs)

        # mapping columns of interest
        self.ylbl = 'Close'

    def sample(self, start, end):

        # get ticker data as dataframe
        try:
            df = pdr.get_data_yahoo(str(self.tckr))
        except:
            # occasionally questrade uses a shortened ticker - bad solution but has not failed yet
            df = pdr.get_data_yahoo(str(self.tckr[:-1]))

        self.data = df
        super().sample(start, end)

        return self.data


def main():
    # DEFINE DATES TO SAMPLE
    start = datetime.datetime.strptime('2021-06-12', '%Y-%m-%d')
    transition = datetime.datetime.strptime('2021-08-12', '%Y-%m-%d')
    end = datetime.datetime.strptime('2022-07-08', '%Y-%m-%d')
    now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # vfv_price_source = YahooMarket('yahoo_market', 'CAD', 'VFV.TO')
    # vfv_sample = vfv_price_source.sample(start, now)

    gen_source = Periodic(name='test', amount=450, period_unit='M', period_size=1,
                          transition_pairs=[[transition, 500]])
    gen_sample = gen_source.sample(start=start, end=end)

    print()


if __name__ == '__main__':
    main()
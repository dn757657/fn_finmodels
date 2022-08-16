import pandas_datareader as pdr
import yfinance as yf
import datetime
import copy
import numpy as np
from forex_python.converter import CurrencyRates
from dn_bpl import Txn, Category, Account, BplModel
from sqlalchemy.orm import Session, DeclarativeMeta
from sqlalchemy import Table
import sqlalchemy as sa
from utils import sqlalch_2_df
import pandas as pd
import logging
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

#internal
from df_utils import incluloc
from df_utils import wrapped_date_range
from forecast import SingleIdx


class FSource:
    """ data sources/pipelines """

    def __init__(
            self,
            name: str,
            ylbl: str,
            interp: str = None,
            forecast: SingleIdx = None,
            **kwargs
    ):
        """
        :param name:            source name
        :param ylbl:            lbl denoting final source data
        :param interp:          interpolation method
        :param forecast:        object with get_forecast method
        :param kwargs:          source options
        """

        self.name = name
        self.data = pd.DataFrame()
        self.lbls = {'final': ylbl + "_final",
                     'base': ylbl,
                     'fcst': ylbl + "_fcst"}

        self.interp = interp  # set interp type
        self.forecast = forecast  # forecast object for predicting future source values

        self.__dict__.update(kwargs)  # optional params and source modifiers

        self.fcst_df = pd.DataFrame()  # forecast data

        pass

    # source options
    def _apply_options(
            self,
            data: pd.DataFrame,
            col: str,
            start: object,
            end: object,
    ):
        """
        apply options to data
        :param data             dataframe where options are applied
        :param col              col within dataframe where option is applied
        :param start            date like object indicating slice start
        :param end              date like object indicating slice end

        :return data as modified by options application
        """

        if hasattr(self, 'cumulative'):
            if self.cumulative:
                data = self._sum(data, col)
        if hasattr(self, 'inverted'):
            if self.inverted:
                data = self._invert(data, col)
        if hasattr(self, 'native_ccy') and hasattr(self, 'output_ccy'):
            if self.native_ccy and self.output_ccy:
                data = convert_ccy(data, col, self.native_ccy, self.output_ccy)
        if hasattr(self, 'slice'):
            if self.slice:
                data[col] = incluloc(data[col], start, end)

        return data

    def _sum(
            self,
            data: pd.DataFrame,
            col: str
    ):
        """
        sum col in data - always sums all base data entries allowing for data consistency
        :param data:    dataframe
        :param col:     lbl within dataframe to sum
        :return:        modified dataframe
        """

        if is_numeric_dtype(data[col]):
            # need to add in cumulative data and adjust final return data
            data['base_temp'] = data[self.lbls['base']]
            data['base_temp'] = pd.DataFrame.cumsum(data['base_temp'])

            # get all index where main ylbl is not nan
            value_indexes = data.index[~data[self.lbls['base']].isna()]
            min_index = value_indexes.min()

            # get cumulative adjustment from base
            adjust = data.loc[min_index, 'base_temp'] - data.loc[min_index, self.lbls['final']]

            data.loc[:, col] = pd.DataFrame.cumsum(data.loc[:, col])
            data.loc[:, col] = data.loc[:, col] + adjust

        else:
            logging.error("cannot sum non-numeric dtype")

        return data

    def _invert(
            self,
            data: pd.DataFrame,
            col: str
    ):
        """
        invert col in data
        :param data:    dataframe
        :param col:     lbl within dataframe to invert
        :return:        modified dataframe
        """
        data[col] = data[col] * -1

        return data

    # indices processing
    def _parse_indices(self):
        """
        process duplicate indices using specified method
        """

        if hasattr(self, 'duplicate_indices'):
            if self.duplicate_indices:
                if self.duplicate_indices == 'sum':
                    self._sum_duplicate_indices()

        return

    def _sum_duplicate_indices(self):
        """
        sum duplicate indices in data
        """
        self.data[self.lbls['base']] = self.data.groupby(self.data.index)[self.lbls['base']].sum()
        self.data = self.data[~self.data.index.duplicated()]

        return

    # future
    # def _normalize(self, data, col, start, end):
    #     """ normalize cumulative data at the start date """
    #
    #     # get minimum date in inclusive requested dataset
    #     inclu_sample = incluloc(data, start, end)
    #     inclu_sample_min = inclu_sample.index.min()
    #
    #     # get integer indexes where minimum inclusive date is index
    #     inclu_sample_min_int = np.where(data.index == inclu_sample_min)
    #
    #     # get the last value not included
    #     offset_int = np.amin(inclu_sample_min_int) - 1
    #     offset = data[col].iloc[offset_int]
    #
    #     # offset dataset by last value not included
    #     data[col] = data[col] - offset
    #
    #     return data

    def sample(
            self,
            start: object,
            end: object):
        """
        populate self.data as df with sample in start/end bounds
        :param start:       date type object lower sample bound
        :param end:         date type object upper sample bound
        """

        self.sample_base()  # populate base data
        self.sample_forecast(end=end,
                             training_df=self.data[self.lbls['base']])
        self.assemble()

        self._apply_options(self.data, self.lbls['final'], start, end)  # apply sample options

        return self.data

    def sample_base(self):
        """
        sample the source base data - each source has own methodology
        sample base must pass data to self.data
        """

        self._parse_indices()
        return

    def sample_forecast(
            self,
            end: object,
            training_df: pd.DataFrame):
        """
        populate self.fcst_df with forecast up to end
        :param end:             datetype object where forecast ends
        :param training_df:     df contianing forecast training data
        """

        # get start and end bounds for forecasting
        # get max date in available data
        max_data_date = training_df.index.max()
        # get max sample date, cannot assume sample_at is sorted
        max_sample_date = end

        # if sample is future than forecast
        if self.forecast:
            if max_sample_date > max_data_date:
                self.fcst_df = self.forecast.get_forecast(start=max_data_date,
                                                          end=max_sample_date,
                                                          training_df=training_df)

            else:
                self.fcst_df = pd.DataFrame()
        else:
            self.fcst_df = pd.DataFrame

        # since get_forecast returns the data in the same column name as the trianing data need to rename as ylbl_fcst
        if not self.fcst_df.empty:
            self.fcst_df = self.fcst_df.rename(columns={self.lbls['base']: self.lbls['fcst']})

        return

    def assemble(self):
        """ copy base to final column and merge forecast if present """

        self.data[self.lbls['final']] = self.data[self.lbls['base']]  # copy base to final

        # add and merge forecast to final dataset if present
        if not self.fcst_df.empty:

            self.data = pd.merge(self.data, self.fcst_df, left_index=True, right_index=True, how='outer')
            self.data[self.lbls['final']].fillna(self.data[self.lbls['fcst']], inplace=True)

        return


def convert_ccy(data, col, ccy_native, ccy_out):
    """ convert native asset currency to output ccy """
    rates = CurrencyRates()

    data['rate'] = np.nan  # blank col
    data['rate'].iloc[0] = rates.get_rate(ccy_native, ccy_out)  # get conversion rate
    data['rate'] = data['rate'].fillna(method='ffill')  # fill rate forwards
    data[col] = data[col] * data['rate']  # convert Close to requested ccy

    return data


class DbSource(FSource):
    def __init__(
            self,
            name: str,
            table,
            session: Session,
            index: str,
            ylbl: str = 'txn_amount',
            interp: str = None,
            forecast: SingleIdx = None,
            filters: list = None,
            joins: list = None,
            **kwargs):
        """

        :param name:            source name for ref
        :param table:           sqlalch table source
        :param index:           sql table column label to use as data index
        :param ylbl:            sql table column label to use as data
        :param interp:          interpolation method
        :param forecast:        forecast object
        :param filters:         sqlalch style query filters
        :param joins:           sqlalch table objects to join to query
        """

        super().__init__(name=name, ylbl=ylbl, interp=interp, forecast=forecast, **kwargs)

        self.session = session
        self.table = table
        self.index = index
        self.filters = filters
        self.joins = joins

    def sample_base(self):
        """ categories and accounts are always filtered as OR condition """

        obs = (self.session.query(self.table))

        if self.joins:
            obs = obs.join(*self.joins)
        if self.filters:
            obs = obs.filter(*self.filters)

        df = sqlalch_2_df(obs)  # create df from sqlalh objects

        df.set_index(self.index, inplace=True)  # index must be desired x column
        df.sort_index(inplace=True)  # sort data

        self.data = df
        super().sample_base()
        return


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
import pandas as pd
import logging
from forecasts import kats_prophet
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
import copy
from forex_python.converter import CurrencyRates
# internal

from dn_pricing.pricing import get_all_yahoo
from dn_qtrade.questrade_2 import QTAccount

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from df_utils import wrapped_date_range, del_col_duplicates
from eval_methods import invest_eval, crypto_eval
from dn_qtrade.questrade_2 import QTAccount
from dn_crypto.crypto2 import cryptoWallet
from dn_pricing.pricing import get_price, get_latest_price


class FAsset:
    def __init__(self, name, interp_type, sources, forecast=None, cumulative=None):
        """
        Attrs:
            data            df sampled from sources, if multiple sources
            name            name of asset provided by user
            interp_type     how y columns in data is interpolated
            forecast        object with get_forcast capability?

        """

        if not isinstance(sources, list):
            sources = [sources]

        # set attributes
        self.name = name
        self.sources = sources
        self.forecast = forecast

        # generated/output
        self.data = pd.DataFrame()

        # options
        self.cumulative = cumulative

        # internal
        self.ylbl = self.name+'_y'

        interp_types = ['to_previous', 'linear']
        if interp_type in interp_types:
            self.interp_type = interp_type
        else:
            logging.error('interp type must be in [%s]' % ', '.join(map(str, interp_types)))

        # alias source y labels to prevent duplicates
        self.source_ylbl_aliased = dict()
        for source in self.sources:
            self.source_ylbl_aliased[source.name] = source.ylbl + "_" + source.name

    def sample(self, sample_at):
        """ sample all sources and set to self.data, sum all sources ylbl into asset ylbl col

        :notes
        use sample at as optional interpolation points??
        """
        sample_at = pd.Series(sample_at)

        # sample sources and rename ylbl to asset ylbl (instead of source ylbl)
        # sum all source ylbl cols into asset ylbl
        sample_start = sample_at.min()
        sample_end = sample_at.max()
        for source in self.sources:
            # sample the source
            sample = source.sample(start=sample_start, end=sample_end)
            # dropping duplicate index at the beginning is important
            # this ensures the most recent datapoint is kept given duplicate index
            # given the data is in the correct order when initiated
            sample = sample[~sample.index.duplicated(keep='last')]

            # apply asset modifiers to source sample
            if self.cumulative:
                sample = self._to_cumulative(sample, source.ylbl)
                # sample = self._to_cumulative(sample, source.ylbl)

            # forecast individual source using asset forecast model
            forecast_df = self.get_forecast(sample_at=sample_at, training_df=sample[source.ylbl])

            # add forecast to sample if exists
            if not forecast_df.empty:
                sample = pd.concat([forecast_df, sample]).drop_duplicates(keep=False)
                # sample = pd.concat([forecast_df, sample]).drop_duplicates(keep=False)
            else:
                logging.error("Cannot sample future without forecast")
                return

            # add sample to asset data
            if not self.data.empty:
                # combine the source data into asset total
                self.data = pd.merge(self.data, sample, left_index=True, right_index=True, how='outer')
            else:
                self.data = sample

            # rename source y as aliased
            self.data = self.data.rename(columns={source.ylbl: self.source_ylbl_aliased[source.name]})

        for source in self.sources:  # first get all samples, then since all dates are included, interpolate
            # interpolate the sources after all dates have been added or else odd behaviour/rogue zeroes
            self.data = self.interpolate(sample_at=sample_at,
                                         ylbl=self.source_ylbl_aliased[source.name],
                                         sample=self.data)

            if self.ylbl in self.data.columns:
                self.data.fillna(0, inplace=True)
                self.data[self.ylbl] = self.data[self.ylbl] + self.data[self.source_ylbl_aliased[source.name]]
            else:
                self.data[self.ylbl] = self.data[self.source_ylbl_aliased[source.name]]

        return self.data

    def interpolate(self, sample_at, ylbl, sample=pd.DataFrame()):
        """
        interpolate sample or self.data, ensuring all sample_at dates are in returned df using interp_type

        :param sample_at:
        :param sample:
        :return:
        """
        if sample.empty:
            sample = self.data
            ylbl = self.ylbl

        # add requested dates to data_df if they are not there already
        # create df from parsed samples to add to data_df
        sample_at_df = pd.DataFrame(sample_at)
        # remove dates from requested sample that are already in sample data
        parsed_sample_at_df = sample_at_df[~sample_at_df[0].isin(sample.index)]
        parsed_sample_at_df.set_index(0, drop=True, inplace=True)

        sample = pd.merge(sample, parsed_sample_at_df, 'outer', left_index=True, right_index=True)
        sample = sample.sort_index()

        # interpolate missing values from sample
        if self.interp_type == 'to_previous':
            sample = self.interp_2_prev(sample, ylbl)
        elif self.interp_type == 'linear':
            sample = self.interp_linear(sample, ylbl)

        return sample

    def get_forecast(self, sample_at, training_df=pd.DataFrame()):
        """ :return data frame with forecasted y within start -> end window as freq=D
        (could resolve freq with code later)
        df has x and y cols along with integer index

        can be used to forecast all?

        :arg
            start           datetime object where forecast starts
            end             datetime object where forecast ends
        """
        if training_df.empty:
            if self.data.empty:
                self.sample(sample_at)

            training_df = self.data

        # get start and end bounds for forecasting
        # get max date in available data
        max_data_date = training_df.index.max()
        # get max sample date, cannot assume sample_at is sorted
        max_sample_date = sample_at.max()

        # if sample is future than forecast
        if self.forecast:
            if max_sample_date > max_data_date:
                self.forecast.get_forecast(start=max_data_date, end=max_sample_date, training_df=training_df)
                forecast_df = self.forecast.forecast

            else:
                forecast_df = pd.DataFrame()
        else:
            forecast_df = pd.DataFrame

        return forecast_df

    # old sample function
    # def sample(self, sample_at, norm_at=None, norm_freq=None):
    #     """ each asset needs a sample function
    #     requires self.data as dataframe
    #     interp as 'to_previous' or 'linear'
    #
    #     :strategy
    #         parse samples
    #         fetch/generate self.data
    #         add all required dates into sampling df prior to interp
    #             add all forecasted
    #             add all sample_at
    #             add all norm_at
    #             **do not overwrite any datapoints that exist already
    #         interp data_df to fill in blanks
    #         norm if requested**
    #         return dates requested from data_df as df
    #
    #         maybe break this badboy into functions for readability and ease of understanding
    #     """
    #
    #     self.sample_2(sample_at=sample_at)
    #
    #     sample_at = pd.Series(sample_at)
    #
    #     # self.sample_sources(sample_at=sample_at)
    #
    #     forecast_df = self.get_forecast(sample_at)
    #     # add forecasted to existing self.data
    #     if not forecast_df.empty:
    #         self.data = pd.concat([forecast_df, self.data]).drop_duplicates(keep=False)
    #     else:
    #         logging.error("Cannot sample future without forecast")
    #         return
    #
    #     sample_data_df = self.data
    #
    #     # forecast and add forecast to data_df
    #     # get max date in available data
    #     max_data_date = sample_data_df[self.x].max()
    #     # get max sample date, cannot assume sample_at is sorted
    #     # leverage pd.series max function by transforming to series
    #     sample_at_series = pd.Series(sample_at)
    #     max_sample_date = sample_at_series.max()
    #
    #     # if sample is future than forecast
    #     if max_sample_date > max_data_date:
    #         forecast_df = self.forecast(start=max_data_date,
    #                                     end=max_sample_date)
    #         if not forecast_df.empty:
    #             sample_data_df = pd.concat([forecast_df, sample_data_df]).drop_duplicates(keep=False)
    #         else:
    #             logging.error("Cannot sample future without forecast")
    #             return
    #
    #     # add requested dates to data_df if they are not there already
    #     # create df from parsed samples to add to data_df
    #     sample_at_df = self.sample_2_df(sample_at=sample_at)
    #     parsed_sample_at_df = del_col_duplicates(sample_at_df, self.x, sample_data_df, self.x)
    #
    #     # combine refined sample_df with self.data & sort
    #     sample_data_df = pd.concat([parsed_sample_at_df, sample_data_df])
    #     sample_data_df.sort_values([self.x], inplace=True)
    #
    #     # interpolate missing values from sample
    #     if self.interp_type == 'to_previous':
    #         sample_data_df = self.interp_2_prev(sample_data_df)
    #     elif self.interp_type == 'linear':
    #         sample_data_df = self.interp_linear(sample_data_df, self.x)
    #
    #     # slice out anything in sample_data_df less than the min requested sample value
    #     sample_data_df.reset_index(inplace=True, drop=True)
    #     sample_data_df = sample_data_df.loc[sample_data_df[(sample_data_df[self.x] == sample_at_df[self.x].min())].index[0] :, :]
    #
    #     # normalize sampled asset to another
    #     # if norm_asset:
    #     if norm_at or norm_freq:
    #         sample_data_df = self.normalize_data(sample_data_df, norm_at, norm_freq)
    #     else:
    #         logging.error("Must provide either norm_at or norm_freq with norm_asset")
    #     pass
    #
    #     # refine values from sampled data only at requested sample values values
    #     sample_data_df = sample_data_df[sample_data_df[self.x].isin(sample_at_df[self.x])]
    #
    #     # sort by x and reset index
    #     sample_data_df.sort_values(self.x, inplace=True)
    #     sample_data_df.reset_index(inplace=True, drop=True)
    #
    #     return sample_data_df[[self.x, self.y]]

    def normalize_data(self, to_norm_data, norm_at=None, norm_freq=None):
        """
            asset is normalized to itself, using the norm_freq as the start point for each new
            normalized period.

        :arg
            to_norm_data        asset data to be normalized
            norm_at             list of x column points where data is normalized
            norm_freq           pandas compatible frequency to generate norm_at

        :notes
            for now we assume y is always normalized using x
            must have either norm_at or norm_freq

            what is normalization?
            -aligning the datasets at given intervals, negating the cumulative effects of a dataset
            -essentially only for cumulative data?

        :strategy
            nothing really changes about output dataset! Simply data manipulations for display?
            get last value of previous period in normalization range requested and apply to next
            period
        """

        # parse norm points
        norm_at = self.parse_norm_at(norm_at, to_norm_data, norm_freq)
        # shift all norm point back one day, to get final value of previous norm period
        norm_at_shifted = list()
        for i in range(0, len(norm_at)):
            norm_at_shifted.append(norm_at[i] - pd.Timedelta(days=1))
        norm_at = norm_at_shifted

        # sample self at adjusted norm points
        norm_sample_df = self.sample(norm_at)
        # add day so that final value of each norm period is applied to the NEXT norm period
        norm_sample_df[self.x] = norm_sample_df[self.x] + pd.Timedelta(days=1)

        # merge both samples and get differences at norm points
        merged_df = pd.merge(to_norm_data, norm_sample_df, on=self.x, how='outer')
        merged_df.sort_values([self.x], inplace=True)

        # ffill values from final values sample such that they are applied to entire sample period
        merged_df.iloc[:, 2].fillna(method='ffill', inplace=True)
        merged_df.iloc[:, 2].fillna(0, inplace=True)  # fill rest with zeroes or you get nan
        merged_df = merged_df.rename(columns={merged_df.columns[1]: self.y})
        merged_df[self.y] = merged_df[self.y] - merged_df.iloc[:, 2]

        # return adjusted df at original points with original columns
        final_df = merged_df[merged_df[self.x].isin(to_norm_data[self.x])]

        final_df = final_df[[self.x, self.y]]
        return final_df

    def parse_norm_at(self, norm_at, to_norm_data, norm_freq):
        """ check or generate norm_at - norm at must be contained by both norm_to and to_norm data

         :returns
            norm_at as list, or none if requested norm_at is invalid
         """

        if not norm_at:
            norm_at = self.get_norm_at(to_norm_data, norm_freq)

        # check if norm_at in both data sources provided
        # norm_to_sample_df = norm_to_asset.sample(norm_at)
        norm_to_df = pd.DataFrame().reindex(columns=self.data.columns)
        norm_to_df[self.x] = norm_at

        # if norm to must not contain any past dates concerning asset datasets
        # if norm_to_df[norm_to_asset.x].min() < norm_to_sample_df[norm_to_asset.x].min():
        #     return None
        if norm_to_df[self.x].min() < self.data[self.x].min():
            if self.generative:  # if generative
                return norm_at
            else:
                return None
        else:
            return norm_at

    def get_norm_at(self, to_norm_data, norm_freq):
        """ find upper and lower bounds of data - generate norm_at using bounds and freq

        :returns
            list containing indicies to normalize data at
        """

        upper = to_norm_data[self.x].max()
        lower = to_norm_data[self.x].min()

        norm_at = wrapped_date_range([upper, lower], norm_freq)
        norm_at = norm_at.tolist()

        return norm_at

    # TODO move interp functions to new file for more complex interps?
    def interp_2_prev(self, df, ylbl):
        """ set all nan in self.y to precceeding value """

        df[ylbl] = df[ylbl].fillna(method='ffill')

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

    def parse_sample(self, sample_at):
        """ elim duplicates from requested samples and try to match asset index dtype """

        if not isinstance(sample_at, list):
            sample_at = [sample_at]

        sample_at.sort()
        # convert non numeric samples to pandas Timestamp
        parsed_sample_at = list()
        if not is_numeric_dtype(self.data.index):
            for sample in sample_at:
                # only append unique values
                if sample not in parsed_sample_at:
                    parsed_sample_at.append(pd.Timestamp(sample))
                else:
                    continue
        else:
            # elim duplicates
            parsed_sample_at = set(sample_at)

        parsed_sample_at = list(parsed_sample_at)

        return parsed_sample_at

    def sample_2_df(self, sample_at):
        """ transform sample_at list to dataframe with same columns as self.data """

        # create blank dataframe same as dates_grouped
        sample_df = pd.DataFrame().reindex(columns=self.data.columns)
        # add sample_at values to x column
        sample_df[self.x] = list(sample_at)

        return sample_df

    def _to_cumulative(self, data, col):
        """ convert y to cumulative sum column """

        if is_numeric_dtype(data[col]):
            data.loc[:, col] = pd.DataFrame.cumsum(data.loc[:, col])
        else:
            logging.error("cannot sum non-numeric dtype")

        return data
# class StaticAsset(FAsset):


class DFAsset(FAsset):
    """ Dataframe Asset """
    def __init__(self, name, interp_type, forecast_type, data_df, x, y, target_asset=None, cumulative=None):
        super().__init__(name=name, interp_type=interp_type, target_asset=target_asset, forecast_type=forecast_type, cumulative=cumulative)

        self.data = data_df

        # process data_df
        # sort by x
        self.data.sort_values(x, inplace=True)
        self.data.reset_index(inplace=True, drop=True)
        # rename columns indicated as defaults from FAsset
        self.data = self.data.rename(columns={x: self.x,
                                     y: self.y})
        # only keep x and y columns
        self.data = self.data[[self.x, self.y]]

        if self.cumulative:
            self._to_cumulative()

        # set generative prop by assessing if generate_data func exists
        generative_check = getattr(self, 'generate_data', None)
        if callable(generative_check):
            self.generative = True

    def parse_args(self, **kwargs):
        """ parse objects kwargs """

        if kwargs['x'] is None or kwargs['y'] is None:
            if self.x not in self.data.columns and self.y in self.data.columns:
                logging.error("data has no 'x', or 'y' columns - cannot prepare asset")
                return
            else:
                self.x_source = self.x
                self.y_source = self.y
        else:
            self.x_source = kwargs['x']
            self.y_source = kwargs['y']

    # def to_cumulative(self, col, **kwargs):
    #     """ convert col to cumulative sum column """
    #     self.data = super().to_cumulative(col=col, **kwargs)
    #
    #     # # match y if summed
    #     # if col == self.y_col:
    #     #     self.data['y'] = self.data[self.y_col]

    def prepare_asset_data(self):
        self.data.sort_values(self.x_source, inplace=True)

        # make new columns of x and y in order not to disturb original data
        self.data[self.x] = self.data[self.x_source]
        self.data[self.y] = self.data[self.y_source]

        # # make index same as 'x' column with diff name to avoid ambiguity
        # self.data['new_idx'] = pd.DatetimeIndex(self.data[self.x_col])
        # self.data.set_index(self.data['new_idx'], inplace=True)
        # self.data.drop(columns=['new_idx'], inplace=True)

    # def parse_sample(self, sample_at):
    #     """ elim duplicates from requested samples and match data index dtype """
    #     if not isinstance(sample_at, list):
    #         sample_at = [sample_at]
    #
    #     # convert non numeric samples to pandas Timestamp
    #     parsed_samples = list()
    #     if not is_numeric_dtype(self.data.index):
    #         # if not is_datetime64_any_dtype(samples):
    #         for sample in sample_at:
    #             # only append unique values
    #             if sample not in parsed_samples:
    #                 parsed_samples.append(pd.Timestamp(sample))
    #             else:
    #                 continue
    #     else:
    #         # elim duplicates
    #         parsed_samples = set(sample_at)
    #
    #     return parsed_samples

    # def sample(self, sample_at):
    #     # TODO self.data_type, cont(inuous)
    #
    #     """ sample self.data matching sample_at values -> df containing x and y data at sample values """
    #
    #     result = super().sample(sample_at=sample_at)
    #
    #     return result


class PeriodicAsset(FAsset):
    """ reoccuring cyclic asset with fixed value

     cumulative is default true for periodic assets, doesnt make sens otherwise and
     pseudo non-cumulative nature can be created using normalization

     active dates and amounts align with when amount comes into affect, same index
     """

    def __init__(self, name, interp_type, active_dates, amounts, period_unit, period_size):
        super().__init__(name=name, interp_type=interp_type, forecast_type=None, cumulative=True)

        if len(active_dates) != len(amounts):
            logging.error("Can only provide one active date per budgeted amount, init() failed")
            return

        self.active_dates = active_dates
        self.amounts = amounts
        self.period_unit = period_unit
        self.period_size = period_size

        self.freq = str(self.period_size) + self.period_unit

        # assign generative attr by assessing if generate_data func exists
        generative_check = getattr(self, 'generate_data', None)
        if callable(generative_check):
            self.generative = True

    def sample(self, sample_at, gen_freq=None, norm_at=None, norm_freq=None):
        """  :returns df containing x and y data at sample values """

        # # always include active dates, ensuring samples always align cumulatively
        # sample_at_gen = copy.deepcopy(sample_at)
        # for date in self.active_dates:
        #     sample_at_gen.append(date)

        self.generate_data(sample_at)

        sample = super().sample(sample_at=sample_at, norm_at=norm_at, norm_freq=norm_freq)

        return sample

    def generate_data(self, sample_at):
        """ generate data inclusive of requested samples """

        # generate dataframe using asset frequency as x, and asset amount as y
        # self.data = pd.DataFrame({self.x: wrapped_date_range(sample_at, self.freq),
        #                           self.y: self.amount})
        # native_dates_df = pd.DataFrame({self.x: wrapped_date_range(sample_at, self.freq),
        #                                 self.y: []})

        # # always include active dates, ensuring samples always align cumulatively
        sample_at_gen = copy.deepcopy(sample_at)
        for date in self.active_dates:
            sample_at_gen.append(date)

        # create df with active dates and amounts in x, and y cols
        active_amounts_df = pd.DataFrame({self.x: self.active_dates,
                                          self.y: self.amounts})

        # create blank dataframe same as dates_grouped
        sample_at_df = pd.DataFrame().reindex(columns=active_amounts_df.columns)
        native_dates_df = pd.DataFrame().reindex(columns=active_amounts_df.columns)
        # add sample_at values to x column
        sample_at_df[self.x] = list(sample_at)
        native_dates_df[self.x] = wrapped_date_range(sample_at_gen, self.freq)

        active_amounts_df = pd.concat([active_amounts_df, sample_at_df])
        active_amounts_df = pd.concat([active_amounts_df, native_dates_df])
        active_amounts_df = active_amounts_df.sort_values([self.x])
        active_amounts_df.drop_duplicates(inplace=True)
        active_amounts_df.fillna(method='ffill', inplace=True)
        active_amounts_df = active_amounts_df[active_amounts_df[self.x].isin(native_dates_df[self.x])]

        self.data = active_amounts_df

        self.data.sort_values([self.x], inplace=True)
        super()._to_cumulative()

        # if self.cumulative:
        # self.data.loc[self.data[self.x] == self.data[self.x].min(), [self.y]] = 0

        return


class EvalAsset(DFAsset):
    """ asset representing quantities that are periodically evaulated, or where values fluctuate with time and

    """

    def __init__(self, name, interp_type, forecast_type, data_df, x, y, eval_method, eval_ccy='CAD', target_asset=None, cumulative=None):
        """ acc_type indicates the type of asset object to use to get quantities of asset
        y is the symbol of asset to be evaluated - maybe not best practcie
        """
        super().__init__(name=name,
                         interp_type=interp_type,
                         target_asset=target_asset,
                         data_df=data_df,
                         x=x, y=y,
                         forecast_type=forecast_type,
                         cumulative=cumulative)

        # self.data = self.data.rename(columns={self.y: y})  # maintain y to store symbol?
        self.eval_method = eval_method  # determines if and how eval asset will be evaluated
        self.eval_ccy = eval_ccy
        self.eval_asset = None

        # if eval method and ccy, create source asset
        if self.eval_method and self.eval_ccy:
            self.eval_asset = EvalSourceAsset(name=name + 'source',
                                              interp_type=interp_type,
                                              forecast_type=forecast_type,
                                              eval_method=eval_method,
                                              eval_symbol=y,
                                              eval_ccy=self.eval_ccy)

        self.symbol = y

        # TODO create own eval asset using internal info and eval_method, that way user doesnt have to make own eval_asset!

    # def _acc_obj_factory(self, acc_type, **kwargs):
    #     acc_obj = None
    #     acc_type_available = ['qtrade', 'crypto']
    #
    #     if acc_type == 'crypto':
    #         if 'addy' in kwargs:
    #             acc_obj = cryptoWallet(addy=kwargs['addy'])
    #         else:
    #             logging.error("Must provide arg:addy with acc_type:crypto")
    #
    #     elif acc_type == 'qtrade':
    #         acc_obj = QTAccount()
    #
    #     return acc_obj

    def sample(self, sample_at, norm_at=None, norm_freq=None):
        """  :returns df containing x and y data at sample values """

        eval = self.eval_asset.sample(sample_at)

        sample = super().sample(sample_at=sample_at, norm_at=norm_at, norm_freq=norm_freq)

        return sample

    # def evaluate(self, eval_at):
    #     """ evaulate quantites in self.data as desired currency """
    #
    #     if self.eval_method == 'invest':
    #         self.data = invest_eval(self.data, self.eval_ccy)
    #     elif self.eval_method == 'crypto':
    #         self.data = crypto_eval(self.data, self.eval_ccy)
    #
    #     return

    # def generate_data(self, sample_at):
    #     """ generate data inclusive of requested samples
    #
    #     :method
    #         bust out into quans
    #         evaulate each in currency requested - defualt CAD
    #         evaluate cumulative always?
    #         transform/sum into self.data
    #      """
    #
    #     self.data = self.acc.get_asset_balances(sample_at)
    #     self.data.sort_values([self.x], inplace=True)
    #
    #     return


class EvalSourceAsset(FAsset):
    """ asset to evaluate quantites in EvalAssets """

    def __init__(self, name, interp_type, forecast_type, eval_method, eval_symbol, eval_ccy='CAD', target_asset=None, cumulative=True):
        """ acc_type indicates the type of asset object to use to get quantities of asset """

        self.eval_method = eval_method
        self.eval_ccy = eval_ccy
        self.eval_symbol = eval_symbol

        super().__init__(name=name,
                         interp_type=interp_type,
                         target_asset=target_asset,
                         forecast_type=forecast_type,
                         cumulative=cumulative)

    def sample(self, sample_at, norm_at=None, norm_freq=None):
        if self.eval_method == 'market':
            self.data = self.source_market()

        sample = super().sample(sample_at=sample_at, norm_at=norm_at, norm_freq=norm_freq)

        return sample

    def source_market(self):
        # TODO extend historical date to the yahoo limit bb
        # TODO make mappings from yahoo df as variable to self.x and self.y
        rates = CurrencyRates()
        qt = QTAccount()

        try:
            df = get_all_yahoo(self.eval_symbol)  # get pricing
        except:
            df = get_all_yahoo(self.eval_symbol[:-1])  # sometimes yahoo cuts off the last letter and it doesnt work

        df = df[['Close']]  # only need clsoe col

        # get asset native currency via txns
        native_asset_ccy = None
        ticker = qt.conn.ticker_information(self.eval_symbol)
        if 'currency' in ticker.keys():
            native_asset_ccy = ticker['currency']

        # cyrptocmd uses USD by default, so if other ccy requested, convert
        if self.eval_ccy != native_asset_ccy:
            df['rate'] = np.nan  # blank col
            df['rate'].iloc[0] = rates.get_rate('USD', self.eval_ccy)  # get conversion rate
            df = df.fillna(method='ffill')  # fill rate forwards
            df['Close'] = df['Close']*df['rate']  # convert Close to requested ccy

        df = df.reset_index()
        df = df[['Date', 'Close']]
        df = df.rename(columns={'Date': self.x,
                                'Close': self.y})

        self.data = df

        return df

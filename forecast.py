import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

from df_utils import wrapped_date_range


class SingleIdx:
    """ forecast which uses a single index """

    def __init__(self, name):
        """
        :param tsource              tsource is Source object for forecast/model training
                                    tsource is new each time such that multiple assets can use the SAME FORECAST
        """
        self.name = name

        self.tsource = None
        self.model = None
        self.forecast = None

    def get_forecast(self, start, end, training_df):
        """
        :param start:                   datetype forecast start
        :param end:                     datetype forecast end
        :param training_df:             must be index and training data column only
        :return:                        df indexed like training with one forecasted column
        """

        self.train(training_df)

        return

    def train(self, training_df):
        """
        :param training_df:             df containing index and single column of historical training data
        :return:                        trained model

        :notes
        - in future could add params for model fine tunning see kats/prophet params
        """

        return

    def days_diff(self, start, end):
        """ some forecast types forecast a number of days rather than via bounds

        start, end as datetypes -> days_diff as int
        """

        dates_df = pd.DataFrame([[start, end]], columns=['start', 'end'])  # create df
        dates_df['td'] = dates_df['end'] - dates_df['start']  # get diff
        days_diff = dates_df['td'].iloc[0].days  # get diff as days from datetype

        return days_diff


class KatsProphet(SingleIdx):
    """  forecast using kats libbrary and Prophet model """

    def __init__(self, name):
        super().__init__(name=name)

    def get_forecast(self, start, end, training_df):
        """
        forcast requested window and set resutling df to self.forecast

        :param start:
        :param end:
        :param training_df:
        :return:
        """
        training_df = training_df.reset_index()  # migrate index to column to be sued by fit
        training_df = training_df.rename(columns={list(training_df)[0]: 'time'})  # x column must be named time
        super().get_forecast(start, end, training_df)

        days_diff = self.days_diff(start, end)

        # make prediction for next days
        fcst = self.model.predict(steps=days_diff, freq="D")

        fcst_df = fcst[['time', 'fcst']]
        fcst_df.set_index("time", drop=True, inplace=True)  # set time col as index and drop

        # rename col back to original name
        fcst_df = fcst_df.rename(columns={'fcst': training_df.columns[1]})

        self.forecast = fcst_df

    def train(self, training_df):
        # training_df = training_df.reset_index()  # migrate index to column to be sued by fit
        # training_df = training_df.rename(columns={list(training_df)[0]: 'time'})  # x column must be named time

        # convert to TimeSeriesData object
        training_ts = TimeSeriesData(training_df)

        # create a model param instance
        # params = ProphetParams(seasonality_mode='multiplicative')  # additive mode gives worse results
        params = ProphetParams()

        # create a prophet model instance using training data
        m = ProphetModel(training_ts, params)

        # fit model simply by calling m.fit()
        m.fit()

        self.model = m

        super().train(training_df)


class AssetForecast(SingleIdx):
    """ use an asset as a forecast """

    def __init__(self, name, asset):
        super().__init__(name=name)

        self.asset = asset

    def get_forecast(self, start, end, training_df):
        super().get_forecast(start=start, end=end, training_df=training_df)

        sample_at = wrapped_date_range(pd.date_range(start=start, end=end, freq='1D'), '1D')

        fcst_df = self.asset.sample(sample_at=sample_at)
        fcst_df = fcst_df.loc[:end]

        self.forecast = fcst_df

        return

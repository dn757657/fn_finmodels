# forecast functions for assets
# all use start, end, historical data df, and str column labels to use as x and y for historical data df
# all return dataframe with asset.x and asset.y columns (defaults)
# with y containing forecasted data in window reflected by start/end

import pandas as pd

from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams


def kats_prophet(historical_df, df_x, df_y, start, end):
    """ historical_df must be formatted ['time', 'data'] """

    # need number of days to project (days diff from start -> end)
    dates_df = pd.DataFrame([[start, end]], columns=['start', 'end'])
    dates_df['td'] = dates_df['end']-dates_df['start']
    days_diff = dates_df['td'].iloc[0].days

    historical_df = historical_df[[df_x, df_y]]  # take only x and y cols
    historical_df = historical_df.rename(columns={df_x: 'time'})  # x column must be named time

    # convert to TimeSeriesData object
    historical_ts = TimeSeriesData(historical_df)

    # create a model param instance
    params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results

    # create a prophet model instance
    m = ProphetModel(historical_ts, params)

    # fit model simply by calling m.fit()
    m.fit()

    # make prediction for next days
    fcst = m.predict(steps=days_diff, freq="D")

    fcst_df = fcst[['time', 'fcst']]
    # rename back to original column names
    fcst_df = fcst_df.rename(columns={'time': df_x, 'fcst': df_y})

    return fcst_df

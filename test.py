from assets import DFAsset, PeriodicAsset, EvalAsset, FAsset
from model import FinModel
from utils import sqlalch_2_df
from df_utils import wrapped_date_range

import tabulate
from dn_bpl import Txn, Category, Account, BplModel
from sqlalchemy.orm import Session
from dateutil.parser import parse
import sqlalchemy as sa
import datetime
import pandas as pd
from validators import Validator
import matplotlib.pyplot as plt
from autots.datasets import load_monthly
from autots import AutoTS
import autots.models.statsmodels as statsmodels
#
from kats.models.prophet import ProphetModel, ProphetParams
from kats.consts import TimeSeriesData
from dn_qtrade.questrade_2 import QTAccount

# # get banking database models and init connection
# db = BplModel()
# session = Session(db.engine)


def main2():
    # get banking database models and init connection
    db = BplModel()
    session = Session(db.engine)

    qt_acc = QTAccount()
    m1 = FinModel()

    # SAMPLING
    # DEFINE DATES TO SAMPLE
    start = datetime.datetime.strptime('2021-06-01', '%Y-%m-%d')
    end = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
    now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # examples of sampling options
    # sample_at = pd.date_range(start=parse('2022-01-01'), end=datetime.datetime.now(),  freq='W').tolist()
    # sample_at = sample_at + pd.date_range(end=datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), periods=4, freq='M').tolist()
    sample_at1 = pd.date_range(start=start, end=end, freq='D').tolist()

    # ASSETS
    tfsa_all_df = qt_acc.get_asset_balances(sample_at1)
    tfsa = EvalAsset(name='tfsa',
                     interp_type='to_previous',
                     forecast_type='prophet',
                     data_df=tfsa_all_df[['x', 'VFV.TO']],
                     x='x',
                     y='VFV.TO',
                     eval_method='market')

    tfsa.sample(sample_at1)

    dbudgeted = get_budgeted()
    df_budegeted = sqlalch_2_df(dbudgeted)
    budgeted_assets = list()

    for key in df_budegeted.keys():

        new_asset = DFAsset(name=key,
                            interp_type='to_previous',
                            forecast_type='prophet',
                            data_df=df_budegeted[key],
                            target_asset=key + "_budget",
                            x='txn_date',
                            y='txn_amount',
                            cumulative=True)
        # new_asset.to_cumulative(new_asset.y)
        budgeted_assets.append(new_asset)

    m1.add_assets(budgeted_assets)

    base = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    food_budget = PeriodicAsset(name='food_budget',
                                interp_type='linear',
                                active_dates=[datetime.datetime.strptime('2020-06-01', '%Y-%m-%d'),
                                              datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'),
                                              datetime.datetime.strptime('2022-01-01', '%Y-%m-%d')],
                                amounts=[-450, -500, -550],
                                period_unit='M',
                                period_size=1,)
    fuel_budget = PeriodicAsset(name='fuel_budget',
                                interp_type='linear',
                                active_dates=[datetime.datetime.strptime('2020-06-01', '%Y-%m-%d')],
                                amounts=[-150],
                                period_unit='M',
                                period_size=1)
    rent_budget = PeriodicAsset(name='rent_budget',
                                interp_type='to_previous',
                                active_dates=[datetime.datetime.strptime('2020-06-01', '%Y-%m-%d'),
                                              datetime.datetime.strptime('2021-01-01', '%Y-%m-%d')],
                                amounts=[-600, -600],
                                period_unit='MS',
                                period_size=1,)

    # m1.add_assets([food_budget, fuel_budget, rent_budget])
    m1.add_assets([food_budget, fuel_budget, rent_budget])

    # SAMPLE AND GRAPH
    # rent = m1.assets['rent'].sample(sample_at1)
    # rent_budget = m1.assets['rent_budget'].sample(sample_at1)
    # food_sample1 = m1.assets['food'].sample(sample_at1, norm_freq='YS')
    # food_budget_sample1 = m1.assets['food_budget'].sample(sample_at1, norm_freq='MS')

    # VALIDATE
    # v_df1 = m1.validate_model(sample_at1, norm_freq='MS')
    # v_df2 = m1.validate_model(sample_at1, norm_freq='YS')

    # GRAPHING
    m1.plot_assets(sample_at1, assets=['food', 'food_budget'], norm_freq='MS')

    print()


def kats_test(model):
    m1 = model
    food_df = m1.assets['food'].data
    food_asset = m1.assets['food']
    food_df = food_df.rename(columns={'food_x': 'time'})
    food_ts = TimeSeriesData(food_df[['time', 'food_y']])

    # create a model param instance
    params = ProphetParams(seasonality_mode='multiplicative')  # additive mode gives worse results

    # create a prophet model instance
    m = ProphetModel(food_ts, params)

    # fit model simply by calling m.fit()
    m.fit()

    # make prediction for next 30 month
    fcst = m.predict(steps=120, freq="D")
    # sample_df[self.x_col] = pd.DatetimeIndex(sample_df[self.x_col])
    fcst['time'] = pd.DatetimeIndex(fcst['time'])

    ax = plt.gca()

    food_df.plot(kind='line', x='time', y=food_asset.y, ax=ax, label='food')
    fcst.plot(kind='line', x='time', y='fcst', ax=ax, label='forecast')

    plt.legend()
    plt.grid()
    plt.show()


def atspy_ARIMA(df):
    food_asset = df
    sample = food_asset.data
    sample = sample[['food_y']]

    # # convert to TimeSeriesData object
    # air_passengers_ts = TimeSeriesData(air_passengers_df)
    #
    # # create a model param instance
    # params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results
    #
    # # create a prophet model instance
    # m = ProphetModel(air_passengers_ts, params)
    #
    # # fit model simply by calling m.fit()
    # m.fit()
    #
    # # make prediction for next 30 month
    # fcst = m.predict(steps=30, freq="MS")
    #
    # print()


def autoTS_optimal_model(df):

    food_asset = df
    sample = food_asset.data

    sample = sample[['food_x', 'food_y']]

    # find best model
    model = AutoTS(
        forecast_length=360,
        frequency='infer',
        ensemble='simple',
        max_generations=5,
        num_validations=len(sample.index),
    )
    model = model.fit(sample, date_col='food_x', value_col='food_y', id_col=None)
    best_model = model

    # Print the name of the best model
    print(model)

    # sample prep
    # sample = sample.loc[:, (food_asset.x_col, food_asset.y_col)]
    # # sample[food_asset.x_col] = pd.DatetimeIndex(sample[food_asset.x_col])
    # # sample.set_index(sample[food_asset.x_col], inplace=True)
    #
    # #
    # model2 = statsmodels.UnobservedComponents(
    #     fillna='ffill',
    #     transformations={'0': 'ClipOutliers', '1': 'DifferencedTransformer', '2': 'DatepartRegression', '3': 'MinMaxScaler', '4': 'HPFilter'},
    #     transformation_params={'0': {'method': 'clip', 'std_threshold': 3.5, 'fillna': None}, '1': {}, '2': {'regression_model': {'model': 'DecisionTree', 'model_params': {'max_depth': 3, 'min_samples_split': 1.0}}, 'datepart_method': 'expanded', 'polynomial_degree': None}, '3': {}, '4': {'part': 'trend', 'lamb': 1600}},
    #     level='fixed intercept',
    #     maxiter=100,
    #     cov_type='opg',
    #     method='powell',
    #     autoregressive=None,
    #     regression_type=None
    # )
    # model2 = model2.fit(sample[[food_asset.y_col]])
    # best_model = model2
    #
    forecast = best_model.predict(forecast_length=30).get_forecast

    ax = plt.gca()

    sample.food_y.plot(kind='line', y=food_asset.y, ax=ax, label='food')

    # forecast swap index from integer to dates starting at sample end date
    # last_index = sample.last_valid_index()
    # forecast_index = pd.date_range(start=last_index, periods=len(forecast), freq='D').tolist()
    # forecast.index = forecast_index

    forecast.food_y.plot(kind='line', ax=ax, label='forecast')

    plt.legend()
    plt.grid()
    plt.show()


def plot_scatter(assets, sample_at):

    ax = plt.gca()
    for key in assets.keys():
        asset = assets[key]

        if type(asset) == PeriodicAsset:
            sample = asset.sample(sample_at, 'linear', norm=True)
        elif type(asset) == DFAsset:
            sample = asset.sample(sample_at, 'to_previous', norm=True)

        sample.plot(kind='line', x=asset.x, y=asset.y, ax=ax, label=key)

    plt.legend()
    plt.show()


def get_budgeted(budgeted=None):
    """ returns dict of lists of txn_objects with categories as keys """

    dbudgeted = dict()
    if not budgeted:
        budgeted = ["food", "fuel", "rent"]
        # budgeted = ["food",]

    for category in budgeted:
        obudg_txns = (session.query(Txn)
                      .join(Category)
                      .filter(sa.or_(Category.id == category, Category.cat_desc == category))
                      ).all()
        dbudgeted[category] = obudg_txns

    return dbudgeted


def forecast_test():
    from sources import YahooMarket
    from forecast import KatsProphet

    # DEFINE DATES TO SAMPLE
    start = datetime.datetime.strptime('2000-06-12', '%Y-%m-%d')
    end = datetime.datetime.strptime('2022-07-08', '%Y-%m-%d')
    now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fcst = end = datetime.datetime.strptime('2023-07-08', '%Y-%m-%d')

    # DEFINE SOURCES AND SAMPLES
    vfv_price_source = YahooMarket(name='yahoo_market', ccy='CAD', tckr='VFV.TO')
    vfv_training_sample = vfv_price_source.sample(start, now)

    # FORECASTING
    market_kats_forecast = KatsProphet('market_kats')
    market_kats_forecast.get_forecast(now, fcst, vfv_training_sample[['Close']])
    fcst = market_kats_forecast.forecast

    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot_date(vfv_training_sample.index, vfv_training_sample[vfv_price_source.ylbl], '-', label=vfv_price_source.name)
    ax.plot_date(fcst.index, fcst[vfv_price_source.ylbl], '-', label=market_kats_forecast.name)

    plt.legend()
    plt.grid()
    plt.show()
    print()

def new_main():
    # TODO use assets as sources to form sub-assets/asset groups
    # TODO ASSETS vs CASHFLOWS - new paradigm??
    # TODO normalize samples? do we need this really?
    from sources import YahooMarket, Bpl_Txns
    from forecast import KatsProphet

    # DEFINE DATES TO SAMPLE
    start = datetime.datetime.strptime('2021-06-12', '%Y-%m-%d')
    end = datetime.datetime.strptime('2022-12-31', '%Y-%m-%d')
    now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fcst = datetime.datetime.strptime('2023-07-08', '%Y-%m-%d')

    # DATE SAMPLES
    sample_at1 = pd.date_range(start=start, end=end, freq='MS').tolist()

    # DEFINE SOURCES AND SAMPLES
    food_spending_src = Bpl_Txns(name='food_spend', ccy_native='CAD', category='food')
    fuel_spending_src = Bpl_Txns(name='fuel_spend', ccy_native='CAD', category='fuel')

    # FORECASTING
    # market_kats_forecast = KatsProphet('market_kats')
    kats_forecast = KatsProphet('banking_kats')

    # ASSETS
    food_spending_asset = FAsset(name='food_spending',
                                 interp_type='to_previous',
                                 sources=[food_spending_src],
                                 forecast=kats_forecast,
                                 cumulative=True)
    fuel_spending_asset = FAsset(name='fuel_spending',
                                 interp_type='to_previous',
                                 sources=[fuel_spending_src],
                                 forecast=kats_forecast,
                                 cumulative=True)

    # MODEL
    model_1 = FinModel()
    model_1.add_assets([food_spending_asset, fuel_spending_asset])
    # TODO normalize option in date_range?
    model_1.plot_assets(sample_at=pd.date_range(start=start, end=end, freq='1W'), sub=True)


if __name__ == '__main__':
    new_main()


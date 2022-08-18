from assets import FAsset
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

from sources import YahooMarket, DbSource, Periodic
from forecast import KatsProphet, Static, Source, ZeroForecast
from model import FinModel


def main():
    # SAMPLING DATES
    start = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    trans = datetime.datetime.strptime('2021-04-13', '%Y-%m-%d')
    end = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
    now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fcst = datetime.datetime.strptime('2023-07-08', '%Y-%m-%d')

    # DATE SAMPLES
    sample_at1 = pd.date_range(start=start, end=end, freq='1D')

    ba = banking_asset_prod()
    ba_sa = ba.sample(sample_at1, src_final=True, sources=['food_spend'])
    plot_df(ba_sa)

    return


def prod_model():
    m = FinModel(name='prod')
    m.add_assets([banking_asset_prod()])

    return m


def banking_asset_prod():

    # FORECASTING
    kats_fcst = KatsProphet(name='standard_kats')

    # budgets
    food_budget_src = Periodic(name='food_budget',
                               interp='linear',
                               amount=-400,
                               period_unit='M',
                               period_size=1,
                               cumulative=False)

    gear_budget_src = Periodic(name='gear_budget',
                               interp='linear',
                               amount=-200,
                               period_unit='A',
                               period_size=1,
                               cumulative=False)

    rent_budget_src = Periodic(name='rent_budget',
                               interp='linear',
                               amount=0,
                               transition_pairs=[[datetime.datetime.strptime('2022-09-01', '%Y-%m-%d'), -750]],
                               period_unit='M',
                               period_size=1,
                               cumulative=False)

    util_budget_src = Periodic(name='util_budget',
                               interp='linear',
                               amount=-200,
                               period_unit='M',
                               period_size=1,
                               cumulative=False)

    income_fcst_src = Periodic(name='income_fcst',
                               interp='linear',
                               amount=1000,
                               period_unit='M',
                               period_size=1,
                               cumulative=False)

    # DEFINE SOURCES
    db = BplModel()
    session = Session(db.engine)
    refined_categories = ['food',
                          'fuel',
                          'rent',
                          'auto',
                          'util',
                          'investing',
                          'crypto',
                          'income',
                          'student loan',
                          'gear']
    filters_remaining = [Category.cat_desc != x for x in refined_categories]

    main_income_src = DbSource(name='main_income',
                               ccy_native='CAD',
                               interp='to_previous',
                               session=session,
                               table=Txn,
                               index='txn_date',
                               joins=[Category],
                               filters=[Category.cat_desc == 'income'],
                               forecast=Source(name=income_fcst_src.name, source=income_fcst_src),
                               cumulative=True,
                               duplicate_indices='sum')

    food_spending_src = DbSource(name='food_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'food'],
                                 forecast=Source(name=food_budget_src.name, source=food_budget_src),
                                 cumulative=True,
                                 duplicate_indices='sum')

    gear_spending_src = DbSource(name='gear_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'gear'],
                                 forecast=Source(name=gear_budget_src.name, source=gear_budget_src),
                                 cumulative=True,
                                 duplicate_indices='sum')

    rent_spending_src = DbSource(name='rent_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'rent'],
                                 forecast=Source(name=rent_budget_src.name, source=rent_budget_src),
                                 cumulative=True,
                                 duplicate_indices='sum')

    util_spending_src = DbSource(name='util_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'utility'],
                                 forecast=Source(name=util_budget_src.name, source=util_budget_src),
                                 cumulative=True,
                                 duplicate_indices='sum')

    fuel_spending_src = DbSource(name='fuel_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'fuel'],
                                 forecast=None,
                                 cumulative=True,
                                 duplicate_indices='sum')

    auto_spending_src = DbSource(name='auto_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'auto'],
                                 forecast=None,
                                 cumulative=True,
                                 duplicate_indices='sum')

    stdn_loan_rep_src = DbSource(name='student_loan_payments',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'student loan'],
                                 forecast=None,
                                 cumulative=True,
                                 duplicate_indices='sum')

    inve_spending_src = DbSource(name='invest_transfers',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'investing'],
                                 forecast=None,
                                 cumulative=True,
                                 duplicate_indices='sum')

    cryp_spending_src = DbSource(name='crypto_transfers',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=[Category.cat_desc == 'crypto'],
                                 forecast=None,
                                 cumulative=True,
                                 duplicate_indices='sum')

    # all non-refined banking categories
    misc_spending_src = DbSource(name='misc_spend',
                                 ccy_native='CAD',
                                 interp='to_previous',
                                 session=session,
                                 table=Txn,
                                 index='txn_date',
                                 joins=[Category],
                                 filters=filters_remaining,
                                 forecast=kats_fcst,
                                 cumulative=True,
                                 duplicate_indices='sum')

    # base account adjustments
    all_acc_adj = DbSource(name='all_acc_adj',
                           ccy_native='CAD',
                           interp='to_previous',
                           session=session,
                           table=Account,
                           index='acc_adj_date',
                           ylbl='acc_adj',
                           cumulative=True,
                           duplicate_indices='sum')

    # ASSEMBLE BANKING ASSET FROM SOURCES

    banking_all = FAsset(name='banking_prod', interp_type='zero',
                         sources=[misc_spending_src,
                                  all_acc_adj,
                                  food_spending_src,
                                  fuel_spending_src,
                                  auto_spending_src,
                                  util_spending_src,
                                  cryp_spending_src,
                                  inve_spending_src,
                                  main_income_src,
                                  stdn_loan_rep_src,
                                  gear_spending_src,
                                  rent_spending_src],
                         cumulative=False)

    return banking_all


def plot_df(df):
    """ plot assets using matplotlib at requested sample points """

    # if not isinstance(dfs, list):
    #     dfs = [dfs]
    # if not isinstance(cols, list):
    #     cols = [cols]

    # num_colors = len(cols)
    num_colors = len(df.columns)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle('color', [cm(1.*i/num_colors) for i in range(num_colors)])

    # for df in dfs:
    for col in df.columns:
        # if col in df.columns:
        ax.plot_date(df.index, df[col], '*', label=col)  # add asset output

    plt.legend()
    plt.grid()
    plt.show()

    return


if __name__ == '__main__':
    main()
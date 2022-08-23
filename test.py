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
from processing import process, df_condenser, flatten
from prod_models import banking_asset_prod
from interpolation import interp_zero

from sources import YahooMarket, DbSource, Periodic
from forecast import KatsProphet, Static, Source, ZeroForecast
from model import FinModel
from interpolation import interp_zero
import numpy as np
import copy

# # get banking database models and init connection
# db = BplModel()
# session = Session(db.engine)

# SAMPLING DATES
start = datetime.datetime.strptime('2021-01-01', '%Y-%m-%d')
trans = datetime.datetime.strptime('2021-04-13', '%Y-%m-%d')
end = datetime.datetime.strptime('2022-10-01', '%Y-%m-%d')
now = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
fcst = datetime.datetime.strptime('2023-07-08', '%Y-%m-%d')

# DATE SAMPLES
sample_at1 = pd.date_range(start=start, end=end, freq='1D')


def assembly_testing():
    # b = banking_asset_prod()
    #
    # food_budget_src = Periodic(name='food_budget',
    #                            interp='linear',
    #                            amount=-400,
    #                            period_unit='M',
    #                            period_size=1,
    #                            cumulative=False)
    # db = BplModel()
    # session = Session(db.engine)
    # food_spending_src = DbSource(name='food_spend',
    #                              ccy_native='CAD',
    #                              interp='to_previous',
    #                              session=session,
    #                              table=Txn,
    #                              index='txn_date',
    #                              joins=[Category],
    #                              filters=[Category.cat_desc == 'food'],
    #                              forecast=Source(name=food_budget_src.name, source=food_budget_src),
    #                              cumulative=True,
    #                              duplicate_indices='sum')
    #
    # subs = [food_spending_src.sample(start, end)[[food_spending_src.lbls['base']]],
    #         food_spending_src.sample(start, end)[[food_spending_src.lbls['fcst']]]]
    #
    # # combine requested index for assembly from all dfs
    # to_list = flatten(subs)
    # to_idx = pd.DatetimeIndex(next(to_list).index)
    # to_idx = [to_idx.union(x.index) for x in to_list][0]
    #
    # idx_df = pd.DataFrame({'index': to_idx, 'rando': np.nan})
    # idx_df.set_index('index', drop=True, inplace=True)
    #
    # sample_df = pd.DataFrame({'index': sample_at1, 'rando': np.nan})
    # sample_df.set_index('index', drop=True, inplace=True)
    #
    # all_idx_df = pd.merge(sample_df, idx_df, right_index=True, left_index=True, how='outer')
    #
    # subs.append(all_idx_df)
    # # subs.append(sample_df)
    # # could also interp future seperately? would need to ensure interp is confined to each
    # # individual sample where its applied?
    # subs.append(interp_zero(copy.copy(pd.merge(subs[0], all_idx_df, right_index=True, left_index=True, how='outer')), food_spending_src.lbls['base']))

    subs = [[[1, 2, 3], [1, 2, 3]], [1, 2, 3]]
    a = flatten(subs, mod_func=addemup)
    a = list(a)

    a = process(subs, df_condenser)

    # subs2 = [a, interp_zero(a, a.columns[0])]

    return

def addemup(xs):
    y = 0
    for x in xs:
        y += x

    return y
# maybe a general py utils file is in store
# from collections.abc import Iterable
# def flatten(xs):
#     for x in xs:
#         if isinstance(x, Iterable) and not isinstance(x, pd.DatetimeIndex) and not isinstance(x, pd.DataFrame)\
#                 and not isinstance(x, (str, bytes)):
#             yield from flatten(x)
#         else:
#             yield x


def main():
    assembly_testing()
    pass


if __name__ == '__main__':
    main()


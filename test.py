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
from interpolation import interp_zero, interp_prev
import numpy as np
import copy
from assembly import DFAssembly, AssembledDF, SourcedDF, ForecastedDF
from processing import func_runner, df_adder, df_condenser

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

    food_budget_src = Periodic(name='food_budget',
                               interp='linear',
                               amount=-400,
                               period_unit='M',
                               period_size=1,
                               cumulative=False)

    db = BplModel()
    session = Session(db.engine)
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

    food_sourced = SourcedDF(name='food_sourced',
                             assembler=df_adder,
                             filler=interp_prev,
                             sources=[{'sourcer': food_spending_src.sample,
                                       'lbl': food_spending_src.lbls['base']}])

    food_forecasted = ForecastedDF(name='food_forecasted',
                                   assembler=df_condenser,
                                   filler=interp_prev,
                                   trainer=food_sourced.sample,
                                   forecaster=food_budget_src.sample)

    bank_assembled = AssembledDF(name='bank_all',
                                 assembler=df_condenser,
                                 filler=interp_prev,
                                 subs=[food_forecasted])

    sample_idx = pd.date_range(start, end, freq='1MS')
    sa = bank_assembled.sample(sample_idx)

    print()

    return


def main():
    assembly_testing()
    pass


if __name__ == '__main__':
    main()


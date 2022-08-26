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
from assembly import DFAssembly
from processing import func_runner, df_adder

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

    test_child1 = DFAssembly(name='tc1',
                             sources=[food_budget_src],
                             source_assembler=df_adder)

    test_child2 = DFAssembly(name='tc2',
                             sources=[food_budget_src,
                                      food_budget_src],
                             source_assembler=df_adder)

    test_child3 = DFAssembly(name='tc3',
                             sources=[food_budget_src,
                                      food_budget_src,
                                      food_budget_src],
                             source_assembler=df_adder)

    sub_parent1 = DFAssembly(name='sp1',
                             sources=[food_budget_src,
                                      test_child1],
                             source_assembler=df_adder)

    sub_parent2 = DFAssembly(name='sp2',
                             sources=[test_child2,
                                      test_child3],
                             source_assembler=df_adder)

    test_parent = DFAssembly(name='test',
                             sources=[test_child1,
                                      sub_parent1,
                                      sub_parent2],
                             source_assembler=df_adder)

    sa = test_parent.sample(start, end)

    print()

    return


def main():
    assembly_testing()
    pass


if __name__ == '__main__':
    main()


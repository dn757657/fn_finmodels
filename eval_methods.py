import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from cryptocmd import CmcScraper
from forex_python.converter import CurrencyRates
# internal

from dn_pricing.pricing import get_all_yahoo
from dn_qtrade.questrade_2 import QTAccount


def invest_eval(assets_df, ccy='CAD'):
    """
    output value of all quans by concat quan_ccy col name
    output original df with total col
    """
    rates = CurrencyRates()
    qt = QTAccount()

    # eval each asset in assets_df assets_df columns are presuemed to be quantites of the column label
    for col in assets_df.columns:
        # do no eval non numeric cols
        if is_numeric_dtype(assets_df[col]):
            try:
                df = get_all_yahoo(col)  # get pricing
            except:
                df = get_all_yahoo(col[:-1])  # sometimes yahoo cuts off the last letter and it doesnt work

            df = df[df.index.isin(assets_df['x'])]  # get requested dates only
            df = df[['Close']]  # only need these cols

            # get asset native currency via txns
            native_asset_ccy = None
            ticker = qt.conn.ticker_information(col)
            if 'currency' in ticker.keys():
                native_asset_ccy = ticker['currency']

            # cyrptocmd uses USD by default, so if other ccy requested, convert
            if ccy != native_asset_ccy:
                df['rate'] = np.nan  # blank col
                df['rate'].iloc[0] = rates.get_rate('USD', ccy)  # get conversion rate
                df = df.fillna(method='ffill')  # fill rate forwards
                df['Close'] = df['Close']*df['rate']  # convert Close to requested ccy

            # convert cols to requested ccy and drop rate cols
            assets_df = pd.concat([assets_df.set_index('x'), df['Close']], axis=1, join='inner').reset_index()
            assets_df.rename(columns={'Close': col + "_unit", "index": 'x'}, inplace=True)
            assets_df[col + "_" + ccy] = assets_df[col + "_unit"] * assets_df[col]
            assets_df.drop(columns=[col + "_unit"], inplace=True)

            # add totals
            if 'total' not in assets_df.columns:
                assets_df['total'] = assets_df[col + "_" + ccy]
            else:
                assets_df['total'] = assets_df['total'] + assets_df[col + "_" + ccy]
    return assets_df


def crypto_eval(assets_df, ccy='CAD'):
        """ assume cryptocmd lib uses USD as default
        output value of all quans by concat quan_ccy col name
        output original df with total col
        """
        rates = CurrencyRates()

        # eval each asset in assets_df assets_df columns are presuemed to be quantites of the column label
        for col in assets_df.columns:
            # do no eval non numeric cols
            if is_numeric_dtype(assets_df[col]):
                scraper = CmcScraper(col)
                df = scraper.get_dataframe()  # get pricing
                df = df[df['Date'].isin(assets_df['x'])]  # get requested dates only
                df = df[['Date', 'Close']]  # only need these cols

                # cyrptocmd uses USD by default, so if other ccy requested, convert
                if ccy != 'USD':
                    df['rate'] = np.nan  # blank col
                    df['rate'].iloc[0] = rates.get_rate('USD', ccy)  # get conversion rate
                    df = df.fillna(method='ffill')  # fill rate forwards
                    df['Close'] = df['Close']*df['rate']  # convert Close to requested ccy

                # convert cols to requested ccy and drop rate cols
                assets_df = pd.concat([assets_df.set_index('x'), df[['Date', 'Close']].set_index('Date')], axis=1, join='inner').reset_index()
                assets_df.rename(columns={'Close': col + "_unit", "index": 'x'}, inplace=True)
                assets_df[col + "_" + ccy] = assets_df[col + "_unit"] * assets_df[col]
                assets_df.drop(columns=[col + "_unit"], inplace=True)

                # add totals
                if 'total' not in assets_df.columns:
                    assets_df['total'] = assets_df[col + "_" + ccy]
                else:
                    assets_df['total'] = assets_df['total'] + assets_df[col + "_" + ccy]

        return assets_df
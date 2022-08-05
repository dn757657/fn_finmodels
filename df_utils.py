import pandas as pd
import tabulate
import logging


def wrapped_date_range(samples, freq):
    """ creates pandas date range that encapsulates all dates in list with upper and low bound """
    # TODO issue with passing freq as one item
    # create initial series
    enc_dr = pd.date_range(start=min(samples),
                           end=max(samples),
                           freq=freq)
    if enc_dr.empty:
        enc_dr = pd.date_range(start=min(samples),
                               periods=1,
                               freq=freq)

    try:
        period_size = int(freq[0])
    except ValueError:
        freq = '1' + freq
        period_size = int(freq[0])

    period_unit = freq[1:]
    dr_start = enc_dr[0]
    dr_end = enc_dr[-1]

    period_size_upper = len(enc_dr)
    upper_enc_dr = pd.DatetimeIndex([])
    while dr_end < max(samples):
        # max sample is not encapsulated
        upper_enc_dr = pd.date_range(start=min(samples),
                                     periods=period_size_upper,
                                     freq=freq)
        dr_end = upper_enc_dr[-1]
        period_size_upper += 1

    period_size_lower = 1
    freq_lower = '-1' + period_unit
    lower_enc_dr = pd.DatetimeIndex([])
    while dr_start > min(samples):
        # min sample is not encapsulated
        lower_enc_dr = pd.date_range(start=min(samples),
                                     periods=period_size_lower,
                                     freq=freq_lower)
        dr_start = lower_enc_dr[-1]
        period_size_lower += 1

    enc_dr = enc_dr.union(lower_enc_dr)
    enc_dr = enc_dr.union(upper_enc_dr)
    enc_dr = enc_dr.drop_duplicates()
    enc_dr.sort_values()

    return enc_dr


def del_col_duplicates(del_df, del_col, ctrl_df, ctrl_col):
    """ delete df_del entries from df_del[del_col] contained in df_ctrl[ctrl_col]

    :args
        del_df          dataframe where entries are to be deleted
        ctrl_df         dataframe to indicate duplicate entries to be deleted
        del_col         str column label indicating column to compare
        ctrl_col        str column label indicating column to compare

    :returns
        copy of del_df without any same entries from ctrl_df in the columns indicated
    """

    # exclude_df is del_df values already in ctrl_df
    exclude_df = del_df.loc[del_df[del_col].isin(ctrl_df[ctrl_col])]

    # combine the duplicates (exclude_df) with del_df, creating two of each duplicate
    # then by dropping and not keeping duplicates, they are removed from del_df
    no_duplicate_df = pd.concat([del_df, exclude_df]).drop_duplicates(keep=False)

    return no_duplicate_df


def df1_col_isin_df2_col(df1, col1, df2, col2):
    """ return true if df2[col2] contains df1[col1] """

    # create df of values contained by df1[col1] and df2[col2]
    same_df = df1.loc[df1[col1].isin(df2[col2])]
    # check is all df1 values are in same_df
    result = df1.equals(same_df)

    return result


def df_to_console(df):
    """ pretty print df to console """

    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    print()
    print(tabulate.tabulate(df, headers='keys',))

    return


def incluloc(df, start, end):
    """ df inclusive loc slicing, ensures start and end are contained by df index post slice

    :notes
    - start and end types must match index type
    - data cannot be fabricated, slice will only be included if it exists in original df!
    - in the case that start/end cannot be included nothing is done
    """

    # sliced dfs to create inclusive slice if required
    end_df = df.loc[end:]
    start_df = df.loc[:start]

    idx_min = df.index.min()
    idx_max = df.index.max()

    df = df.loc[start:end]

    # start and end dates MUST be in dataframe
    # if end is not included ad one additional row from original to end of sliced df
    if end not in df.index:
        if idx_max >= end:
            # we can only include the end if the end is included within original df!
            df = pd.concat([df, end_df.iloc[[0]]])

    # if start is not included ad one additional row from original to start of sliced df
    if start not in df.index:
        if idx_min <= start:
            # start cannot be included if it is not in original df
            df = pd.concat([df, start_df.iloc[[-1]]])

    df.sort_index(inplace=True)

    return df

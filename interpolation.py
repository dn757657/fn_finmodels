from pandas.api.types import is_numeric_dtype


def interp_prev(df, ylbl):
    """ set all nan in self.y to precceeding value """

    df.loc[:, ylbl] = df.loc[:, ylbl].fillna(method='ffill')

    return df


def interp_zero(df, ylbl):
    """ all interpolations are always zero """

    df.loc[:, ylbl].fillna(value=0, inplace=True)

    return df


def interp_linear(df, ylbl, interp_index=None):
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

    return df
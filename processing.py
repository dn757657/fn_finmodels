import pandas as pd
import numpy as np
import logging
from typing import List
from collections.abc import Iterable, Callable


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) \
                and not isinstance(x, (str, bytes)) \
                and not isinstance(x, pd.DatetimeIndex) \
                and not isinstance(x, pd.DataFrame):
            yield from flatten(x)
        else:
            yield x


def process(to_process, processor_func, *args):
    """
    condense all lists in condense list using the condenser function

    :param to_process:          list of components or lists of any level to be processed by condenser func
    :param processor_func:      function that takes a list of items and returns a single item
    :return:                    condense list fully condensed to single item
    """

    while any(isinstance(x, list) for x in to_process):
        for i, sub in enumerate(to_process):
            if isinstance(sub, list):
                if any(isinstance(x, list) for x in sub):
                    # recurse for each sub list until no lists
                    process(to_process=sub, processor_func=processor_func)
                else:
                    to_process[i] = processor_func(to_process, *args)
                    break

    processed_df = processor_func(to_process)

    return processed_df


def df_condenser(df_list: List[pd.DataFrame]):
    """
    populate assembly df by merging dfs in list from right to left in native list order -
    then fill leftmost using columns right to left, thereby 'condensing' all dfs into left most column
    using the hierarchy of the list passed

    :param df_list:     list of dataframe to be condensed in order of passed list heirarchy
    :return:            single column dataframe containing condensed data
    """

    condensed_lbl = 'condensed'  # default output lbl
    assembly_df = pd.DataFrame()

    for df in df_list:
        # get df from passed with lbl only
        assembly_df = pd.merge(assembly_df,
                               df,
                               how='outer',
                               right_index=True,
                               left_index=True
                               )
        if df.columns[0] in assembly_df.columns:
            fill_lbl = df.columns[0]
        else:
            # for duplicate lbl names
            fill_lbl = [x for x in assembly_df.columns if df.columns[0] in x
                        and x.endswith('_y')][0]

        # fill or assign in order of df_list (right to left sort of)
        if condensed_lbl in assembly_df:
            assembly_df[condensed_lbl].fillna(assembly_df[fill_lbl], inplace=True)
        else:
            assembly_df[condensed_lbl] = assembly_df[fill_lbl]

    # if out lbl contains null values it could not be condensed completely
    if assembly_df[condensed_lbl].isnull().values.any():
        logging.error(f"Cannot create full assembly from sub-assembly: \n{df_list}")

    return assembly_df[[condensed_lbl]]


def func_runner(func_list: List[Callable], *args):
    """
    run func with passed args
    :param func_list:
    :param args:
    :return:
    """

    run_funcs = list()

    for func in func_list:
        run_funcs.append(func(*args))

    return run_funcs

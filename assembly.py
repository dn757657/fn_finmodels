import pandas as pd
import numpy as np
import logging

class Assembler:
    """ assemble dataframes """

    def __init__(self):
        pass

    def assembler(
            self,
            subs: list,
            out_lbl='out'
    ):
        """
        assemble groups of dataframes into the same index using fillna in order of list priority
        :param subs:
        :param to:
        :param out_lbl:
        :return:
        """

        # combine requested index for assembly from all subs
        to_list = flatten(subs)
        to_idx = pd.DatetimeIndex(next(to_list).index)
        to_idx = [to_idx.union(x.index) for x in to_list][0]

        self.dummy_drill(subs=subs)

        return assembly_df

    def dummy_assembler(self, subs):
        # while any(isinstance(x, list) for x in subs):
        subs = self.dummy_drill(subs=subs)

        final = self.dummy_assemble(suba=subs)
        return final

    def dummy_drill(self, subs):

        while any(isinstance(x, list) for x in subs):
            for i, sub in enumerate(subs):
                if isinstance(sub, list):
                    if any(isinstance(x, list) for x in sub):
                        self.dummy_drill(subs=sub)
                    else:
                        subs[i] = self.dummy_assemble(sub)
                        break

        return subs

    def dummy_assemble(self, suba):
        dummy = 0
        for sub in suba:
            dummy += sub

        return dummy

    def assemble(
            self,
            subs,
            to,
            out_lbl='out',
    ):

        assembly_df = pd.DataFrame({'index': to, out_lbl: np.nan})
        assembly_df.set_index('index', drop=True, inplace=True)

        # fill in list order of priority
        # [{df:, lbl:,}, {df:, lbl:,}]
        for sub in subs:
            # get df from passed with lbl only
            assembly_df = pd.merge(assembly_df,
                                   sub,
                                   how='outer',
                                   right_index=True,
                                   left_index=True
                                   )
            assembly_df.fillna(assembly_df[sub.columns[0]], inplace=True)

        if assembly_df[out_lbl].isnull().values.any():
            logging.error(f"Cannot create full assembly from sub-assembly: \n{subs}")

        return [{'df': assembly_df, 'lbl': out_lbl}]


from collections.abc import Iterable


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) \
                and not isinstance(x, (str, bytes)) \
                and not isinstance(x, pd.DatetimeIndex) \
                and not isinstance(x, pd.DataFrame):
            yield from flatten(x)
        else:
            yield x
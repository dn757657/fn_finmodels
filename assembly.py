from typing import List
from typing import Callable, Iterable
import copy
# from collections.abc import Callable
import pandas as pd
import datetime as dt
from processing import func_runner, df_condenser, process
from functools import reduce


# class SampleFxn(Protocol):
#     def __call__(self,
#                  start: dt.datetime,
#                  end: dt.datetime) -> pd.DataFrame:
#         pass
#
#
# class ForecastFxn(Protocol):
#     def __call__(self,
#                  start: dt.datetime,
#                  end: dt.datetime,
#                  training_data: pd.DataFrame) -> pd.DataFrame:
#         pass

forecast_fxn = Callable[[dt.datetime, dt.datetime, pd.DataFrame], pd.DataFrame]
sample_fxn = Callable[[dt.datetime, dt.datetime], pd.DataFrame]
mod_fxn = Callable[[pd.DataFrame], pd.DataFrame]
assembly_fxn = Callable[[List[pd.DataFrame]], pd.DataFrame]


class SubDF:

    def __init__(self,
                 name: str,
                 assembler,
                 filler,
                 ):

        self.data = pd.DataFrame
        self.name = name

        self._samples = list()
        self._assemlber = assembler
        self.filler = filler

    def sample(self, sample_idx):
        # populate samples

        # assemble
        self.assemble()
        # self.data = self._assemlber(self._samples)

        return self.data

    def assemble(self):
        self.data = process(self._samples, self._assemlber)
        self.data.rename(columns={self.data.columns[0]: self.name}, inplace=True)


class SourcedDF(SubDF):

    def __init__(self,
                 name: str,
                 assembler,
                 filler,
                 sources):

        super().__init__(name=name,
                         assembler=assembler,
                         filler=filler)

        self.sources = sources

    def sample(self, sample_idx):
        self._samples = list()

        self._sample_sources(sample_idx)
        super().sample(sample_idx=sample_idx)

        return self.data

    def _sample_sources(self, sample_idx):
        upper_bound = sample_idx.min()
        lower_bound = sample_idx.max()

        for source in self.sources:
            sourcer = source['sourcer']
            source_lbl = source['lbl']

            source_df = sourcer(upper_bound, lower_bound)
            self._samples.append(source_df[[source_lbl]])


class ForecastedDF(SubDF):

    def __init__(self,
                 name: str,
                 assembler,
                 filler,
                 trainer,
                 forecaster):

        super().__init__(name=name,
                         assembler=assembler,
                         filler=filler)

        self._trainer = trainer
        self._training = pd.DataFrame

        self._forecaster = forecaster
        self._forecast = pd.DataFrame

    def sample(self, sample_idx):
        self._sample_training(sample_idx)
        self._sample_forecast(sample_idx)

        super().sample(sample_idx=sample_idx)

        return self.data

    def _sample_training(self, sample_idx):
        self._training = self._trainer(sample_idx)
        self._samples.append(self._training)

    def _sample_forecast(self, sample_idx):
        """ collect them samples into """

        training_max = sample_idx.max()
        training_min = self._training.index.max()

        if training_max > training_min:
            self._forecast = self._forecaster(start=training_min,
                                              end=training_max,
                                              training=self._training)

        self._samples.append(self._forecast)


class AssembledDF(SubDF):

    def __init__(self,
                 name: str,
                 assembler,
                 filler,
                 subs):

        super().__init__(name=name,
                         assembler=assembler,
                         filler=filler)

        self.subs = subs

        self._incomplete_samples = list()
        self._complete_samples = list()

    def sample(self, sample_idx):
        self._sample_subs(sample_idx=sample_idx)
        self._sample_fill([sample_idx])

        super().sample(sample_idx=sample_idx)

        self.data = self.filler(self.data, self.name)  # fill any blanks not filled by sub fillers
        self.data = self.data[self.data.index.isin(sample_idx)]

        return self.data

    def _sample_subs(self, sample_idx):
        for sub in self.subs:
            self._incomplete_samples.append(sub.sample(sample_idx))

    def _sample_fill(self, idxs: list = None):

        # get all idx by default
        if not idxs:
            idxs = [_flatten_indices(self._samples)]
        else:
            idx_all = _flatten_indices(self._incomplete_samples)
            idxs.append(idx_all)

        idx_final = reduce(lambda x, y: x.union(y), idxs)

        for sub in self.subs:
            # append filled data to sub for assembly
            filled_df = sub.data
            filled_df = assemble_index(filled_df, idx_final)
            filled_df = sub.filler(filled_df, sub.name)
            self._complete_samples.append(filled_df)

        self._samples = self._complete_samples





class DFAssembly:

    def __init__(self,
                 name: str,
                 sources,
                 assembler: assembly_fxn = None,  # TODO None is temporary for testing
                 forecast: forecast_fxn = None,
                 fill: mod_fxn = None,
                 mods: List[mod_fxn] = None,
                 ):

        # params
        self.name = name
        self.sources = sources
        # self.source_assembler = source_assembler
        self.assembler = assembler
        self.forecast = forecast
        self.fill = fill
        self.mods = mods

        # internal
        self._source_data = pd.DataFrame
        self._forecast_data = pd.DataFrame
        self._fill_data = pd.DataFrame

        # external
        self.data = pd.DataFrame

    def sample(self, start, end, **kwargs):
        """
        sample sources, sample forecast, get idx from top level, get fill, assemble, apply mods

        :param sample_idx
        :return:
        """

        if 'freq' in kwargs:
            sample_idx = pd.date_range(start, end, freq=kwargs['freq'])
        else:
            sample_idx = None

        # get sources
        self._sample_sources(start, end, **kwargs)
        self.data = self.assembler(self.data)

        # get forecast
        self._sample_forecast(start, end)
        # get fill
        idx_all = _flatten_indices(self.data)
        self._sample_fill(start, end, [idx_all, sample_idx])

        return self.data

    def _sample_sources(self, start, end, **kwargs):
        # flattened_sources = self._flatten_sources(start, end)

        sampled_sources = list()
        for source in self.sources:
            sampled_sources.append(source.sample(start, end, **kwargs))

        self.data = sampled_sources

        return

    def _sample_forecast(self, start, end):

        if self.forecast:
            source_end = self.data.index.max()

            if source_end < end:
                self._forecast_data = forecast_fxn(start, end, self.data)
                self.data.append(self._forecast_data)

        return

    def _sample_fill(self, start, end, idxs):
        fill_df = self._source_data

        for idx in idxs:
            assemble_index(fill_df, idx)

        fill_df = self.fill(fill_df)
        self.data.append(fill_df)

        return


def assemble_index(
        df,
        assembly_idx: pd.DatetimeIndex):
    """ merge sample_at with main data index such that requested points are contained in index """

    # create df from sample
    new_idx_df = pd.DataFrame(assembly_idx)
    # remove dates from requested sample that are already in sample data
    new_idx_df = new_idx_df[~new_idx_df[0].isin(df.index)]
    # remove requested samples not within the scope of existing index (cannot use interpolation as forecast)
    new_idx_df = new_idx_df.loc[new_idx_df[0] > df.index.min()]
    new_idx_df = new_idx_df.loc[new_idx_df[0] < df.index.max()]
    new_idx_df.set_index(0, drop=True, inplace=True)

    # merge and sort into main asset data
    df = pd.merge(df, new_idx_df, 'outer', left_index=True, right_index=True)
    df = df.sort_index()

    return df


def _flatten_indices(to_flatten):
    """ combine indices of irregular list of dataframes into single index """

    # combine requested index for assembly from all subs
    to_list = flatten(to_flatten)
    to_idx = pd.DatetimeIndex(next(to_list).index)
    if to_list.gi_yieldfrom:
        to_idx = [to_idx.union(x.index) for x in to_list][0]

    return to_idx


# def process(to_process, process_fxn, types_to_process, start, end, *args):
#     """
#     process some shit idk yet this is very confusing
#     """
#
#     while any(isinstance(x, types_to_process) for x in to_process):
#         for i, sub in enumerate(to_process):
#             if isinstance(sub, list):
#                 if any(isinstance(x, types_to_process) for x in sub):
#                     # recurse for each sub list until no lists
#                     process(to_process=sub, process_fxn=process_fxn, types_to_process=types_to_process)
#             else:
#                 sources = sub._sample_sources(start, end)
#                 # to_process[i] = sub.source_assembler(sources)
#                 break
#
#     processed = process_fxn(to_process, *args)
#
#     return processed


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) \
                and not isinstance(x, (str, bytes)) \
                and not isinstance(x, pd.DatetimeIndex) \
                and not isinstance(x, pd.DataFrame):
            yield from flatten(x)
        else:
            yield x
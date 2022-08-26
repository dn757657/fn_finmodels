from typing import List
from typing import Callable, Iterable
import copy
# from collections.abc import Callable
import pandas as pd
import datetime as dt
from processing import func_runner, df_condenser, process


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


class DFAssembly:

    def __init__(self,
                 name: str,
                 sources,
                 source_assembler,
                 assembler: assembly_fxn = None,  # TODO None is temporary for testing
                 forecast: forecast_fxn = None,
                 fill: mod_fxn = None,
                 mods: List[mod_fxn] = None,
                 ):

        # params
        self.name = name
        self.sources = sources
        self.source_assembler = source_assembler
        self.assembler = assembler
        self.forecast = forecast
        self.fill = fill
        self.mods = mods

        # internal
        self._source_data = pd.DataFrame
        self._forecast_data = pd.DataFrame

        # external
        self.data = pd.DataFrame

    def sample(self, start, end):

        # sample and assemble sources
        self._sample_sources(start, end)
        self.data = process(self.data, self.source_assembler)

        self._sample_forecast(start, end)
        # self.data.append(self._forecast_data)

        idx_all = _flatten_indices(self.data)

        return self.data

    def _sample_sources(self, start, end):
        # flattened_sources = self._flatten_sources(start, end)

        sampled_sources = list()
        for source in self.sources:
            sampled_sources.append(source.sample(start, end))

        self.data = sampled_sources

        return

    def _sample_forecast(self, start, end):

        if self.forecast:
            source_end = self.data.index.max()

            if source_end < end:
                self._forecast_data = forecast_fxn(start, end, self.data)
                self.data.append(self._forecast_data)

        return


def _flatten_indices(to_flatten):
    """ combine indices of irregular list of dataframes into single index """

    # combine requested index for assembly from all subs
    to_list = flatten(to_flatten)
    to_idx = pd.DatetimeIndex(next(to_list).index)
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
from typing import List, Dict, TypedDict
from collections.abc import Callable
import pandas as pd
import datetime as dt
from processing import process, func_runner, df_condenser


sample_func = Callable[[dt.datetime, dt.datetime], pd.DataFrame]


class Source:

    def __init__(self,
                 raw_sources: List[sample_func],
                 forecast: sample_func,
                 interpolator: sample_func):
        # inputs
        self.raw_sources = raw_sources
        self.forecast = forecast
        self.interpolator = interpolator

        # internal
        self._assembly = list()

    def assemble(self):
        self._assembly.append(self.raw_sources)
        self._assembly.append(self.forecast)
        self._assembly.append(self.interpolator)

        pass

    def process(self, start, end):
        sampled = process(self._assembly, func_runner, start, end)
        processed = process(sampled, df_condenser)

        # TODO process mods funcs, functions to modify processed data, df in df out
        # TODO assemble everything as functions and execute at endd
        # TODO incorporate samplefuncs into forecasting? samplefunc as training

        return processed

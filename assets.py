import pandas as pd
from interpolation import interp_prev, interp_zero, interp_linear
import copy

# object declaration imports
from sources import FSource


class FAsset:
    """
    Collection of sources with added sampling utility
    """

    def __init__(
            self,
            name: str,
            sources: list,
            **kwargs):
        """
        :param name:        asset name
        :param sources:     list of FSource objects
        :param kwargs:
        """

        # set attributes
        self.name = name
        self.sources = sources

        # generated/output
        self.data = pd.DataFrame()

        # internal
        self.ylbl = self.name+'_y'

        # alias source y labels to prevent duplicates
        self.aliased_lbls = self.alias_lbls(self.sources)
        # add kwargs to attributes
        self.__dict__.update(kwargs)

    @ staticmethod
    def alias_lbls(objects):
        aliased = dict()

        if objects:
            for obj in objects:
                aliased[obj.name] = copy.copy(obj.lbls)  # get labels from source object

                # alias labels to prevent duplication
                for key in obj.lbls:
                    aliased[obj.name][key] = obj.name + "_" + obj.lbls[key]

        return aliased

    def sample(
            self,
            sample_at: pd.DatetimeIndex):
        """
        sample aggregate asset components and return dataframe sample

        :param sample_at:   pandas compatible date-type series
        :return: df containing source sample data and asset sample data at the specified dates in sample_at
        """

        self.sample_sources(sample_at=sample_at)  # populate sources with data
        self.assemble(sample_at=sample_at)  # populate asset with data per sources
        self.data = self.data[self.data.index.isin(sample_at)]  # return requested sample dates

        # remove garbage columns
        all_lbls = self.flatten_lbls(True, True, True, True)
        self.data = self.data[all_lbls]

        return self.data

    def sample_sources(
            self,
            sample_at: pd.DatetimeIndex):
        """ sample asset sources """

        self.sample_components(sample_at=sample_at,
                               components=self.sources,)

        return

    def sample_components(
            self,
            sample_at: pd.DatetimeIndex,
            components):
        """
        components data are populated per sample requests
        :param sample_at:
        :param components:
        """

        # prep/parse sample axis
        sample_at = pd.Series(sample_at)
        sample_start = sample_at.min()
        sample_end = sample_at.max()

        for comp in components:
            # sample the source - sample func returns sample but it is also stored in source.data
            comp.sample(start=sample_start, end=sample_end)

        return

    def assemble(
            self,
            sample_at: pd.DatetimeIndex):
        """ assemble source samples into asset data and add to asset ylbl """

        self.assemble_sources()
        self.assemble_index(sample_at=sample_at)
        self.assemble_ylbl()

        return

    def assemble_sources(self):
        """ merge available source data into asset main data and rename labels as aliased """

        for source in self.sources:
            # assemble source columns to asset data
            for key, lbl in source.lbls.items():
                if lbl in source.data.columns:  # sometimes no forecast col
                    # merge and rename source col as aliased
                    self.data = self.data = pd.merge(self.data,
                                                     source.data[lbl],
                                                     left_index=True, right_index=True, how='outer')
                    self.data = self.data.rename(columns={lbl: self.aliased_lbls[source.name][key]})

        return

    def assemble_index(
            self,
            sample_at: pd.DatetimeIndex):
        """ merge sample_at with main data index such that requested points are contained in index """

        # create df from sample
        new_idx_df = pd.DataFrame(sample_at)
        # remove dates from requested sample that are already in sample data
        parsed_sample_at_df = new_idx_df[~new_idx_df[0].isin(self.data.index)]
        # remove requested samples not within the scope of existing index (cannot use interpolation as forecast)
        parsed_sample_at_df = parsed_sample_at_df.loc[parsed_sample_at_df[0] > self.data.index.min()]
        parsed_sample_at_df = parsed_sample_at_df.loc[parsed_sample_at_df[0] < self.data.index.max()]
        parsed_sample_at_df.set_index(0, drop=True, inplace=True)

        # merge and sort into main asset data
        self.data = pd.merge(self.data, parsed_sample_at_df, 'outer', left_index=True, right_index=True)
        self.data = self.data.sort_index()

        return

    def assemble_ylbl(self):
        """ assemble all source data into main asset data label """

        for source in self.sources:
            # interp source data to match index of asset ylbl - use temp col as to not disturb original data
            source_final_lbl = self.aliased_lbls[source.name]['final']
            self.data['final_temp'] = self.data[source_final_lbl]
            self.interpolate(source.interp, self.data, 'final_temp')

            if self.ylbl in self.data.columns:
                self.data[self.ylbl] = self.data[self.ylbl] + self.data['final_temp']
            else:
                self.data[self.ylbl] = self.data['final_temp']

        return

    def interpolate(
            self,
            method: str,
            data: pd.DataFrame,
            ylbl: str):
        """
        interpolate requested column using index in data
        :param method:          string name of interpolation type
        :param data:            dataframe containing data to interpolate
        :param ylbl:            lbl to interpolate
        :return:
        """

        # may have utility in the future to add speed but right now this is not needed
        # # only want to apply interpolation on relevant data so we get min and max of existing sample
        # # get all index where main ylbl is not nan
        # value_indexes = self.data.index[~self.data[ylbl].isna()]
        # min_index = value_indexes.min()
        # max_index = value_indexes.max()

        # interpolate missing values from sample
        if method == 'to_previous':
            self.data = interp_prev(self.data, ylbl)
        elif method == 'linear':
            self.data = interp_linear(self.data, ylbl)
        elif method == 'zero':
            self.data = interp_zero(self.data, ylbl)

        return data

    def flatten_lbls(self, final=True, base=False, fcst=False, asset=True):
        flat_lbls = list()

        for lbls in self.aliased_lbls.values():
            for key, lbl in lbls.items():
                if key == 'final' and final and lbl in self.data.columns:
                    flat_lbls.append(lbl)
                if key == 'base' and base and lbl in self.data.columns:
                    flat_lbls.append(lbl)
                if key == 'fcst' and fcst and lbl in self.data.columns:
                    flat_lbls.append(lbl)
        if asset:
            flat_lbls.append(self.ylbl)

        return flat_lbls

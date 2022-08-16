import pandas as pd
from interpolation import interp_prev, interp_zero, interp_linear


# TODO restructure aliases flatter and better access, maybe dict of original:aliased setup?
class FAsset:
    """
    Collection of sources with added sampling utility
    """

    def __init__(self, name, interp_type, sources, **kwargs):
        """

        :param name:
        :param interp_type:
        :param sources:
        :param kwargs:
        """

        if not isinstance(sources, list):
            sources = [sources]

        # set attributes
        self.name = name
        self.sources = sources

        # generated/output
        self.data = pd.DataFrame()

        # internal
        self.ylbl = self.name+'_y'
        self.interp_type = interp_type
        # alias source y labels to prevent duplicates
        self.aliased_ylbls = self.alias_ylbls(self.sources)
        # add kwargs to attributes
        self.__dict__.update(kwargs)

    def alias_ylbls(self, objects):
        aliased = dict()

        if objects:
            for obj in objects:
                aliased[obj.name] = dict()
                aliased[obj.name]['final'] = obj.name + "_" + obj.ylbl
                aliased[obj.name]['base'] = obj.name + "_" + obj.ylbl_base
                aliased[obj.name]['fcst'] = obj.name + "_" + obj.ylbl_fcst

        return aliased

    def sample(self, sample_at):
        """
        :param sample_at:   pandas compatible date-type series
        :return: df containing source sample data and asset sample data at the specified dates in sample_at
        """

        self.sample_sources(sample_at=sample_at)  # populate sources with data
        self.assemble(sample_at=sample_at)  # populate asset with data per sources
        self.data = self.data[self.data.index.isin(sample_at)]  # return requested sample dates

        return self.data

    def sample_sources(self, sample_at):
        """ sample asset sources """

        self.sample_components(sample_at=sample_at,
                               components=self.sources,)

        return

    def sample_components(self, sample_at, components):
        """ components list is sampled, sample is stored within component """

        # prep/parse sample axis
        sample_at = pd.Series(sample_at)
        sample_start = sample_at.min()
        sample_end = sample_at.max()

        for comp in components:
            # sample the source - sample func returns sample but it is also stored in source.data
            comp.sample(start=sample_start, end=sample_end)

        return

    def assemble(self, sample_at):
        """ assemble source samples into asset data and add to asset ylbl """
        self.assemble_sources()
        self.assemble_index(sample_at=sample_at)
        self.assemble_ylbl()

        return

    def assemble_sources(self):
        """ merge available source data into asset main data and rename labels as aliased """

        for source in self.sources:
            # assemble source columns to asset data
            col_aliases = self.aliased_ylbls[source.name]

            self.data = self.data = pd.merge(self.data,
                                             source.data[source.ylbl_base],
                                             left_index=True, right_index=True, how='outer')
            self.data = self.data.rename(columns={source.ylbl_base: col_aliases['base']})
            self.data = self.data = pd.merge(self.data,
                                             source.data[source.ylbl],
                                             left_index=True, right_index=True, how='outer')
            self.data = self.data.rename(columns={source.ylbl: col_aliases['final']})
            if source.ylbl_fcst in source.data:
                self.data = self.data = pd.merge(self.data,
                                                 source.data[source.ylbl_fcst],
                                                 left_index=True, right_index=True, how='outer')
                self.data = self.data.rename(columns={source.ylbl_fcst: col_aliases['fcst']})

        return

    def assemble_index(self, sample_at):
        """ merge sample_at with main data index """

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

        # TODO interpolate self.ylbl first? no stupid it doesnt exist yet

        for source in self.sources:
            # interp source data to match index of asset ylbl
            # source.interpolate(self.data.index)

            self.data['final_temp'] = source.data[source.ylbl]
            self.interpolate(source.interp, self.data, 'final_temp')
            self.interpolate(source.interp, self.data, self.aliased_ylbls[source.name]['final'])

            if self.ylbl in self.data.columns:
                # self.data['final_temp'] = source.data[source.ylbl]
                # self.data['final_temp'].fillna(0, inplace=True)
                # self.interpolate(source.interp, self.data, self.aliased_ylbls[source.name]['final'])

                # self.data[self.ylbl].fillna(0, inplace=True)  # ideally we would also interpolate main ylbl? not fll na
                self.data[self.ylbl] = self.data[self.ylbl] + self.data['final_temp']
            else:
                self.data[self.ylbl] = self.data['final_temp']

        return

    # def _apply_options(self):
    #     """ apply options to sample
    #     :param data             dataframe to apply option to
    #     :param col              col within dataframe where option is applied
    #     """
    #     # if not col:
    #     #     col = self.ylbl  # default col is the ylbl col for source
    #
    #     if hasattr(self, 'cumulative'):
    #         if self.cumulative:
    #             self.data = self._to_cumulative()
    #
    #     # self.data = data
    #     return
    #
    # def _to_cumulative(self):
    #     """ convert y to cumulative sum column """
    #     data = self.data
    #     lbls = [self.ylbl]
    #     lbls.extend(self.unpack_lbls(True, True, True))
    #
    #     # for lbl in lbls:
    #     #     if lbl in self.data:
    #     lbl = self.ylbl
    #     if is_numeric_dtype(self.data[lbl]):
    #         data.loc[:, lbl].fillna(0, inplace=True)  # zeroes for continuity
    #         data.loc[:, lbl] = pd.DataFrame.cumsum(data.loc[:, lbl])
    #     else:
    #         logging.error("cannot sum non-numeric dtype")
    #
    #     return data

    def interpolate(self, method, data, ylbl):
        """ interpolate specified col to new axis using source interp_type """

        # lbls = [self.ylbl, self.ylbl_base, self.ylbl_fcst]

        # # create df from index
        # new_idx_df = pd.DataFrame(new_index)
        # # remove dates from requested sample that are already in sample data
        # parsed_sample_at_df = new_idx_df[~new_idx_df[0].isin(self.data.index)]
        # # parsed_sample_at_df = sample_at_df
        #
        # # remove requested samples not within the scope of existing index (cannot use interpolation as forecast)
        # parsed_sample_at_df = parsed_sample_at_df.loc[parsed_sample_at_df[0] > self.data.index.min()]
        # parsed_sample_at_df = parsed_sample_at_df.loc[parsed_sample_at_df[0] < self.data.index.max()]
        # parsed_sample_at_df.set_index(0, drop=True, inplace=True)
        #
        # self.data = pd.merge(self.data, parsed_sample_at_df, 'outer', left_index=True, right_index=True)
        # self.data = self.data.sort_index()

        # only want to apply interpolation on relevant data so we get min and max of existing sample
        # get all index where main ylbl is not nan
        value_indexes = self.data.index[~self.data[ylbl].isna()]
        min_index = value_indexes.min()
        max_index = value_indexes.max()

        # for ylbl in lbls:
        # if ylbl in self.data:
        # interpolate missing values from sample
        if method == 'to_previous':
            self.data = interp_prev(self.data, ylbl)
        elif method == 'linear':
            self.data = interp_linear(self.data, ylbl)
        elif method == 'zero':
            self.data = interp_zero(self.data, ylbl)

        return data

    def unpack_lbls(self, base=False, fcst=False, final=False):
        unpacked_lbls = list()

        for alias_group in self.aliased_ylbls:
            aliases = self.aliased_ylbls[alias_group]
            if base:
                unpacked_lbls.append(aliases['base'])
            if fcst:
                unpacked_lbls.append(aliases['fcst'])
            if final:
                unpacked_lbls.append(aliases['final'])

        return unpacked_lbls
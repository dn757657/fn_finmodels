import pandas as pd
import tabulate
import logging
from pandas.api.types import is_numeric_dtype

from df_utils import wrapped_date_range
from assets import FAsset
import matplotlib.pyplot as plt
from df_utils import df_to_console


# TODO use all assets to make a big ol cashflow dataframe for validations requering entire cahs flow (such as min account balance)


class FinModel:

    def __init__(self):
        self.assets = dict()
        # self.validators = dict()

    def add_assets(self, assets):
        """
        :param assets:      asset objects
        :param x:           asset.data columns containing x data
        :param y:           asset.data
        :return:
        """
        if not isinstance(assets, list):
            assets = [assets]

        dassets2add = dict()
        for asset in assets:
            dassets2add[asset.name] = asset

        for key in dassets2add:
            self.assets[key] = dassets2add[key]

        return

    # def add_validator(self, validators):
    #     if not isinstance(validators, list):
    #         validators = [validators]
    #
    #     for v in validators:
    #         if self.validate_validator(v):
    #             self.validators[v.name] = v
    #         else:
    #             continue

    # def validate_validator(self, validator):
    #     v = validator
    #     valid = True
    #     if v.validator_name and v.to_validate_name in self.assets.keys():
    #         to_validate_asset = self.assets[v.to_validate_name]
    #         validator_asset = self.assets[v.validator_name]
    #         if to_validate_asset.data is not None:
    #             if v.to_validate_col not in to_validate_asset.data.columns:
    #                 valid = False
    #         elif validator.data is not None:
    #             if v.validator_col not in validator_asset.data.columns:
    #                 valid = False
    #         elif v.to_validate_col not in [to_validate_asset.x, to_validate_asset.y]:
    #             valid = False
    #         elif v.validator_col not in [validator_asset.x, validator_asset.y]:
    #             valid = False
    #     else:
    #         valid = False
    #
    #     return valid
    #
    # def validate(self, sample_at, **kwargs):
    #     """ validate one asset against another using basic operators """
    #
    #     validation_df = pd.DataFrame([])
    #
    #     # validate sample_at
    #     sample_at = FAsset('test', 'test').parse_sample(sample_at)
    #     validation_dfs = list()
    #     for v in self.validators.values():
    #         # cumulative option stes initial sample value to zero and validates all other cumulatively with
    #         # zero as reference
    #         if 'cumulative' in kwargs.keys():
    #             temp = v.validate(sample_at, self.assets[v.to_validate_name], self.assets[v.validator_name])
    #             if not validation_df.empty:
    #                 validation_df = pd.concat([validation_df, temp], axis=1)
    #                 validation_df.rename(columns={'result': v.name}, inplace=True)
    #             else:
    #                 validation_df = pd.DataFrame(temp)
    #                 validation_df.rename(columns={'result': v.name}, inplace=True)
    #
    #         # batch validation sets each batch first sample to zero, and validates cumulatively based on this
    #         # rolling reference
    #         # requires batch size as dataframe.daterange offset alias
    #         elif 'batch' in kwargs.keys():
    #             validation_df = pd.DataFrame([])
    #             if 'size' and 'unit' in kwargs.keys():
    #                 size = kwargs['size']
    #                 unit = kwargs['unit']
    #             else:
    #                 logging.error("must provide size and unit of batch given batch option")
    #                 return
    #
    #             # create dataframe daterange containing all requested dates (aka wrapped date_range)
    #             ranges_df = pd.DataFrame({'ranges': wrapped_date_range(sample_at, str(size) + unit)})
    #             for i in range(1, len(ranges_df['ranges'])):
    #                 lower_lim = ranges_df['ranges'].iloc[i-1]
    #                 upper_lim = ranges_df['ranges'].iloc[i]
    #                 upper_lim_index = None
    #                 lower_lim_index = None
    #
    #                 # get lower_lim_index
    #                 for k in range(0, len(sample_at)):
    #                     sample = sample_at[k]
    #                     lower_lim_index = k
    #                     if lower_lim <= sample:
    #                         break
    #                 # get upper_lim_index
    #                 for k in range(0, len(sample_at)):
    #                     sample = sample_at[k]
    #                     upper_lim_index = k
    #                     if upper_lim <= sample:
    #                         if k != len(sample_at):
    #                             upper_lim_index += 1
    #                         break
    #
    #                 if not upper_lim_index:
    #                     logging.error("limits not found - validation could not be completed")
    #                     return
    #
    #                 if upper_lim_index == len(sample_at)-1:
    #                     sample_subset = sample_at[lower_lim_index:]
    #                 else:
    #                     sample_subset = sample_at[lower_lim_index:upper_lim_index]
    #
    #                 temp = v.validate(sample_subset, self.assets[v.to_validate_name], self.assets[v.validator_name])
    #                 # create single validatation dataframe from all batches in each individual validator
    #                 if not validation_df.empty:
    #
    #                     validation_df = pd.concat([validation_df, temp], axis=1)
    #                     validation_df[v.name] = validation_df[v.name].fillna(validation_df['result'])
    #                     validation_df.drop(columns=['result'], inplace=True)
    #                     validation_df.index.drop_duplicates(keep='first')
    #                 else:
    #                     validation_df = pd.DataFrame(temp)
    #                     validation_df.rename(columns={'result': v.name}, inplace=True)
    #
    #             # create single df from all processed validators
    #             # validation dfs is list of dataframe validations for each validator
    #             validation_dfs.append(validation_df)
    #             if len(validation_dfs) > 1:
    #                 final_v_df = pd.DataFrame([])
    #                 for df in validation_dfs:
    #                     if final_v_df.empty:
    #                         final_v_df = df
    #                     else:
    #                         final_v_df = pd.concat([final_v_df, df], axis=1)
    #                 validation_df = final_v_df
    #
    #     if 'report' in kwargs.keys():
    #         pd.set_option('display.max_rows', None, 'display.max_columns', None)
    #         print()
    #         print(tabulate.tabulate(validation_df, headers='keys',))
    #     else:
    #
    #         return validation_df

    def validate_model(self, validate_at, assets=None, norm_at=None, norm_freq=None, to_console=True):

        if not isinstance(validate_at, list):
            validate_at = [validate_at]

        if not assets:
            assets = self.assets.keys()

        validations_df = pd.DataFrame()
        for asset_key in assets:
            asset_to_validate = self.assets[asset_key]
            if asset_to_validate.target:
                v_new = self.validate_asset(asset_to_validate, validate_at, norm_at=norm_at, norm_freq=norm_freq)
                validations_df = pd.concat([validations_df, v_new], axis=1)

        if to_console:
            df_to_console(validations_df)
        # TODO create total surplus/deficit column once all validations are aggregated
        return validations_df

    def validate_asset(self, asset_to_validate, validate_at, norm_at=None, norm_freq=None):
        # get both assets from model (self)
        to_validate_asset = asset_to_validate
        validator_asset = self.assets[to_validate_asset.target]

        # get samples at requested validation points
        to_validate_sample_df = to_validate_asset.sample(validate_at,
                                                         norm_at=norm_at,
                                                         norm_freq=norm_freq)
        # rename columns for easy reference
        to_validate_col_name = to_validate_asset.name + '_' + to_validate_asset.y
        to_validate_xcol_name = 'x_del'
        to_validate_sample_df = to_validate_sample_df.rename(columns={to_validate_asset.y: to_validate_col_name,
                                                                      to_validate_asset.x: to_validate_xcol_name})
        # get sample at validation points for validator asset comparison
        validator_sample_df = validator_asset.sample(validate_at,
                                                     norm_at=norm_at,
                                                     norm_freq=norm_freq)
        # rename columns for easy reference
        validator_col_name = validator_asset.name + '_' + validator_asset.y
        validator_sample_df = validator_sample_df.rename(columns={validator_asset.y: validator_col_name})

        # combine both samples (keep all columns)
        validate_df = pd.concat([to_validate_sample_df, validator_sample_df], axis=1)

        # creat difference column
        diff_col = to_validate_asset.name + '_diff'
        validate_df[diff_col] = validate_df[to_validate_col_name] - validate_df[validator_col_name]
        validate_df = validate_df.drop(['x_del'], axis=1)

        validate_df = validate_df[[validator_asset.x, to_validate_col_name, validator_col_name, diff_col]]

        # reorder and return with diff column and both asset y columns
        return validate_df

    def plot_assets(self, sample_at, assets=None, norm_freq=None, norm_at=None, sub=None):
        """ plot assets using matplotlib at requested sample points """

        if not assets:
            assets = self.assets.keys()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for a in assets:
            asset = self.assets[a]
            sample = asset.sample(sample_at)
            ax.plot_date(sample.index, sample[asset.ylbl], '-', label=asset.name)  # add asset output

            if sub:  # graph sources within asset as well (sub-assets)
                if len(asset.sources) > 1:
                    for source in asset.sources:
                        ax.plot_date(sample.index,
                                     sample[asset.source_ylbl_aliased[source.name]],
                                     '-', label=source.name)

        plt.legend()
        plt.grid()
        plt.show()

        return

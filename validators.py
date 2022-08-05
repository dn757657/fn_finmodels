import pandas as pd
import matplotlib.pyplot as plt


class Validator:
    def __init__(self,
                 to_validate_name,
                 to_validate_col,
                 validator_name,
                 validator_col,
                 name=None):

        self.to_validate_name = to_validate_name
        self.to_validate_col = to_validate_col

        self.validator_name = validator_name
        self.validator_col = validator_col

        # self.norm_freq = norm_freq
        # self.norm_at = norm_at

        # self.operator = operator

        if not name:
            self.name = self.to_validate_name + '_' + self.validator_name
        else:
            self.name = name

    # def validate(self, validate_at, to_validate_asset, validator_asset, graph=False):
    #     # TODO validate does not work with first value when to_validate len=1
    #     # get samples at requested validation points for both validator and to_validate assets
    #     to_validate_sample_df = to_validate_asset.sample(validate_at, **self.to_validate_sample_params)
    #     to_validate_sample_df[self.to_validate_col] = to_validate_sample_df[self.to_validate_col].abs()
    #
    #     validator_sample_df = validator_asset.sample(validate_at, **self.validator_sample_params)
    #     validator_sample_df[self.validator_col] = validator_sample_df[self.validator_col].abs()
    #
    #     validate_df = pd.concat([to_validate_sample_df, validator_sample_df], axis=1)
    #
    #     # TODO implement more than bool result in future (as func?)
    #     greater_indexes = validate_df[validate_df[self.validator_col] > validate_df[self.to_validate_col]].index
    #     validate_df['result'] = 'Invalid'
    #     validate_df.loc[greater_indexes, 'result'] = 'Valid'
    #
    #     if graph:
    #         ax = plt.gca()
    #         validate_df.plot(kind='line', x=self.to_validate_name + '_x', y=self.to_validate_name + '_y', ax=ax)
    #         validate_df.plot(kind='line', x=self.validator_name + '_x', y=self.validator_name + '_y', ax=ax)
    #
    #         plt.legend()
    #         plt.show()
    #
    #     # TODO option to return difference instead of valid.invalid
    #     return validate_df['result']

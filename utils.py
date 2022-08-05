import pandas as pd
from df_utils import wrapped_date_range


def sqlalch_2_df(sqlalch):
    """ transform SQLAlchemy object(s) to df with same columns as mapped object attributes
     :returns dict of dataframes - same keys as input dict """

    df = pd.DataFrame([t.__dict__ for t in sqlalch])

    return df


def normalize_data(norm_to_asset, to_norm_asset, norm_at=None, norm_freq=None):
    """
    2 components:
        one asset must be the normalizer, which all others are normalized to, it is the 'zero'
        chose points/freq to normalize at

    :arg
        norm_to_asset       asset object to which others are normalized (zero point(s))
        to_norm_asset       asset object to be normalized
        norm_at             list of x column points where data is normalized
        norm_freq           pandas compatible frequency to generate norm_at

    :notes
        for now we assume y is always normalized using x
        must have either norm_at or norm_freq

        whats is normalization?
        -aligning the datasets at given intervals, negating the cumulative effects of a dataset
        -essentially only for cumulative data?

    :strategy
        might be easier to do while sampling? build up sample with norm_at dates then apply norm before returning sample
        generate norm_at if not supplied, use inclusive dates generator func
        norm_at must be available in both assets.data[self.x] - check
        get values of norm_sample at norm_at points by sampling normalizing asset - likely need to assets not just samples! move to sampling
        iterate through them and apply differences in samples by calculating diff at norm_at points
    """

    return

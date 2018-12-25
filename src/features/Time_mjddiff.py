import feather
import pandas as pd
import numpy as np
import sys
import time
import logging
import gc
from pathlib import Path
from sklearn.cluster import KMeans

if Path.cwd().name == 'kaggle_plasticc':
    from .base import Feature
elif Path.cwd().name == 'features':
    from base import Feature


def get_module_logger(modname):
    logger = logging.getLogger(modname)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

""" flux_diff * thres
def get_diff(ts_data):
    ts_data = ts_data.sort_values(["object_id", "passband", "mjd"])
    fluxdiff_maxmin = ts_data.groupby("object_id").agg({"flux": ["max", "min"]})
    fluxdiff_maxmin.columns = ["flux_max", "flux_min"]
    fluxdiff_maxmin.reset_index(drop=False, inplace=True)
    fluxdiff_maxmin["flux_diff"] = fluxdiff_maxmin["flux_max"] - fluxdiff_maxmin["flux_min"]

    thres_list = [0.05, 0.1, 0.2, 0.3, 0.4]
    result_list = []
    for thres in thres_list:
        result = _get_diff(ts_data, fluxdiff_maxmin, thres)
        result_list.append(result)

    result = pd.concat(result_list, axis=1).reset_index(drop=False)
    import pdb; pdb.set_trace()
    return result


def _get_diff(ts_data, fluxdiff_maxmin, thres=0.2):
    import pdb; pdb.set_trace()
    fluxdiff_maxmin["thres"] = fluxdiff_maxmin["flux_diff"] * thres
    ts_data = pd.merge(ts_data, fluxdiff_maxmin, on="object_id")
    ts_data["thres_flg"] = (ts_data["flux"] > ts_data["thres"]).astype(int)

    grp = ts_data.loc[ts_data['thres_flg'] == 1, ['object_id', 'mjd']].groupby('object_id')['mjd']
    result = grp.agg({"mjd": ["max", "min"]})
    result.columns = ["mjd_max", "mjd_min"]
    result[f"mjd_diff_{thres}"] = result["mjd_max"] - result["mjd_min"]

    return result[f"mjd_diff_{thres}"]
"""


def get_diff(ts_data):
    ts_data = ts_data.sort_values(["object_id", "passband", "mjd"])

    q_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    result_list = []
    for q in q_list:
        result = _get_diff(ts_data, q)
        result_list.append(result)

    result = pd.concat(result_list, axis=1).reset_index(drop=False)
    return result


def _get_diff(ts_data, q=0.2):
    thres = ts_data.groupby("object_id")["flux"].quantile(q=0.25)
    thres.name = "thres"
    thres = thres.reset_index()
    ts_data = pd.merge(ts_data, thres, on="object_id")
    ts_data["thres_flg"] = (ts_data["flux"] > ts_data["thres"]).astype(int)

    grp = ts_data.loc[ts_data['thres_flg'] == 1, ['object_id', 'mjd']].groupby('object_id')['mjd']
    result = grp.agg({"mjd": ["max", "min"]})
    result.columns = ["mjd_max", "mjd_min"]
    result[f"mjd_diff_{q}"] = result["mjd_max"] - result["mjd_min"]

    return result[f"mjd_diff_{q}"]


class Time_mjddiff(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        logger = get_module_logger(__name__)

        # make train features
        start = time.time()
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        agg_train = get_diff(train_ts)
        agg_train = pd.merge(train_meta[["object_id"]], agg_train, on="object_id", how="outer")
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train
        logger.debug('train done in %5.1f' % ((time.time() - start) / 60))

        """
        # make features by test-data
        start = time.time()
        chunks = 5000000
        chunk_last = pd.DataFrame()
        test_row_num = 453653104
        total_steps = int(np.ceil(test_row_num / chunks))

        reader = pd.read_csv('../../data/input/test_set.csv', chunksize=chunks, iterator=True)
        for i_c, df_ts in enumerate(reader):
            df_ts = pd.concat([chunk_last, df_ts], ignore_index=True)

            if i_c + 1 < total_steps:
                id_last = df_ts['object_id'].values[-1]
                mask_last = (df_ts['object_id'] == id_last).values
                chunk_last = df_ts[mask_last]
                df_ts = df_ts[~mask_last]

            df_ts = add_fluxfactor(df_ts, test_meta)
            agg_test = get_basic_ts(df_ts)

            # tmp save
            logger.debug('%15d save in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            if i_c == 0:
                agg_test_tmp = agg_test.copy()
            else:
                agg_test_tmp = pd.concat([agg_test_tmp, agg_test], axis=0)

            del agg_test
            gc.collect()
            logger.debug('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        agg_test = pd.merge(test_meta[["object_id"]], agg_test_tmp, on="object_id", how="outer")
        agg_test.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_test.drop('object_id', axis=1, inplace=True)
        self.test_feature = agg_test
        """
        self.test_feature = agg_train

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


def get_basic_ts_passband(ts_data):
    agg_dict = {
        'fluxdiff': ['min', 'max', 'mean', 'median', 'std', 'skew']
    }

    # aggregate
    agg_results = ts_data.groupby(['object_id', 'passband']).agg(agg_dict).unstack()
    column_name = [f'{col[0]}_{col[1]}_{col[2]}' for col in agg_results.columns]
    agg_results.columns = column_name

    passband_list = [0, 1, 2, 3, 4, 5]
    for passband in passband_list:
        agg_results[f'fluxdiff_diff_{passband}'] = agg_results[f'fluxdiff_max_{passband}'] - agg_results[f'fluxdiff_min_{passband}']
        agg_results[f'fluxdiff_dif2_{passband}'] = (agg_results[f'fluxdiff_diff_{passband}']) / agg_results[f'fluxdiff_mean_{passband}']

    agg_results.reset_index(drop=False, inplace=True)
    return agg_results


class Time_mjddiff_passband(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        logger = get_module_logger(__name__)

        # make train features
        start = time.time()
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        agg_train = get_basic_ts_passband(train_ts)
        agg_train = pd.merge(train_meta[["object_id"]], agg_train, on="object_id", how="outer")
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train
        logger.debug('train done in %5.1f' % ((time.time() - start) / 60))

        """
        # make features by test-data
        start = time.time()
        chunks = 5000000
        chunk_last = pd.DataFrame()
        test_row_num = 453653104
        total_steps = int(np.ceil(test_row_num / chunks))

        reader = pd.read_csv('../../data/input/test_set.csv', chunksize=chunks, iterator=True)
        for i_c, df_ts in enumerate(reader):
            df_ts = pd.concat([chunk_last, df_ts], ignore_index=True)

            if i_c + 1 < total_steps:
                id_last = df_ts['object_id'].values[-1]
                mask_last = (df_ts['object_id'] == id_last).values
                chunk_last = df_ts[mask_last]
                df_ts = df_ts[~mask_last]

            df_ts = add_fluxfactor(df_ts, test_meta)
            agg_test = get_basic_ts_passband(df_ts)

            # tmp save
            logger.debug('%15d save in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            if i_c == 0:
                agg_test_tmp = agg_test.copy()
            else:
                agg_test_tmp = pd.concat([agg_test_tmp, agg_test], axis=0)

            del agg_test
            gc.collect()
            logger.debug('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        agg_test = pd.merge(test_meta[["object_id"]], agg_test_tmp, on="object_id", how="outer")
        agg_test.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_test.drop('object_id', axis=1, inplace=True)
        self.test_feature = agg_test
        """
        self.test_feature = agg_train

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    # load data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make Time_mjddiff features
    f = Time_mjddiff('../../data/feature/')
    f.run(train_meta, test_meta).save()

    # make Time_mjddiff_passband features
    f = Time_mjddiff_passband('../../data/feature/')
    f.run(train_meta, test_meta).save()

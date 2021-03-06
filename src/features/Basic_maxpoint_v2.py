import feather
import pandas as pd
import numpy as np
import sys
import time
import logging
import gc
from pathlib import Path
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

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


def get_aggregations():
    agg_dict = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew']
    }
    return agg_dict


def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def add_features_to_agg(df):
    df['flux_diff'] = df['flux_max'] - df['flux_min']
    df['flux_dif2'] = (df['flux_max'] - df['flux_min']) / df['flux_mean']
    df['flux_w_mean'] = df['flux_by_flux_ratio_sq_sum'] / df['flux_ratio_sq_sum']
    df['flux_dif3'] = (df['flux_max'] - df['flux_min']) / df['flux_w_mean']

    # add detected == 1 feature
    df['flux_diff_detected1'] = df['flux_max_detected1'] - df['flux_min_detected1']
    df['flux_dif2_detected1'] = (df['flux_max_detected1'] - df['flux_min_detected1']) / df['flux_mean_detected1']
    df['flux_w_mean_detected1'] = df['flux_by_flux_ratio_sq_sum_detected1'] / df['flux_ratio_sq_sum_detected1']
    df['flux_dif3_detected1'] = (df['flux_max_detected1'] - df['flux_min_detected1']) / df['flux_w_mean_detected1']
    return df


def get_basic_ts(ts_data):
    # add columns
    ts_data['flux_ratio_sq'] = np.power(ts_data['flux'] / ts_data['flux_err'], 2.0)
    ts_data['flux_by_flux_ratio_sq'] = ts_data['flux'] * ts_data['flux_ratio_sq']

    # aggregate
    agg_dict = get_aggregations()
    agg_results = ts_data.groupby('object_id').agg(agg_dict)
    new_columns = get_new_columns(agg_dict)
    agg_results.columns = new_columns

    # add detected == 1 feature
    detected1_results = ts_data.query('detected == 1').groupby('object_id').agg(agg_dict)
    new_columns = get_new_columns(agg_dict)
    new_columns = [f'{col}_detected1' for col in new_columns]
    detected1_results.columns = new_columns
    agg_results = pd.concat([agg_results, detected1_results], axis=1).sort_index()

    agg_results = add_features_to_agg(df=agg_results)
    agg_results.reset_index(drop=False, inplace=True)   # object_idは残したまま。

    return agg_results


def get_divided_point(obj_data, thres_dist):
    maxpoint = obj_data.loc[obj_data["flux"].idxmax()]["mjd"]
    points = np.concatenate([np.array([maxpoint - thres_dist]), np.array([maxpoint + thres_dist])])
    return points


def get_maxpoint(obj_data, thres_dict):
    points = get_divided_point(obj_data, thres_dict)
    return obj_data.query('mjd >= @points[0] & mjd < @points[1]')


def add_maxpoint(ts_data, meta_data, thres_dict=100):
    object_id_list = ts_data.object_id.unique().tolist()
    results = Parallel(n_jobs=-1)(
        [delayed(get_maxpoint)(ts_data.query('object_id == @object_id'), thres_dict) for object_id in object_id_list])
    ts_data = pd.concat(results, axis=0, ignore_index=True)
    return ts_data


class Basic_maxpoint(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        self.suffix = "maxpoint"
        logger = get_module_logger(__name__)

        # make train features
        start = time.time()
        train_ts = pd.read_csv('../../data/interim/training_set_maxpoint_thres100days.csv')
        # train_ts = add_maxpoint(train_ts, train_meta)
        agg_train = get_basic_ts(train_ts)
        agg_train = pd.merge(train_meta[["object_id"]], agg_train, on="object_id", how="outer")
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train
        logger.debug('train done in %5.1f' % ((time.time() - start) / 60))

        # make features by test-data
        start = time.time()
        chunks = 5000000
        chunk_last = pd.DataFrame()
        test_row_num = 107146165
        total_steps = int(np.ceil(test_row_num / chunks))

        reader = pd.read_csv('../../data/interim/test_set_maxpoint_thres100days.csv', chunksize=chunks, iterator=True)
        for i_c, df_ts in enumerate(reader):
            df_ts = pd.concat([chunk_last, df_ts], ignore_index=True)

            if i_c + 1 < total_steps:
                id_last = df_ts['object_id'].values[-1]
                mask_last = (df_ts['object_id'] == id_last).values
                chunk_last = df_ts[mask_last]
                df_ts = df_ts[~mask_last]

            # df_ts = add_maxpoint(df_ts, test_meta)
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

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


def get_basic_ts_passband(ts_data):
    agg_dict = {
        'flux': ['min', 'max', 'mean'],
        'flux_ratio_sq': ['sum'],
        'flux_by_flux_ratio_sq': ['sum']
    }

    # add columns
    ts_data['flux_ratio_sq'] = np.power(ts_data['flux'] / ts_data['flux_err'], 2.0)
    ts_data['flux_by_flux_ratio_sq'] = ts_data['flux'] * ts_data['flux_ratio_sq']

    # aggregate
    agg_results = ts_data.groupby(['object_id', 'passband']).agg(agg_dict).unstack()
    column_name = [f'{col[0]}_{col[1]}_{col[2]}' for col in agg_results.columns]
    agg_results.columns = column_name

    passband_list = [0, 1, 2, 3, 4, 5]
    for passband in passband_list:
        agg_results[f'flux_diff_{passband}'] = agg_results[f'flux_max_{passband}'] - agg_results[f'flux_min_{passband}']
        agg_results[f'flux_dif2_{passband}'] = (agg_results[f'flux_diff_{passband}']) / agg_results[f'flux_mean_{passband}']
        agg_results[f'flux_w_mean_{passband}'] = agg_results[f'flux_by_flux_ratio_sq_sum_{passband}'] / agg_results[f'flux_ratio_sq_sum_{passband}']
        agg_results[f'flux_dif3_{passband}'] = (agg_results[f'flux_diff_{passband}']) / agg_results[f'flux_w_mean_{passband}']

    # diff other passband
    base_passband_list = [5]
    target_passband_list = [0, 1, 2, 3, 4]
    for base in base_passband_list:
        for target in target_passband_list:
            agg_results[f'flux_max_diff_{base}_{target}'] = agg_results[f'flux_max_{base}'] - agg_results[f'flux_max_{target}']
            agg_results[f'flux_min_diff_{base}_{target}'] = agg_results[f'flux_min_{base}'] - agg_results[f'flux_min_{target}']
            agg_results[f'flux_mean_diff_{base}_{target}'] = agg_results[f'flux_mean_{base}'] - agg_results[f'flux_mean_{target}']
            agg_results[f'flux_diff_diff_{base}_{target}'] = agg_results[f'flux_diff_{base}'] - agg_results[f'flux_diff_{target}']
            agg_results[f'flux_dif2_diff_{base}_{target}'] = agg_results[f'flux_dif2_{base}'] - agg_results[f'flux_dif2_{target}']
            agg_results[f'flux_w_mean_diff_{base}_{target}'] = agg_results[f'flux_w_mean_{base}'] - agg_results[f'flux_w_mean_{target}']
            agg_results[f'flux_dif3_diff_{base}_{target}'] = agg_results[f'flux_dif3_{base}'] - agg_results[f'flux_dif3_{target}']

    agg_results.reset_index(drop=False, inplace=True)
    return agg_results


class Basic_maxpoint_passband(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        self.suffix = "maxpoint"
        logger = get_module_logger(__name__)

        # make train features
        start = time.time()
        train_ts = pd.read_csv('../../data/interim/training_set_maxpoint_thres100days.csv')
        # train_ts = add_maxpoint(train_ts, train_meta)

        train_ts = train_ts.query("detected == 1")
        train_ts = pd.merge(train_ts, train_meta, on="object_id", how="inner")
        flux = train_ts["flux"]
        train_ts['flux'] = np.where(flux < 1, 2.5*np.log10(-flux+2), -2.5*np.log10(flux)) - train_ts['distmod']

        agg_train = get_basic_ts_passband(train_ts)
        agg_train = pd.merge(train_meta[["object_id"]], agg_train, on="object_id", how="outer")
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train
        logger.debug('train done in %5.1f' % ((time.time() - start) / 60))

        # make features by test-data
        start = time.time()
        chunks = 5000000
        chunk_last = pd.DataFrame()
        test_row_num = 107146165
        total_steps = int(np.ceil(test_row_num / chunks))

        reader = pd.read_csv('../../data/interim/test_set_maxpoint_thres100days.csv', chunksize=chunks, iterator=True)
        for i_c, df_ts in enumerate(reader):
            df_ts = pd.concat([chunk_last, df_ts], ignore_index=True)

            if i_c + 1 < total_steps:
                id_last = df_ts['object_id'].values[-1]
                mask_last = (df_ts['object_id'] == id_last).values
                chunk_last = df_ts[mask_last]
                df_ts = df_ts[~mask_last]

            # df_ts = add_maxpoint(df_ts, test_meta)
            df_ts = df_ts.query("detected == 1")
            df_ts = pd.merge(df_ts, test_meta, on="object_id", how="inner")
            flux = df_ts["flux"]
            df_ts['flux'] = np.where(flux < 1, 2.5*np.log10(-flux+2), -2.5*np.log10(flux)) - df_ts['distmod']
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

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


"""thres200days
def make_dataframe(train_meta, test_meta):
    logger = get_module_logger(__name__)

    # make train features
    start = time.time()
    train_ts = pd.read_csv('../../data/input/training_set.csv')
    train_ts = add_maxpoint(train_ts, train_meta, thres_dict=200)
    train_ts.to_csv("../../data/interim/training_set_maxpoint_thres200days.csv", header=True, index=False)
    logger.debug('train done in %5.1f' % ((time.time() - start) / 60))

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

        # tmp
        if chunks * (i_c + 1) <= 400000000:
            logger.debug('%15d skip in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            continue

        df_ts = add_maxpoint(df_ts, test_meta, thres_dict=200)

        # tmp save
        logger.debug('%15d save in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
        if i_c == 0:
            df_ts.to_csv('../../data/interim/test_set_maxpoint_thres200days.csv', header=True, index=False)
        else:
            df_ts.to_csv('../../data/interim/test_set_maxpoint_thres200days.csv', header=False, index=False, mode='a')

        del df_ts
        gc.collect()
        logger.debug('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
"""

"""thres100days
def make_dataframe(train_meta, test_meta):
    logger = get_module_logger(__name__)

    # make train features
    start = time.time()
    train_ts = pd.read_csv('../../data/input/training_set.csv')
    train_ts = add_maxpoint(train_ts, train_meta, thres_dict=100)
    train_ts.to_csv("../../data/interim/training_set_maxpoint_thres100days.csv", header=True, index=False)
    logger.debug('train done in %5.1f' % ((time.time() - start) / 60))

    # make features by test-data
    start = time.time()
    chunks = 5000000
    chunk_last = pd.DataFrame()
    test_row_num = 154085601
    total_steps = int(np.ceil(test_row_num / chunks))

    reader = pd.read_csv('../../data/interim/test_set_maxpoint_thres200days.csv', chunksize=chunks, iterator=True)
    for i_c, df_ts in enumerate(reader):
        df_ts = pd.concat([chunk_last, df_ts], ignore_index=True)

        if i_c + 1 < total_steps:
            id_last = df_ts['object_id'].values[-1]
            mask_last = (df_ts['object_id'] == id_last).values
            chunk_last = df_ts[mask_last]
            df_ts = df_ts[~mask_last]

        df_ts = add_maxpoint(df_ts, test_meta, thres_dict=100)

        # tmp save
        logger.debug('%15d save in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
        if i_c == 0:
            df_ts.to_csv('../../data/interim/test_set_maxpoint_thres100days.csv', header=True, index=False)
        else:
            df_ts.to_csv('../../data/interim/test_set_maxpoint_thres100days.csv', header=False, index=False, mode='a')

        del df_ts
        gc.collect()
        logger.debug('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
"""


if __name__ == '__main__':
    # load data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make dataframe
    # make_dataframe(train_meta, test_meta)

    # make Basic_maxpoint_passband features
    f = Basic_maxpoint_passband('../../data/feature/')
    f.run(train_meta, test_meta).save()

    # make Basic_fluxfactor features
    #f = Basic_maxpoint('../../data/feature/')
    #f.run(train_meta, test_meta).save()

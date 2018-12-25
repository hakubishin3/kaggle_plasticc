import feather
import pandas as pd
import numpy as np
import time
import sys
import itertools
import logging
import gc
import dask.dataframe as dd
from pathlib import Path
from astropy.time import Time
from dask.multiprocessing import get
from dask.array import stats

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


def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def add_features_to_agg(df):
    passband_list = [0, 1, 2, 3, 4, 5]
    comb = list(itertools.combinations(passband_list, 2))

    for base_pb, target_pb in comb:
        df[f'flux_diff_{target_pb}-{base_pb}'] = \
            df[f'flux_{target_pb}-{base_pb}_max'] - df[f'flux_{target_pb}-{base_pb}_min']
        df[f'flux_dif2_{target_pb}-{base_pb}'] = \
            (df[f'flux_{target_pb}-{base_pb}_max'] - df[f'flux_{target_pb}-{base_pb}_min']) / df[f'flux_{target_pb}-{base_pb}_mean']
        df[f'flux_w_mean_{target_pb}-{base_pb}'] = \
            df[f'flux_by_flux_ratio_sq_{target_pb}-{base_pb}_sum'] / df[f'flux_ratio_sq_{target_pb}-{base_pb}_sum']
        df[f'flux_dif3_{target_pb}-{base_pb}'] = \
            (df[f'flux_{target_pb}-{base_pb}_max'] - df[f'flux_{target_pb}-{base_pb}_min']) / df[f'flux_w_mean_{target_pb}-{base_pb}']
    return df


def get_aggregations():
    agg_list = ['min', 'max', 'mean', 'median', 'std', 'skew', 'sum']
    return agg_list


def make_pivot(ts_data, freq):
    passband_list = [0, 1, 2, 3, 4, 5]
    comb = list(itertools.combinations(passband_list, 2))

    ts_data['mjd_dt'] = pd.to_datetime(Time(ts_data.mjd.values, format='mjd').iso)
    ts_data = ts_data.sort_values(['object_id', 'mjd'])

    pivot_tbl = pd.pivot_table(
        ts_data, index=['object_id', pd.Grouper(key='mjd_dt', freq=freq)],
        columns='passband', values=['flux', 'flux_err'], aggfunc=[np.mean]
    )
    pivot_tbl.reset_index(drop=False, inplace=True)

    col_list = []
    for base_pb, target_pb in comb:
        pivot_tbl[f"flux_{target_pb}-{base_pb}"] = pivot_tbl["mean"]["flux"][target_pb] - pivot_tbl["mean"]["flux"][base_pb]
        pivot_tbl[f"flux_err_{target_pb}-{base_pb}"] = np.sqrt(pivot_tbl["mean"]["flux_err"][target_pb]**2 + pivot_tbl["mean"]["flux_err"][base_pb]**2)
        col_list.extend([f"flux_{target_pb}-{base_pb}", f"flux_err_{target_pb}-{base_pb}"])

    col_list.append('object_id')
    pivot_tbl = pivot_tbl[col_list]
    pivot_tbl.columns = col_list

    # add features
    for base_pb, target_pb in comb:
        pivot_tbl[f'flux_ratio_sq_{target_pb}-{base_pb}'] = \
            np.power(pivot_tbl[f'flux_{target_pb}-{base_pb}'] / pivot_tbl[f"flux_err_{target_pb}-{base_pb}"], 2.0)
        pivot_tbl[f'flux_by_flux_ratio_sq_{target_pb}-{base_pb}'] = \
            pivot_tbl[f'flux_{target_pb}-{base_pb}'] * pivot_tbl[f'flux_ratio_sq_{target_pb}-{base_pb}']

    return pivot_tbl


class DiffFluxFeatures_TimeGrouper_14d(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        logger = get_module_logger(__name__)
        freq = '14d'
        self.prefix = freq

        # make train features
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        pivot_train = make_pivot(train_ts, freq)
        pivot_train.to_csv("../../data/interim/pivot_train_DiffFluxFeatures_TimeGrouper_14d.csv", header=True, index=False)
        agg_list = get_aggregations()
        agg_train = pivot_train.groupby('object_id').agg(agg_list)
        cols = ['_'.join([s for s in col if len(s) != 0]) if len(col) != 1 else col for col in agg_train.columns]
        agg_train.columns = cols
        agg_train = add_features_to_agg(agg_train)

        agg_train.reset_index(drop=False, inplace=True)
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train

        # make test features
        start = time.time()
        chunks = 10000000
        chunk_last = pd.DataFrame()
        test_row_num = 453653104
        total_steps = int(np.ceil(test_row_num / chunks))

        reader = pd.read_csv('../../data/input/test_set.csv', chunksize=chunks, iterator=True)
        for i_c, df_ts in enumerate(reader):
            logger.debug('%15d start in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            df_ts = pd.concat([chunk_last, df_ts], ignore_index=True)

            if i_c + 1 < total_steps:
                id_last = df_ts['object_id'].values[-1]
                mask_last = (df_ts['object_id'] == id_last).values
                chunk_last = df_ts[mask_last]
                df_ts = df_ts[~mask_last]

            pivot_test = make_pivot(df_ts, freq)
            agg_list = get_aggregations()
            agg_test = pivot_test.groupby('object_id').agg(agg_list)
            cols = ['_'.join([s for s in col if len(s) != 0]) if len(col) != 1 else col for col in agg_test.columns]
            agg_test.columns = cols
            agg_test = add_features_to_agg(agg_test)
            agg_test.reset_index(drop=False, inplace=True)

            # tmp save
            logger.debug('%15d save in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            if i_c == 0:
                agg_test.to_csv('../../data/feature/tmp/DiffFluxFeatures_TimeGrouper_14d_test.csv', header=True, index=False)
                pivot_test.to_csv("../../data/interim/pivot_test_DiffFluxFeatures_TimeGrouper_14d.csv", header=True, index=False)
            else:
                agg_test.to_csv('../../data/feature/tmp/DiffFluxFeatures_TimeGrouper_14d_test.csv', header=False, index=False, mode='a')
                pivot_test.to_csv("../../data/interim/pivot_test_DiffFluxFeatures_TimeGrouper_14d.csv", header=False, index=False, mode='a')

            del agg_test
            del pivot_test
            gc.collect()
            logger.debug('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        agg_test = pd.read_csv('../../data/feature/tmp/DiffFluxFeatures_TimeGrouper_14d_test.csv')
        agg_test.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_test.drop('object_id', axis=1, inplace=True)
        self.test_feature = agg_test

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    # load data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make DiffFluxFeatures_TimeGrouper_14d features
    f = DiffFluxFeatures_TimeGrouper_14d('../../data/feature/')
    f.run(train_meta, test_meta).save()

import feather
import pandas as pd
import numpy as np
import sys
import time
import gc
from pathlib import Path
from sklearn.cluster import KMeans

if Path.cwd().name == 'kaggle_plasticc':
    from .base import Feature
elif Path.cwd().name == 'features':
    from base import Feature


def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) from
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(
        np.power(np.sin(np.divide(dlat, 2)), 2),
        np.multiply(np.cos(lat1), np.multiply(np.cos(lat2), np.power(np.sin(np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    latlon1 = np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2))

    return haversine, latlon1


class Basic_meta(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """

        """
        # Coordinate
        self.train_feature['ra'] = train_meta['ra']
        self.test_feature['ra'] = test_meta['ra']
        self.train_feature['decl'] = train_meta['decl']
        self.test_feature['decl'] = test_meta['decl']
        self.train_feature['gal_l'] = train_meta['gal_l']
        self.test_feature['gal_l'] = test_meta['gal_l']
        self.train_feature['gal_b'] = train_meta['gal_b']
        self.test_feature['gal_b'] = test_meta['gal_b']

        # ddf
        self.train_feature['ddf'] = train_meta['ddf']
        self.test_feature['ddf'] = test_meta['ddf']
        """

        # hostgal_specz
        # trainとtestの偏りがひどく大きいため、いったん使用を控える。
        # self.train_feature['hostgal_specz'] = train_meta['hostgal_specz']
        # self.test_feature['hostgal_specz'] = test_meta['hostgal_specz']

        # hostgal_photoz
        self.train_feature['hostgal_photoz'] = train_meta['hostgal_photoz']
        self.test_feature['hostgal_photoz'] = test_meta['hostgal_photoz']
        self.train_feature['hostgal_photoz_err'] = train_meta['hostgal_photoz_err']
        self.test_feature['hostgal_photoz_err'] = test_meta['hostgal_photoz_err']

        # distmod
        # NA means that it is a galactic source.
        self.train_feature['distmod'] = train_meta['distmod'].fillna(0)
        self.test_feature['distmod'] = test_meta['distmod'].fillna(0)

        # MWEBV
        self.train_feature['mwebv'] = train_meta['mwebv']
        self.test_feature['mwebv'] = test_meta['mwebv']

        # distance
        haversine_train, latlon1_train = haversine_plus(
            train_meta['ra'].values, train_meta['decl'].values,
            train_meta['gal_l'].values, train_meta['gal_b'].values)
        self.train_feature['haversine'] = haversine_train
        self.train_feature['latlon1'] = latlon1_train

        haversine_test, latlon1_test = haversine_plus(
            test_meta['ra'].values, test_meta['decl'].values,
            test_meta['gal_l'].values, test_meta['gal_b'].values)
        self.test_feature['haversine'] = haversine_test
        self.test_feature['latlon1'] = latlon1_test

        # hostgal_photoz_certain
        self.train_feature['hostgal_photoz_certain'] = np.multiply(
            train_meta['hostgal_photoz'].values,
            np.exp(train_meta['hostgal_photoz_err'].values))

        self.test_feature['hostgal_photoz_certain'] = np.multiply(
            test_meta['hostgal_photoz'].values,
            np.exp(test_meta['hostgal_photoz_err'].values))

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


def get_aggregations():
    agg_dict = {
        # 'mjd': ['min', 'max', 'size'],
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
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
    # add mjd feature
    ts_detected_grpby = ts_data.query('detected == 1').groupby('object_id')
    mjd_feature = ts_detected_grpby.max()["mjd"] - ts_detected_grpby.min()["mjd"]
    mjd_feature.name = "diff_mjd_maxmin_detected1"
    agg_results = pd.concat([agg_results, mjd_feature], axis=1).sort_index()
    # add detected == 1 feature
    detected1_results = ts_data.query('detected == 1').groupby('object_id').agg(agg_dict)
    new_columns = get_new_columns(agg_dict)
    new_columns = [f'{col}_detected1' for col in new_columns]
    detected1_results.columns = new_columns
    agg_results = pd.concat([agg_results, detected1_results], axis=1).sort_index()
    agg_results.drop("detected_mean_detected1", axis=1, inplace=True)

    agg_results = add_features_to_agg(df=agg_results)
    agg_results.reset_index(drop=False, inplace=True)   # object_idは残したまま。

    return agg_results


class Basic_ts(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        # make train features
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        agg_train = get_basic_ts(train_ts)
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train

        # make features by test-data
        start = time.time()
        chunks = 50000000
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

            agg_test = get_basic_ts(df_ts)

            # tmp save
            if i_c == 0:
                agg_test.to_csv(
                    '../../data/feature/tmp/agg_test.csv', header=True, index=False)
            else:
                agg_test.to_csv(
                    '../../data/feature/tmp/agg_test.csv', header=False, index=False, mode='a')

            del agg_test
            gc.collect()
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        agg_test = pd.read_csv('../../data/feature/tmp/agg_test.csv')
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

    # flux_diff is same as amplitude of cesium feature
    agg_results.drop(['flux_min_0', 'flux_min_1', 'flux_min_2', 'flux_min_3', 'flux_min_4', 'flux_min_5'], axis=1, inplace=True)
    agg_results.drop(['flux_max_0', 'flux_max_1', 'flux_max_2', 'flux_max_3', 'flux_max_4', 'flux_max_5'], axis=1, inplace=True)
    agg_results.drop(['flux_mean_0', 'flux_mean_1', 'flux_mean_2', 'flux_mean_3', 'flux_mean_4', 'flux_mean_5'], axis=1, inplace=True)
    agg_results.drop(['flux_diff_0', 'flux_diff_1', 'flux_diff_2', 'flux_diff_3', 'flux_diff_4', 'flux_diff_5'], axis=1, inplace=True)
    agg_results.reset_index(drop=False, inplace=True)
    return agg_results


class Basic_ts_passband(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        logger = get_module_logger(__name__)

        # make train features
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        agg_train = get_basic_ts_passband(train_ts)
        agg_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = agg_train

        # make features by test-data
        start = time.time()
        chunks = 50000000
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
        agg_test_tmp.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        agg_test_tmp.drop('object_id', axis=1, inplace=True)
        self.test_feature = agg_test_tmp

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    # load data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make Basic_meta features
    # f = Basic_meta('../../data/feature/')
    # f.run(train_meta, test_meta).save()

    # make Basic_ts_passband features
    f = Basic_ts_passband('../../data/feature/')
    f.run(train_meta, test_meta).save()

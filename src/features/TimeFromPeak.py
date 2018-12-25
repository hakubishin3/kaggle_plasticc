import gc
import feather
import pandas as pd
import numpy as np
import time
import sys
import itertools
import logging
from pathlib import Path
from tqdm import tnrange, tqdm_notebook, tqdm
from collections import OrderedDict
from cesium.time_series import TimeSeries
import cesium.featurize as featurize
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


def _find_peak(extract_obj, passband):
    thres_per_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    extract_passband = extract_obj.query('passband == @passband & detected == 1')

    if extract_passband.shape[0] == 0:
        result_list = [np.nan for i in range(int(len(thres_per_list) * 2))]

    else:
        import warnings
        warnings.filterwarnings("ignore")
        extract_passband["flux_add_err"] = extract_passband["flux"] + extract_passband["flux_err"]
        peak_flux = extract_passband["flux"].max()
        peak_point = extract_passband[extract_passband.flux == peak_flux]["mjd"].values[0]

        data_before_peak = extract_passband.query('mjd < @peak_point')
        data_after_peak = extract_passband.query('mjd > @peak_point')

        min_flux_before = extract_passband["flux"].min()
        min_flux_after = extract_passband["flux"].min()
        flux_diff_before = peak_flux - min_flux_before
        flux_diff_after = peak_flux - min_flux_after

        result_list = []

        result_after = []
        for thres_per in thres_per_list:
            thres = min_flux_before + flux_diff_before * thres_per
            point = data_after_peak.query("flux < @thres")["mjd"].min()
            diff_point = point - peak_point
            result_after.append(diff_point)
        result_list.extend(result_after)

        result_before = []
        for thres_per in thres_per_list:
            thres = min_flux_after + flux_diff_after * thres_per
            point = data_before_peak.query("flux < @thres")["mjd"].max()
            diff_point = peak_point - point
            result_before.append(diff_point)
        result_list.extend(result_before)

        """
        result_diff = []
        for af, bef in zip(result_after, result_before):
            result_diff.append(af + bef)
        result_list.extend(result_diff)
        """

    return result_list


def find_peak(extract_obj):
    object_id = extract_obj["object_id"].unique()[0]
    passband_list = [0, 1, 2, 3, 4, 5]

    result_list = [object_id]
    for passband in passband_list:
        result = _find_peak(extract_obj, passband)
        result_list.extend(result)

    return result_list


def get_features(ts_data):
    object_id_list = ts_data["object_id"].unique()
    results = Parallel(n_jobs=-1)(
        [delayed(find_peak)(ts_data.query('object_id == @object_id')) for object_id in object_id_list])

    thres_per_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    passband_list = [0, 1, 2, 3, 4, 5]
    colname_list = ['object_id']

    for passband in passband_list:
        # for which in ['after', 'before', 'diff']:
        for which in ['after', 'before']:
            for thres in thres_per_list:
                colname_list.append(f'time_from_peak_{which}_thres{thres}_pass{passband}')

    result_df = pd.DataFrame(results, columns=colname_list)

    return result_df


class TimeFromPeak(Feature):
    def create_features(self, train_meta, test_meta):
        # load ts data
        logger = get_module_logger(__name__)
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        print('train_ts:', train_ts.shape)

        # make feature
        start = time.time()
        features_train = get_features(train_ts)
        features_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        features_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = features_train
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

            features_test = get_features(df_ts)

            logger.debug('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            # tmp save
            if i_c == 0:
                features_test.to_csv(
                    '../../data/feature/tmp/TimeFromPeak_test.csv', header=True, index=False)
            else:
                features_test.to_csv(
                    '../../data/feature/tmp/TimeFromPeak_test.csv', header=False, index=False, mode='a')

            del features_test
            gc.collect()
            logger.debug('%15d save in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        features_test = pd.read_csv('../../data/feature/tmp/TimeFromPeak_test.csv')
        features_test.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        features_test.drop('object_id', axis=1, inplace=True)
        self.test_feature = features_test

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


def get_diff_peak(ts_data):
    # fluxの一番大きい時点をpassbandごとに抽出
    ts_data = ts_data.query("detected == 1")
    agg_result = ts_data.loc[ts_data.groupby(["object_id", "passband"]).idxmax()["flux"]]
    agg_result = agg_result.set_index(["object_id", "passband"])[["mjd"]].unstack()
    colname = ["pass0", "pass1", "pass2", "pass3", "pass4", "pass5"]
    agg_result.columns = colname

    comb = list(itertools.combinations(colname, 2))
    for pass_a, pass_b in comb:
        agg_result[f'peakpoint_{pass_a}-{pass_b}'] = agg_result[pass_a] - agg_result[pass_b]

    agg_result.drop(colname, axis=1, inplace=True)
    agg_result.reset_index(drop=False, inplace=True)
    return agg_result


class DiffPeak(Feature):
    def create_features(self, train_meta, test_meta):
        # load ts data
        logger = get_module_logger(__name__)
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        print('train_ts:', train_ts.shape)

        # make feature
        features_train = get_diff_peak(train_ts)
        features_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        features_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = features_train

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

            agg_test = get_diff_peak(df_ts)

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
    # load meta data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make features
    f = TimeFromPeak('../../data/feature/')
    f.run(train_meta, test_meta).save()

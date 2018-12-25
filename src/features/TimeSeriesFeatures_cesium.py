import gc
import feather
import pandas as pd
import numpy as np
import time
import sys
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


def get_ts_features(ts_data, meta_data, features_to_use, n_jobs=-1):
    # settings
    ts_dict = OrderedDict()   # save TimeSeries object.
    object_list = ts_data['object_id'].unique().tolist()
    pbmap = OrderedDict(
        [(0, 'u'), (1, 'g'), (2, 'r'), (3, 'i'), (4, 'z'), (5, 'y')]
    )

    # get TimeSeries objects
    results = Parallel(n_jobs=n_jobs)(
        [delayed(get_ts_obj)(object_id, meta_data, ts_data, pbmap) for object_id in object_list])
    for object_id, ts_obj in results:
        ts_dict[object_id] = ts_obj

    # get features
    results = Parallel(n_jobs=n_jobs)(
        [delayed(worker)(ts_obj, features_to_use) for ts_obj in list(ts_dict.values())])
    featuretable = featurize.assemble_featureset(features_list=results, time_series=ts_dict.values())

    old_names = featuretable.columns.values
    new_names = ['{}_{}'.format(x, pbmap.get(y)) for x, y in old_names]
    featuretable.columns = new_names

    featuretable.index.name = 'object_id'
    featuretable.reset_index(drop=False, inplace=True)   # object_idは最後まで残しておく。

    return featuretable


def get_ts_obj(object_id, meta_data, ts_data, pbmap):
    pbnames = list(pbmap.values())
    row = meta_data.query('object_id == @object_id')
    if 'target' in meta_data.columns:
        target = row['target']
    else:
        target = None

    extract_ts = ts_data.query('object_id == @object_id')
    pbind = [(extract_ts['passband'] == pb) for pb in pbmap]
    t = [extract_ts['mjd'][mask].values for mask in pbind]
    m = [extract_ts['flux'][mask].values for mask in pbind]
    e = [extract_ts['flux_err'][mask].values for mask in pbind]

    ts_obj = TimeSeries(
        t=t, m=m, e=e, label=target, name=object_id,
        channel_names=pbnames
    )

    return (object_id, ts_obj)


def worker(ts_obj, features_to_use):
    this_feats = featurize.featurize_single_ts(
        ts_obj, features_to_use=features_to_use,
        raise_exceptions=False
    )

    return this_feats


def get_general_features():
    features_list = [
        "amplitude",
        "flux_percentile_ratio_mid20",
        "flux_percentile_ratio_mid35",
        "flux_percentile_ratio_mid50",
        "flux_percentile_ratio_mid65",
        "flux_percentile_ratio_mid80",
        "max_slope",
        "maximum",
        "median",
        "median_absolute_deviation",
        "minimum",
        "percent_amplitude",
        "percent_beyond_1_std",
        "percent_close_to_median",
        "percent_difference_flux_percentile",
        "period_fast",
        "qso_log_chi2_qsonu",
        "qso_log_chi2nuNULL_chi2nu",
        "skew",
        "std",
        "stetson_j",
        "stetson_k",
        "weighted_average",
    ]
    return features_list


def get_freq_features():
    features_list = [
        "fold2P_slope_10percentile",
        "fold2P_slope_90percentile",
        "freq1_amplitude1",
        "freq1_amplitude2",
        "freq1_amplitude3",
        "freq1_amplitude4",
        "freq1_freq",
        "freq1_lambda",
        "freq1_rel_phase2",
        "freq1_rel_phase3",
        "freq1_rel_phase4",
        "freq1_signif",
        "freq2_amplitude1",
        "freq2_amplitude2",
        "freq2_amplitude3",
        "freq2_amplitude4",
        "freq2_freq",
        "freq2_rel_phase2",
        "freq2_rel_phase3",
        "freq2_rel_phase4",
        "freq3_amplitude1",
        "freq3_amplitude2",
        "freq3_amplitude3",
        "freq3_amplitude4",
        "freq3_freq",
        "freq3_rel_phase2",
        "freq3_rel_phase3",
        "freq3_rel_phase4",
        "freq_amplitude_ratio_21",
        "freq_amplitude_ratio_31",
        "freq_frequency_ratio_21",
        "freq_frequency_ratio_31",
        "freq_model_max_delta_mags",
        "freq_model_min_delta_mags",
        "freq_model_phi1_phi2",
        "freq_n_alias",
        "freq_signif_ratio_21",
        "freq_signif_ratio_31",
        "freq_varrat",
        "freq_y_offset",
        "linear_trend",
        "medperc90_2p_p",
        "p2p_scatter_2praw",
        "p2p_scatter_over_mad",
        "p2p_scatter_pfold_over_mad",
        "p2p_ssqr_diff_over_var",
        "scatter_res_raw",
    ]
    return features_list


def get_cad_features():
    features_list = [
        "all_times_nhist_numpeaks",
        "all_times_nhist_peak1_bin",
        "all_times_nhist_peak2_bin",
        "all_times_nhist_peak3_bin",
        "all_times_nhist_peak4_bin",
        "all_times_nhist_peak_1_to_2",
        "all_times_nhist_peak_1_to_3",
        "all_times_nhist_peak_1_to_4",
        "all_times_nhist_peak_2_to_3",
        "all_times_nhist_peak_2_to_4",
        "all_times_nhist_peak_3_to_4",
        "all_times_nhist_peak_val",
        "avg_double_to_single_step",
        "avg_err",
        "avgt",
        "cad_probs_1",
        "cad_probs_10",
        "cad_probs_20",
        "cad_probs_30",
        "cad_probs_40",
        "cad_probs_50",
        "cad_probs_100",
        "cad_probs_500",
        "cad_probs_1000",
        "cad_probs_5000",
        "cad_probs_10000",
        "cad_probs_50000",
        "cad_probs_100000",
        "cad_probs_500000",
        "cad_probs_1000000",
        "cad_probs_5000000",
        "cad_probs_10000000",
        "cads_avg",
        "cads_med",
        "cads_std",
        "mean",
        "med_double_to_single_step",
        "med_err",
        "n_epochs",
        "std_double_to_single_step",
        "std_err",
        "total_time",
    ]
    return features_list


class TimeSeriesFeatures_cesium_common(Feature):
    def create_features(self, train_meta, test_meta):
        # load ts data
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        print('train_ts:', train_ts.shape)

        # make feature
        start = time.time()
        features_to_use = get_general_features()
        ts_features_train = get_ts_features(train_ts, train_meta, features_to_use, n_jobs=-1)
        ts_features_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        ts_features_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = ts_features_train
        print('train done in %5.1f' % ((time.time() - start) / 60))

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

            ts_features_test = get_ts_features(df_ts, test_meta, features_to_use, n_jobs=-1)

            # tmp save
            if i_c == 0:
                ts_features_test.to_csv(
                    '../../data/feature/tmp/ts_features_test.csv', header=True, index=False)
            else:
                ts_features_test.to_csv(
                    '../../data/feature/tmp/ts_features_test.csv', header=False, index=False, mode='a')

            del ts_features_test
            gc.collect()
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        ts_features_test = pd.read_csv('../../data/feature/tmp/ts_features_test.csv')
        ts_features_test.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        ts_features_test.drop('object_id', axis=1, inplace=True)
        self.test_feature = ts_features_test

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


class TimeSeriesFeatures_cesium_cad(Feature):
    def create_features(self, train_meta, test_meta):
        # load ts data
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        print('train_ts:', train_ts.shape)

        # make feature
        start = time.time()
        features_to_use = get_cad_features()
        ts_features_train = get_ts_features(train_ts, train_meta, features_to_use, n_jobs=-1)
        ts_features_train.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        ts_features_train.drop('object_id', axis=1, inplace=True)
        self.train_feature = ts_features_train
        print('train done in %5.1f' % ((time.time() - start) / 60))

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

            ts_features_test = get_ts_features(df_ts, test_meta, features_to_use, n_jobs=-1)

            # tmp save
            if i_c == 0:
                ts_features_test.to_csv(
                    '../../data/feature/tmp/TimeSeriesFeatures_cesium_cad.csv', header=True, index=False)
            else:
                ts_features_test.to_csv(
                    '../../data/feature/tmp/TimeSeriesFeatures_cesium_cad.csv', header=False, index=False, mode='a')

            del ts_features_test
            gc.collect()
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

        # finish
        ts_features_test = pd.read_csv('../../data/feature/tmp/TimeSeriesFeatures_cesium_cad.csv')
        ts_features_test.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        ts_features_test.drop('object_id', axis=1, inplace=True)
        self.test_feature = ts_features_test

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    # load meta data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make features
    f = TimeSeriesFeatures_cesium_common('../../data/feature/')
    f.run(train_meta, test_meta).save()

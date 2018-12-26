import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def get_timeseries_by_column_name(ts_data, passband, column_name):
    ts_data_passband = ts_data.query("passband == @passband")
    ts_column_timeseries = ts_data_passband.groupby('object_id')[column_name].apply(lambda df: df.reset_index(drop=True)).unstack()
    ts_column_timeseries = ts_column_timeseries.reset_index()

    # add colname (for sort name)
    if column_name == "mjd":
        ts_column_timeseries['feature_id'] = "1_mjd"
    elif column_name == "flux":
        ts_column_timeseries['feature_id'] = "2_flux"
    elif column_name == "flux_err":
        ts_column_timeseries['feature_id'] = "3_flux_err"

    return ts_column_timeseries


def get_timeseries(ts_data, passband):
    colnames = ['mjd', 'flux', 'flux_err']
    n_obj = ts_data.object_id.nunique()
    # df = pd.concat([get_timeseries_by_column_name(ts_data, passband, col) for col in colnames])
    results = Parallel(n_jobs=-1)(
        [delayed(get_timeseries_by_column_name)(ts_data, passband, col) for col in colnames])
    df = pd.concat(results)
    df = df.set_index(['object_id', 'feature_id']).sort_index().unstack()
    x_pad = df.values.reshape(n_obj, -1, len(colnames)).astype("float32")

    return x_pad


def preprocess(X_raw, m_max=np.inf):
    X = X_raw.copy()

    # Replace times lags
    X[:, :, 0] = times_to_lags(X[:, :, 0])

    # re-scaling
    # flux_means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
    # flux_scales = np.atleast_2d(np.nanstd(X[:, :, 1], axis=1)).T
    flux_means = np.nanmean(X[:, :, 1])
    flux_scales = np.nanstd(X[:, :, 1])
    X[:, :, 1] -= flux_means
    X[:, :, 1] /= flux_scales

    time_means = np.atleast_2d(np.nanmean(X[:, :, 0], axis=1)).T
    time_scales = np.atleast_2d(np.nanstd(X[:, :, 0], axis=1)).T
    X[:, :, 0] -= time_means
    X[:, :, 0] /= time_scales

    return X, flux_means, flux_scales


def preprocess_trans(X_raw, flux_means, flux_scales, m_max=np.inf):
    X = X_raw.copy()

    # Replace times lags
    X[:, :, 0] = times_to_lags(X[:, :, 0])

    # re-scaling
    X[:, :, 1] -= flux_means
    X[:, :, 1] /= flux_scales

    time_means = np.atleast_2d(np.nanmean(X[:, :, 0], axis=1)).T
    time_scales = np.atleast_2d(np.nanstd(X[:, :, 0], axis=1)).T
    X[:, :, 0] -= time_means
    X[:, :, 0] /= time_scales

    return X


def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.zeros(T.shape[0]), np.diff(T, axis=1)]

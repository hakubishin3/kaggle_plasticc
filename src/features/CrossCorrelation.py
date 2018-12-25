import feather
import pandas as pd
import numpy as np
import sys
import itertools
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from astropy.time import Time
from joblib import Parallel, delayed

if Path.cwd().name == 'kaggle_plasticc':
    from .base import Feature
elif Path.cwd().name == 'features':
    from base import Feature
    import PYCCF as myccf


def convert_np_array(df):
    mjd = df['mjd'].values
    flux = df['flux'].values
    flux_err = df['flux_err'].values
    return mjd, flux, flux_err


def _calc_ccf(extract_obj, pass1, pass2, lag_range, interp):
    object_id = extract_obj.iloc[0]['object_id']
    extract_pass1 = extract_obj.query('passband == @pass1')
    extract_pass2 = extract_obj.query('passband == @pass2')
    mjd1, flux1, err1 = convert_np_array(extract_pass1)
    mjd2, flux2, err2 = convert_np_array(extract_pass2)

    nsim = 100
    mcmode = 2
    sigmode = 0.2
    perclim = 84.1344746

    tlag_peak, status_peak, tlag_centroid, status_centroid,\
        ccf_pack, max_rval, status_rval, pval = \
        myccf.peakcent(mjd1, flux1, mjd2, flux2, lag_range[0], lag_range[1], interp)
    tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak,\
        nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = myccf.xcor_mc(
            mjd1, flux1, abs(err1), mjd2, flux2, abs(err2),
            lag_range[0], lag_range[1], interp,
            nsim=nsim, mcmode=mcmode, sigmode=0.2
        )

    lag = ccf_pack[1]
    r = ccf_pack[0]

    centau = stats.scoreatpercentile(tlags_centroid, 50)
    centau_uperr = (stats.scoreatpercentile(tlags_centroid, perclim)) - centau
    centau_loerr = centau - (stats.scoreatpercentile(tlags_centroid, (100. - perclim)))

    peaktau = stats.scoreatpercentile(tlags_peak, 50)
    peaktau_uperr = (stats.scoreatpercentile(tlags_peak, perclim)) - centau
    peaktau_loerr = centau - (stats.scoreatpercentile(tlags_peak, (100. - perclim)))

    # add
    pass_comb = f'{pass1}-{pass2}'
    centau_err = centau_uperr + centau_loerr
    peaktau_err = peaktau_uperr + peaktau_loerr

    result_list = [object_id, centau, centau_uperr, centau_loerr, peaktau, peaktau_uperr, peaktau_loerr, nsuccess_peak / nsim]
    result_list.extend(r.tolist())

    return result_list


def calc_ccf(ts_data, pass1, pass2):
    ts_data = ts_data.sort_values(['object_id', 'mjd'])
    object_list = ts_data['object_id'].unique().tolist()

    # parameter
    lag_range = [-100, 100]
    interp = 10.

    results = Parallel(n_jobs=-1)(
        [delayed(_calc_ccf)(ts_data.query('object_id == @object_id'), pass1, pass2, lag_range, interp) for object_id in object_list])

    n_r = int((lag_range[1] - lag_range[0]) / interp) + 1
    column_name = ['object_id', 'centau', 'centau_uperr', 'centau_loerr', 'peaktau', 'peaktau_uperr', 'peaktau_loerr', 'success_ratio']
    column_name.extend([f"r_{col+1}" for col in range(n_r)])
    results = pd.DataFrame(results, columns=column_name)

    return results


class CrossCorrelation_CCF(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        # make train features
        train_ts = pd.read_csv('../../data/input/training_set.csv')

        # make features
        pass1 = 1
        pass2 = 5
        results = calc_ccf(train_ts, pass1, pass2)

        results.sort_values('object_id', ascending=True, inplace=True)   # object_idの降順にソート
        results.drop('object_id', axis=1, inplace=True)
        self.train_feature = results
        # dummy
        self.test_feature = results

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    # load data
    train_meta = pd.read_csv('../../data/input/training_set_metadata.csv')
    test_meta = pd.read_csv('../../data/input/test_set_metadata.csv')
    print('train_meta:', train_meta.shape)
    print('test_meta:', test_meta.shape)

    # make CrossCorrelation_CCF features
    f = CrossCorrelation_CCF('../../data/feature/')
    f.run(train_meta, test_meta).save()

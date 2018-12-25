import feather
import pandas as pd
import numpy as np
import sys
import itertools
import sncosmo
import astropy
import warnings
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from astropy.time import Time
from joblib import Parallel, delayed
from multiprocessing import Pool
from scipy.optimize import fmin_l_bfgs_b

if Path.cwd().name == 'kaggle_plasticc':
    from .base import Feature
elif Path.cwd().name == 'features':
    from base import Feature


def create_SALT2_model(ts_data, meta_data, object_id_list):
    warnings.simplefilter("ignore")

    result_list = []
    for object_id in object_id_list:
        if object_id is None:
            # for itertools.zip_longest method
            continue

        result_tmp = [object_id]
        photoz = meta_data.query('object_id == @object_id')["hostgal_photoz"].values[0]
        mwebv = meta_data.query('object_id == @object_id')["mwebv"].values[0]

        if photoz == 0:
            result_tmp.extend([np.nan, np.nan, np.nan, np.nan])
            result_list.append(result_tmp)
            continue

        extract_obj = ts_data.query('object_id == @object_id')
        data = astropy.table.Table(
            extract_obj.values,
            names=['object_id', 'mjd', 'band', 'flux', 'flux_err', 'detected'],
            dtype=('str', 'float', 'str', 'float', 'float', 'str')
        )

        # model = sncosmo.Model(source='salt2')
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(
            source='salt2-extended',
            effects=[dust, dust],
            effect_names=['host', 'mw'],
            effect_frames=['rest', 'obs'])
        model.set(z=photoz, mwebv=mwebv, hostebv=0, hostr_v=3.1, mwr_v=3.1)

        # cutting data
        data_mask = model.bandoverlap(data["band"], z=photoz)
        data = data[data_mask]

        def objective(parameters):
            model.parameters[1:5] = parameters
            model_flux = model.bandflux(data['band'], data['mjd'])
            return np.sum(((data['flux'] - model_flux) / data['flux_err'])**2)

        start_parameters = [60000, 1e-5, 0., 0.]  # t0, x0, x1, c
        bounds = [(extract_obj.mjd.min(), extract_obj.mjd.max()), (None, None), (None, None), (None, None)]
        parameters, val, info = fmin_l_bfgs_b(objective, start_parameters, bounds=bounds, approx_grad=True)
        result_tmp.extend(parameters)

        result_list.append(result_tmp)

    return result_list


def get_parallel_object_id(object_id_list, n_obj):
    parallel_object_id_list = []

    for list_ in itertools.zip_longest(*[iter(object_id_list)] * n_obj):
        parallel_object_id_list.append(list_)

    return parallel_object_id_list


class SALT2_model(Feature):
    def create_features(self, train_meta, test_meta):
        """
        train: meta-data
        test: meta-data
        """
        # make train features
        train_ts = pd.read_csv('../../data/input/training_set.csv')
        train_ts = train_ts.sort_values(['object_id', 'mjd'])

        # make features
        passband_dict = {0: "lsstu", 1: "lsstg", 2: "lsstr", 3: "lssti", 4: "lsstz", 5: "lssty"}
        train_ts["passband"] = train_ts["passband"].map(passband_dict)
        n_obj = 10
        parallel_object_id_list = get_parallel_object_id(train_meta.object_id.unique(), n_obj)

        result_tmp = Parallel(n_jobs=1, verbose=1)(
            [delayed(create_SALT2_model)(
                train_ts.query('object_id in @object_id_list'),
                train_meta.query('object_id in @object_id_list'),
                object_id_list
            ) for object_id_list in parallel_object_id_list])

        result_extend = []
        for l in result_tmp:
            result_extend.extend(l)

        results = pd.DataFrame(
            result_extend,
            columns=['object_id', 'SALT2_t0', 'SALT2_x0', 'SALT2_x1', 'SALT2_c']
        )
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

    # make SALT2_model features
    f = SALT2_model('../../data/feature/')
    f.run(train_meta, test_meta).save()

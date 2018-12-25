import json
import argparse
import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe, Trials
from src.data.load_dataset import load_dataset
from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json
from src.features.base import load_features
from src.models.lightgbm import LightGBM, multi_weighted_logloss
from src.models.get_folds import get_StratifiedKFold


model_map = {
    'lightgbm': LightGBM
}

space = {
    'num_leaves': hp.quniform('num_leaves', 30, 100, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'max_bin': hp.quniform('max_bin', 100, 400, 50),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 30, 5),
    'learning_rate': hp.uniform('learning_rate', 0.02, 0.05),
}


def main():
    """run.pyの結果を使って、モデルのチューニングを行う。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./config/lightgbm_52.json')
    parser.add_argument('--feature', '-f', default='./data/output/output_52.json')
    parser.add_argument('--out', '-o', default='tune_0')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))
    args_log = {"args": {
        "feature": args.feature,
        "config": args.config,
        "out": args.out
    }}
    config.update(args_log)

    feature = json.load(open(args.feature))
    feature_list_gal = [*feature["evals_result_gal"]["feature_importance"].keys()]
    feature_list_exgal = [*feature["evals_result_exgal"]["feature_importance"].keys()]

    # load dataset
    # read only metadata.
    logger.info('load dataset.')
    train_meta, test_meta = load_dataset(feature, 'meta', True)
    logger.debug(f'train_meta: {train_meta.shape}, test_meta: {test_meta.shape}')

    # load features
    logger.info('load features')
    x_train, _ = load_features(feature, True)
    logger.debug(f'number of features: {x_train.shape[1]}')

    # pre-processing (get target values)
    y_train = train_meta['target']

    # check classes
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1.00104, 15: 2.00189, 16: 1.00104, 42: 1.00104, 52: 1.00104, 53: 1.00000, 62: 1.00104,
                     64: 2.00710, 65: 1.00104, 67: 1.00104, 88: 1.00104, 90: 1.00104, 92: 1.00104, 95: 1.00104}

    # galastic vs extra-galastic
    gal_classes = [6, 16, 53, 65, 92]
    exgal_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]
    gal_class_weights = {k: v for k, v in class_weights.items() if k in gal_classes}
    exgal_class_weights = {k: v for k, v in class_weights.items() if k in exgal_classes}

    train_gal_index = train_meta.query("hostgal_photoz == 0").index
    train_exgal_index = train_meta.query("hostgal_photoz != 0").index
    x_train_gal = x_train.loc[train_gal_index].reset_index(drop=True)
    y_train_gal = y_train.loc[train_gal_index].reset_index(drop=True)
    x_train_exgal = x_train.loc[train_exgal_index].reset_index(drop=True)
    y_train_exgal = y_train.loc[train_exgal_index].reset_index(drop=True)

    x_train_gal = x_train_gal[feature_list_gal]
    x_train_exgal = x_train_exgal[feature_list_exgal]
    logger.debug(f'x_train_gal: {x_train_gal.shape}')
    logger.debug(f'y_train_gal: {y_train_gal.shape}')
    logger.debug(f'x_train_exgal: {x_train_exgal.shape}')
    logger.debug(f'y_train_exgal: {y_train_exgal.shape}')

    # train galastic model
    logger.info(f'train galastic model')
    validation_name = config['cv']['method']
    if validation_name == 'StratifiedKFold':
        folds = get_StratifiedKFold(y_train_gal)

    def objective(params):
        config['gal_model']["model_params"]['num_leaves'] = int(params['num_leaves'])
        config['gal_model']["model_params"]['max_depth'] = int(params['max_depth'])
        config['gal_model']["model_params"]['max_bin'] = int(params['max_bin'])
        config['gal_model']["model_params"]['min_data_in_leaf'] = int(params['min_data_in_leaf'])
        config['gal_model']["model_params"]['learning_rate'] = params['learning_rate']

        model_name = config['gal_model']['name']
        model = model_map[model_name]()
        clfs_gal, oof_preds_gal, evals_result_gal = model.cv(
            x_train=x_train_gal, y_train=y_train_gal, folds=folds, params=config['gal_model'],
            classes=gal_classes, class_weights=gal_class_weights
        )
        oof_score_gal = multi_weighted_logloss(y_train_gal, oof_preds_gal, gal_classes, gal_class_weights)

        return oof_score_gal

    trials = Trials()
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    result = {
        'result_gal_model': {
            'best': best,
            'trials': trials.best_trial
        }
    }
    config.update(result)

    # train extra-galastic model
    logger.info(f'train extra-galastic model')
    validation_name = config['cv']['method']
    if validation_name == 'StratifiedKFold':
        folds = get_StratifiedKFold(y_train_exgal)

    def objective(params):
        config['exgal_model']["model_params"]['num_leaves'] = int(params['num_leaves'])
        config['exgal_model']["model_params"]['max_depth'] = int(params['max_depth'])
        config['exgal_model']["model_params"]['max_bin'] = int(params['max_bin'])
        config['exgal_model']["model_params"]['min_data_in_leaf'] = int(params['min_data_in_leaf'])
        config['exgal_model']["model_params"]['learning_rate'] = params['learning_rate']

        model_name = config['exgal_model']['name']
        model = model_map[model_name]()
        clfs_exgal, oof_preds_exgal, evals_result_exgal = model.cv(
            x_train=x_train_exgal, y_train=y_train_exgal, folds=folds, params=config['exgal_model'],
            classes=exgal_classes, class_weights=exgal_class_weights
        )
        oof_score_exgal = multi_weighted_logloss(y_train_exgal, oof_preds_exgal, exgal_classes, exgal_class_weights)

        return oof_score_exgal

    trials = Trials()
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    result = {
        'result_exgal_model': {
            'best': best,
            'trials': trials.best_trial
        }
    }
    config.update(result)

    # save json file
    save_json(config, args.out, logger)


if __name__ == '__main__':
    main()

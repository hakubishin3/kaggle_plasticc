import json
import argparse
import numpy as np
import pandas as pd
from src.data.load_dataset import load_dataset
from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json
from src.utils.get_conf_mat import get_conf_mat
from src.features.base import load_features
from src.models.lightgbm import LightGBM, multi_weighted_logloss
from src.models.get_folds import get_StratifiedKFold


model_map = {
    'lightgbm': LightGBM
}


def iter_run(x_train, y_train, folds, params, classes, class_weights, evals_result, model_name, logger, select_per_iter=50, n_iter=4):
    for iter_ in range(n_iter):
        if iter_ == 0:
            current_oof_score = evals_result["evals_result"]["oof_score"]
            features_sort_list = [*evals_result["evals_result"]["feature_importance"]]

        n_select_features = (iter_ + 1) * select_per_iter
        feature_list = features_sort_list[:n_select_features]
        model = model_map[model_name]()
        _, _, evals_result = model.cv(
            x_train=x_train[feature_list], y_train=y_train, folds=folds, params=params,
            classes=classes, class_weights=class_weights
        )

        logger.info(f'train iteration: {iter_ + 1}, oof_score: {evals_result["evals_result"]["oof_score"]}')
        if current_oof_score > evals_result["evals_result"]["oof_score"]:
            best_iter = iter_
            current_oof_score = evals_result["evals_result"]["oof_score"]

    # finish iteration
    logger.info(f'best iteration: {best_iter + 1}, oof_score: {current_oof_score}')
    n_select_features = (best_iter + 1) * select_per_iter
    feature_list = features_sort_list[:n_select_features]

    model = model_map[model_name]()
    clfs, oof_preds, evals_result = model.cv(
        x_train=x_train[feature_list], y_train=y_train, folds=folds, params=params,
        classes=classes, class_weights=class_weights
    )

    return clfs, oof_preds, evals_result, feature_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./configs/lightgbm_0.json')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--iter', '-i', action='store_true')
    parser.add_argument('--out', '-o', default='output_0')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))
    args_log = {"args": {
        "config": args.config,
        "debug_mode": args.debug,
        "iter": args.iter,
        "out": args.out
    }}
    config.update(args_log)

    # load dataset
    logger.info('load dataset.')
    train_meta, test_meta = load_dataset(config, 'meta', args.debug)
    logger.debug(f'train_meta: {train_meta.shape}, test_meta: {test_meta.shape}')

    # load features
    logger.info('load features')
    x_train, x_test = load_features(config, args.debug)
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

    x_train_gal.dropna(how="all", axis=1, inplace=True)     # 全ての値が欠損である列を除外
    x_train_exgal.dropna(how="all", axis=1, inplace=True)   # 全ての値が欠損である列を除外
    std_gal = x_train_gal.apply(lambda x: np.std(x.dropna()))
    x_train_gal.drop(std_gal[std_gal == 0].index.tolist(), axis=1, inplace=True)   # 同じ値しか持たない列を除外
    std_exgal = x_train_exgal.apply(lambda x: np.std(x.dropna()))
    x_train_exgal.drop(std_exgal[std_exgal == 0].index.tolist(), axis=1, inplace=True)   # 同じ値しか持たない列を除外
    logger.debug(f'x_train_gal: {x_train_gal.shape}')
    logger.debug(f'y_train_gal: {y_train_gal.shape}')
    logger.debug(f'x_train_exgal: {x_train_exgal.shape}')
    logger.debug(f'y_train_exgal: {y_train_exgal.shape}')

    if args.debug is False:
        test_gal_index = test_meta.query("hostgal_photoz == 0").index
        test_exgal_index = test_meta.query("hostgal_photoz != 0").index
        x_test_gal = x_test.loc[test_gal_index].reset_index(drop=True)
        x_test_exgal = x_test.loc[test_exgal_index].reset_index(drop=True)

        test_gal_objectid = test_meta.query("hostgal_photoz == 0").object_id.values
        test_exgal_objectid = test_meta.query("hostgal_photoz != 0").object_id.values
        x_test_gal = x_test_gal[x_train_gal.columns]
        x_test_exgal = x_test_exgal[x_train_exgal.columns]

    # train galastic model
    logger.info(f'train galastic model')
    validation_name = config['cv']['method']
    if validation_name == 'StratifiedKFold':
        folds = get_StratifiedKFold(y_train_gal)

    model_name = config['gal_model']['name']
    model = model_map[model_name]()
    clfs_gal, oof_preds_gal, evals_result_gal = model.cv(
        x_train=x_train_gal, y_train=y_train_gal, folds=folds, params=config['gal_model'],
        classes=gal_classes, class_weights=gal_class_weights, original_index=train_gal_index
    )
    if args.iter:
        clfs_gal, oof_preds_gal, evals_result_gal, gal_feature_list = iter_run(
            x_train_gal, y_train_gal, folds, config['gal_model'], gal_classes, gal_class_weights,
            evals_result_gal, model_name, logger, select_per_iter=30, n_iter=4)
        if args.debug is False:
            x_test_gal = x_test_gal[gal_feature_list]

    evals_result_gal["evals_result_gal"] = evals_result_gal["evals_result"]
    evals_result_gal.pop("evals_result")
    config.update(evals_result_gal)

    # train extra-galastic model
    logger.info(f'train extra-galastic model')
    validation_name = config['cv']['method']
    if validation_name == 'StratifiedKFold':
        folds = get_StratifiedKFold(y_train_exgal)

    model_name = config['exgal_model']['name']
    model = model_map[model_name]()
    clfs_exgal, oof_preds_exgal, evals_result_exgal = model.cv(
        x_train=x_train_exgal, y_train=y_train_exgal, folds=folds, params=config['exgal_model'],
        classes=exgal_classes, class_weights=exgal_class_weights, original_index=train_exgal_index
    )

    if args.iter:
        clfs_exgal, oof_preds_exgal, evals_result_exgal, exgal_feature_list = iter_run(
            x_train_exgal, y_train_exgal, folds, config['exgal_model'], exgal_classes, exgal_class_weights,
            evals_result_exgal, model_name, logger, select_per_iter=50, n_iter=4)
        if args.debug is False:
            x_test_exgal = x_test_exgal[exgal_feature_list]

    evals_result_exgal["evals_result_exgal"] = evals_result_exgal["evals_result"]
    evals_result_exgal.pop("evals_result")
    config.update(evals_result_exgal)

    # check total oof-score
    y_train_tmp = pd.concat([y_train_gal, y_train_exgal], axis=0)
    oof_preds_gal_df = pd.DataFrame(oof_preds_gal, columns=[f"class_{col}" for col in gal_classes])
    oof_preds_exgal_df = pd.DataFrame(oof_preds_exgal, columns=[f"class_{col}" for col in exgal_classes])
    oof_preds_tmp = pd.concat([oof_preds_gal_df, oof_preds_exgal_df], axis=0, sort=True).fillna(0)
    oof_preds_tmp = oof_preds_tmp[[f"class_{col}" for col in classes]].values   # sort columns
    oof_score = multi_weighted_logloss(y_train_tmp, oof_preds_tmp, classes, class_weights)
    logger.debug(f'OOF Score: {oof_score}')
    config.update({"total_oof_score": oof_score})

    # make confusion matrix
    y_pred_train = np.array([gal_classes[i] for i in oof_preds_gal.argmax(axis=1)])
    ax = get_conf_mat(y_train_gal, y_pred_train, gal_classes)
    ax.figure.savefig(f'./data/output/{args.out}_confmat_gal.png')

    y_pred_train = np.array([exgal_classes[i] for i in oof_preds_exgal.argmax(axis=1)])
    ax = get_conf_mat(y_train_exgal, y_pred_train, exgal_classes)
    ax.figure.savefig(f'./data/output/{args.out}_confmat_exgal.png')

    # prediction
    logger.info('prediction')

    preds_gal = None
    for clf_gal in clfs_gal:
        if preds_gal is None:
            preds_gal = clf_gal.predict_proba(x_test_gal, num_iteration=clf_gal.best_iteration_) / len(clfs_gal)
        else:
            preds_gal += clf_gal.predict_proba(x_test_gal, num_iteration=clf_gal.best_iteration_) / len(clfs_gal)
    preds_gal_df = pd.DataFrame(preds_gal, columns=['class_' + str(s) for s in clfs_gal[0].classes_])
    preds_gal_df['object_id'] = test_gal_objectid

    preds_exgal = None
    for clf_exgal in clfs_exgal:
        if preds_exgal is None:
            preds_exgal = clf_exgal.predict_proba(x_test_exgal, num_iteration=clf_exgal.best_iteration_) / len(clfs_exgal)
        else:
            preds_exgal += clf_exgal.predict_proba(x_test_exgal, num_iteration=clf_exgal.best_iteration_) / len(clfs_exgal)
    preds_exgal_df = pd.DataFrame(preds_exgal, columns=['class_' + str(s) for s in clfs_exgal[0].classes_])
    preds_exgal_df['object_id'] = test_exgal_objectid
    pred_df = pd.concat([preds_gal_df, preds_exgal_df], axis=0, sort=True).fillna(0)

    preds_99 = np.ones(pred_df.shape[0])
    for col in ['class_' + str(s) for s in classes]:
        preds_99 *= (1 - pred_df[col])

    pred_df['class_99'] = 0.18 * preds_99 / np.mean(preds_99)

    pred_df.to_csv(f'./data/output/predictions_{args.out}.csv', header=True, index=False, float_format='%.6f')

    pred_results = {"pred_result": {
        "class_99_mean": pred_df['class_99'].mean(),
        "class_99_std": pred_df['class_99'].std()
    }}
    config.update(pred_results)

    # save json file
    save_json(config, args.out, logger)


if __name__ == '__main__':
    main()

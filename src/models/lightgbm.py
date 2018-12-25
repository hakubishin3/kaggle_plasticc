import copy
import pandas as pd
import lightgbm as lgb
import eli5
import numpy as np
from typing import List, Tuple
from functools import partial


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)

    return loss


class LightGBM(object):
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1.00104, 15: 2.00189, 16: 1.00104, 42: 1.00104, 52: 1.00104, 53: 1.00000, 62: 1.00104,
                     64: 2.00710, 65: 1.00104, 67: 1.00104, 88: 1.00104, 90: 1.00104, 92: 1.00104, 95: 1.00104}
    """
    def fit(self, x_train, y_train, x_valid, y_valid, params: dict, classes, class_weights, sample_weights):
        # Compute class weights
        lgb_class_weights = {k: v / sum(class_weights.values()) for k, v in class_weights.items()}
        lgb_class_weights = {k: v for k, v in enumerate(lgb_class_weights.values())}

        # Compute eval metric
        def lgb_multi_weighted_logloss(y_true, y_preds):
            loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
            return 'wloss', loss, False

        evals_result = {}
        clf = lgb.LGBMClassifier(class_weight=lgb_class_weights, **params['model_params'])
        clf.fit(
            x_train, y_train,
            eval_metric=lgb_multi_weighted_logloss,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_names=['train', 'valid'],
            sample_weight=y_train.map(sample_weights).values,
            **params['train_params']
        )
        return clf, evals_result

    def cv(self, x_train, y_train, folds, params: dict, classes, class_weights, original_index):
        # init predictions
        oof_preds = np.zeros([x_train.shape[0], len(classes)])
        importances = pd.DataFrame(index=x_train.columns)
        best_iteration = 0
        cv_score_list = []
        clfs = []

        # for pseudo labeling
        w = y_train[:len(original_index)].value_counts()
        sample_weights = {i: np.sum(w) / w[i] for i in w.index}

        # Run cross-validation
        n_folds = len(folds)

        for i_fold, (trn_idx, val_idx) in enumerate(folds):
            # train model
            clf, evals_result = self.fit(
                x_train.iloc[trn_idx], y_train[trn_idx],
                x_train.iloc[val_idx], y_train[val_idx], params,
                classes, class_weights, sample_weights
            )
            cv_score_list.append(dict(clf.best_score_))
            best_iteration += clf.best_iteration_ / n_folds

            # predict out-of-fold and test
            oof_preds[val_idx, :] = clf.predict_proba(
                x_train.iloc[val_idx], num_iteration=clf.best_iteration_)

            # get feature importances
            importances_tmp = pd.DataFrame(
                clf.feature_importances_,
                columns=[f'gain_{i_fold+1}'],
                index=x_train.columns
            )
            importances = importances.join(importances_tmp, how='inner')
            clfs.append(clf)

        oof_score = multi_weighted_logloss(y_train, oof_preds, classes, class_weights)
        print(f'OOF Score: {oof_score}')

        feature_importance = importances.mean(axis=1)
        feature_importance = feature_importance.sort_values(ascending=False).to_dict()

        train_results = {"evals_result": {
            "oof_score": oof_score,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
            "n_data": len(x_train),
            "best_iteration": best_iteration,
            "n_features": len(x_train.columns),
            "feature_importance": feature_importance
        }}

        return clfs, oof_preds, train_results

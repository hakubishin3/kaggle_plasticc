import json
import argparse
import numpy as np
import pandas as pd
import copy
import gc
from collections import Counter, OrderedDict
from operator import itemgetter
from src.data.load_dataset import load_dataset
from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json
from src.utils.get_conf_mat import get_conf_mat
from src.models.get_folds import get_StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.data.rnn_preprocess import get_timeseries, preprocess, preprocess_trans
from keras.utils import to_categorical
from src.models.RNN import build_model, multi_weighted_logloss
from keras.utils import plot_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./configs/rnn_1.json')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--out', '-o', default='output_rnn_0')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))
    args_log = {"args": {
        "config": args.config,
        "debug_mode": args.debug,
        "out": args.out
    }}
    config.update(args_log)

    # load dataset
    logger.info('load dataset.')
    train_meta = pd.read_csv("./data/input/training_set_metadata.csv")
    test_meta = pd.read_csv("./data/input/test_set_metadata.csv")
    logger.debug(f'train_meta: {train_meta.shape}, test_meta: {test_meta.shape}')
    train_ts = pd.read_csv("./data/input/training_set.csv")
    logger.debug(f'train_ts: {train_ts.shape}')

    # =========================================
    # === prepare lightgbm features
    # =========================================

    # load lightgbm features
    logger.info('load lightgbm features')
    x_train = pd.read_feather("./data/feature/train_sub19.ftr")
    x_train = x_train.query("object_id in @train_meta.object_id.tolist()")

    x_test_1 = pd.read_feather("./data/feature/test_sub19_1.ftr")
    x_test_2 = pd.read_feather("./data/feature/test_sub19_2.ftr")
    x_test = pd.concat([x_test_1, x_test_2], axis=0)

    x_train = x_train.sort_values("object_id")
    x_train.drop("object_id", axis=1, inplace=True)
    x_test = x_test.sort_values("object_id")
    x_test.drop("object_id", axis=1, inplace=True)
    x_test = x_test[x_train.columns]
    logger.debug(f'number of features: {x_train.shape}')

    # pre-processing features
    logger.info('pre-processing lightgbm features')
    x_all = pd.concat([x_train, x_test], axis=0)
    train_index = range(0, len(x_train))
    test_index = range(len(x_train), len(x_train) + len(x_test))
    columns_list = x_all.columns

    # add missing flg
    isnull = x_all.isnull().sum()
    missing_col_list = isnull[isnull != 0].index.tolist()
    for col in missing_col_list:
        new_col = f"missing_flg_{col}"
        x_all[new_col] = x_all[col].isnull().astype(int)

    # correct missing value by mean
    x_train = x_all.iloc[train_index]
    x_test = x_all.iloc[test_index]
    x_train = x_train.fillna(x_train.mean())
    x_test = x_test.fillna(x_test.mean())
    x_all = pd.concat([x_train, x_test], axis=0)

    # standard scaler
    ss = StandardScaler()
    x_all[columns_list] = ss.fit_transform(x_all[columns_list])
    x_train = x_all.iloc[train_index]
    x_test = x_all.iloc[test_index]

    # =========================================
    # === prepare RNN features
    # =========================================

    # pre-processing(flux factor)
    train_ts = pd.merge(train_ts, train_meta[["object_id", "distmod"]], on="object_id")
    train_ts["distmod"] = train_ts["distmod"].fillna(0)
    flux = train_ts["flux"].values
    train_ts["flux"] = np.where(flux < 1, 2.5 * np.log10(-flux + 2), -2.5 * np.log10(flux)) - train_ts['distmod']

    logger.info('pre-processing: pad_sequences')
    passband_list = [0, 1, 2, 3, 4, 5]
    x_pad_list = []

    for passband in passband_list:
        x_pad = get_timeseries(train_ts, passband)
        x_pad_list.append(x_pad)
        logger.debug(f'passband:{passband}, {x_pad.shape}')

    # pre-processing(scaling)
    logger.info('pre-processing: scaling')
    x_scaled_list = []
    mean_list = []
    scale_list = []
    for i, x_pad in enumerate(x_pad_list):
        x_scaled, means, scales = preprocess(x_pad)
        x_scaled = np.nan_to_num(x_scaled)   # 欠損を0に変換。maskingは0に設定しておく。
        x_scaled_list.append(x_scaled)
        mean_list.append(means)
        scale_list.append(scales)
        logger.debug(f'passband:{i}, {x_scaled.shape}')

    # =========================================
    # === prepare model setting
    # =========================================

    # check classes
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1.00104, 15: 2.00189, 16: 1.00104, 42: 1.00104, 52: 1.00104, 53: 1.00000, 62: 1.00104,
                     64: 2.00710, 65: 1.00104, 67: 1.00104, 88: 1.00104, 90: 1.00104, 92: 1.00104, 95: 1.00104}

    # pre-processing (get target values)
    y_train = train_meta['target']
    unique_y = np.unique(y_train)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i
    y_map = np.zeros((y_train.shape[0],))
    y_map = np.array([class_map[val] for val in y_train])
    y_categorical = to_categorical(y_map)

    y_count = Counter(y_map)
    wtable = np.zeros((len(unique_y),))
    for i in range(len(unique_y)):
        wtable[i] = y_count[i] / y_map.shape[0]

    # get fold
    folds = get_StratifiedKFold(y_map)

    # =========================================
    # === train model
    # =========================================
    model_list = []
    loss_func_list = []
    val_loss_list = []
    history_list = []
    oof_preds = np.zeros((len(x_train), len(classes)))

    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        x_scaled_list_train = [x_scaled[trn_idx] for x_scaled in x_scaled_list]
        x_scaled_list_valid = [x_scaled[val_idx] for x_scaled in x_scaled_list]

        model, val_loss, pred, loss_func = build_model(
            x_train.iloc[trn_idx], x_scaled_list_train, y_categorical[trn_idx],
            x_train.iloc[val_idx], x_scaled_list_valid, y_categorical[val_idx],
            config, wtable, i_fold
        )
        model_list.append(model)
        loss_func_list.append(loss_func)
        val_loss_list.append(val_loss)
        # history_list.append(history.history)
        oof_preds[val_idx, :] = pred

        logger.debug(f'oof loss: {multi_weighted_logloss(y_categorical[val_idx], pred, classes, class_weights)}')

    oof_score = multi_weighted_logloss(y_categorical, oof_preds, classes, class_weights)
    logger.debug(f'Multi weighted log loss: {oof_score}')

    train_results = {"evals_result": {
        "oof_score": oof_score,
        "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(val_loss_list)},
        "n_features": len(x_train.columns),
        # "history": {f"history{i+1}": history for i, history in enumerate(history_list)}
    }}
    config.update(train_results)
    plot_model(model, to_file=f'./data/output/{args.out}_model.png', show_shapes=True)

    # make confusion matrix
    y_pred_train = np.array([classes[i] for i in oof_preds.argmax(axis=1)])
    ax = get_conf_mat(y_train, y_pred_train, classes)
    ax.figure.savefig(f'./data/output/{args.out}_confmat.png')

    # =========================================
    # === prepare test data for prediction
    # =========================================

    logger.info('pre-processing: pad_sequences')
    passband_list = [0, 1, 2, 3, 4, 5]
    x_pad_list = []

    dir_path = "./data/interim/rnn/"
    x_pad_passband0_np = dir_path + "test_x_pad_passband0_pure.npy"
    x_pad_passband1_np = dir_path + "test_x_pad_passband1_pure.npy"
    x_pad_passband2_np = dir_path + "test_x_pad_passband2_pure.npy"
    x_pad_passband3_np = dir_path + "test_x_pad_passband3_pure.npy"
    x_pad_passband4_np = dir_path + "test_x_pad_passband4_pure.npy"
    x_pad_passband5_np = dir_path + "test_x_pad_passband5_pure.npy"
    x_pad_np_list = [
        x_pad_passband0_np, x_pad_passband1_np, x_pad_passband2_np,
        x_pad_passband3_np, x_pad_passband4_np, x_pad_passband5_np
    ]

    for passband, x_pad_np in enumerate(x_pad_np_list):
        x_pad = np.load(x_pad_np)
        x_pad_list.append(x_pad)
        logger.debug(f'passband:{passband}, {x_pad.shape}')

    # pre-processing(scaling)
    logger.info('pre-processing: scaling')
    x_scaled_list = []
    for i, x_pad in enumerate(x_pad_list):
        mean = mean_list[i]
        scale = scale_list[i]
        x_scaled = preprocess_trans(x_pad, mean, scale)
        # x_scaled, means, scales = preprocess(x_pad)
        x_scaled = np.nan_to_num(x_scaled)   # 欠損を0に変換。maskingは0に設定しておく。
        x_scaled_list.append(x_scaled)
        logger.debug(f'passband:{i}, {x_scaled.shape}')

    # remove flux_err
    for i in passband_list:
        x_scaled_list[i] = x_scaled_list[i][:, :, :2]

    x_test_set = {
        'main_input_0': x_scaled_list[0],
        'main_input_1': x_scaled_list[1],
        'main_input_2': x_scaled_list[2],
        'main_input_3': x_scaled_list[3],
        'main_input_4': x_scaled_list[4],
        'main_input_5': x_scaled_list[5],
        'fc_input': x_test
    }

    # =========================================
    # === prediction
    # =========================================

    logger.info('prediction')
    preds = None
    for model in model_list:
        if preds is None:
            preds = model.predict(x_test_set, batch_size=16384) / len(model_list)
        else:
            preds += model.predict(x_test_set, batch_size=16384) / len(model_list)
    pred_df = pd.DataFrame(preds, columns=['class_' + str(s) for s in classes])
    pred_df['object_id'] = test_meta.object_id.values

    preds_99 = np.ones(pred_df.shape[0])
    for col in ['class_' + str(s) for s in classes]:
        preds_99 *= (1 - pred_df[col])

    # Create DataFrame from predictions
    pred_df['class_99'] = 0.18 * preds_99 / np.mean(preds_99)
    pred_df.to_csv(f'./data/output/predictions_{args.out}.csv', header=True, index=False, float_format='%.6f')

    # save json file
    save_json(config, args.out, logger)


if __name__ == '__main__':
    main()

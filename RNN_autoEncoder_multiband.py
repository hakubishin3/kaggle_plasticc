import json
import os
import codecs
import argparse
import random
import numpy as np
import pandas as pd
from src.data.rnn_preprocess import get_timeseries, preprocess
from src.data.load_dataset import load_dataset
from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json, MyEncoder
from src.models.autoencoder import encoder, decoder, build_model
from joblib import Parallel, delayed
from keras.utils import plot_model
from src.models.get_folds import get_StratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--model_no', '-o', type=str)   # example: v0.0
    parser.add_argument('--passband', '-p', type=int)
    parser.add_argument('--type', '-t', type=str, choices=["gal", "exgal"])
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))
    args_log = {"args": {
        "config": args.config,
        "model_no": args.model_no,
        "passband": args.passband,
        "type": args.type
    }}
    config.update(args_log)

    # load meta-data
    logger.info('load meta-data.')
    train_meta, test_meta = load_dataset(config, 'meta', False)
    if args.type == "gal":
        train_meta = train_meta[train_meta.hostgal_photoz == 0]
        test_meta = test_meta[test_meta.hostgal_photoz == 0]
    elif args.type == "exgal":
        train_meta = train_meta[train_meta.hostgal_photoz != 0]
        test_meta = test_meta[test_meta.hostgal_photoz != 0]
    logger.debug(f'train_meta: {train_meta.shape}, test_meta: {test_meta.shape}')

    # load ts-data
    logger.info('load ts-data.')
    train_ts = pd.read_csv("./data/input/training_set.csv")
    test_ts = pd.read_feather("./data/input/test_set.ftr")
    train_ts = train_ts.query("object_id in @train_meta.object_id.tolist()")
    test_ts = test_ts.query("object_id in @test_meta.object_id.tolist()")
    train_ts = train_ts.sort_values(["object_id", "passband", "mjd"])
    test_ts = test_ts.sort_values(["object_id", "passband", "mjd"])
    logger.debug(f'train_ts: {train_ts.shape}, test_ts: {test_ts.shape}')

    # pre-processing(flux factor)
    """
    train_ts = pd.merge(train_ts, train_meta[["object_id", "distmod"]], on="object_id")
    flux = train_ts["flux"].values
    train_ts["flux"] = np.where(flux < 1, 2.5 * np.log10(-flux + 2), -2.5 * np.log10(flux)) - train_ts['distmod']
    test_ts = pd.merge(test_ts, test_meta[["object_id", "distmod"]], on="object_id")
    flux = test_ts["flux"].values
    test_ts["flux"] = np.where(flux < 1, 2.5 * np.log10(-flux + 2), -2.5 * np.log10(flux)) - test_ts['distmod']
    """

    # add ts-data from test
    n_choice = config["model"]["n_use_test"]
    random.seed(71)
    choice_obj_list = random.sample(test_meta.object_id.tolist(), n_choice)
    train_ts = pd.concat([train_ts, test_ts.query("object_id in @choice_obj_list")], axis=0)
    train_ts = train_ts.sort_values(["object_id", "passband", "mjd"])

    # pre-processing(time-series)
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
    for i, x_pad in enumerate(x_pad_list):
        x_scaled, means, scales = preprocess(x_pad)
        x_scaled = np.nan_to_num(x_scaled)   # 欠損を0に変換。maskingは0に設定しておく。
        x_scaled_list.append(x_scaled)
        logger.debug(f'passband:{i}, mean:{means}')
        logger.debug(f'passband:{i}, scales:{scales}')
        logger.debug(f'passband:{i}, {x_scaled.shape}')
        config.update(
            {f"scaling_{i}":
                {"mean": means,
                 "scale": scales
                 }
             })

    # set train and validation
    extract = train_ts.query("passband == @args.passband").groupby("object_id").size()
    fold_ids = get_StratifiedKFold(extract, n_splits=2)
    train_ids = fold_ids[0][0]
    valid_ids = fold_ids[0][1]
    logger.debug(f'n_train: {len(train_ids)}, n_valid: {len(valid_ids)}')

    # set model
    model, history, val_loss = build_model(x_scaled_list, train_ids, valid_ids, config)
    config.update(
        {"train_log":
            {"best": val_loss,
             "history": history.history
             }
         })

    # save result
    output_path = f"./data/output/rnn/model_{args.model_no}/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_json = output_path + "output.json"
    f = codecs.open(output_json, 'w', 'utf-8')
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    logger.info(f'save json-file. {output_json}')

    output_model = output_path + "model.h5"
    plot_model(model, to_file=f'{output_path}model.png', show_shapes=True)
    model.save(output_model)
    logger.info(f'save model. {output_model}')


if __name__ == "__main__":
    main()

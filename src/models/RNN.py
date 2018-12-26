import os
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.layers import (Input, Dense, BatchNormalization, TimeDistributed, LSTM, GRU, Dropout, concatenate,
                          Flatten, RepeatVector, Recurrent, Bidirectional, SimpleRNN, Activation, Masking)
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_ohe = y_true
    y_p = y_preds

    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).astype(float)
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)

    return loss


def getNewestModel(model, dirname):
    """get the newest model file within a directory"""
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model


def RNN_model(main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5, fc_input, config):
    num_layers = config["model"]["num_layers"]
    bidirectional = config["model"]["bidirectional"]
    n_hidden = config["model"]["n_hidden"]
    n_out = config["model"]["n_out"]
    drop_frac = config["model"]["drop_frac"]
    n_class = config["model"]["n_class"]
    l_hidden = config["model"]["l_hidden"]

    # masking input
    main_input_0 = Masking(mask_value=0)(main_input_0)
    main_input_1 = Masking(mask_value=0)(main_input_1)
    main_input_2 = Masking(mask_value=0)(main_input_2)
    main_input_3 = Masking(mask_value=0)(main_input_3)
    main_input_4 = Masking(mask_value=0)(main_input_4)
    main_input_5 = Masking(mask_value=0)(main_input_5)
    main_input_list = [main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5]

    # rnn per passband
    encode_list = []
    for i_band, main_input in enumerate(main_input_list):
        encode = main_input

        for i_layer in range(num_layers):
            wrapper = Bidirectional if bidirectional else lambda x: x
            encode = wrapper(GRU(n_hidden, name=f'encode_{i_band}_{i_layer}',
                             return_sequences=(i_layer < num_layers - 1)))(encode)
            if drop_frac > 0.0:
                encode = Dropout(drop_frac, name=f'drop_encode_{i_band}_{i_layer}')(encode)

        encode_list.append(encode)

    # concat encode
    for i_band, encode in enumerate(encode_list):
        if i_band == 0:
            concat_encode = encode
        else:
            concat_encode = concatenate([concat_encode, encode])

    concat_encode = Dense(n_out, activation='linear', name='encoding')(concat_encode)

    # concat lightgbm features
    concat = concatenate([concat_encode, fc_input])

    l1 = Dense(l_hidden, activation=None)(concat)
    l1 = BatchNormalization()(l1)
    l1 = Activation("relu")(l1)
    l1 = Dropout(drop_frac)(l1)

    l1 = Dense(l_hidden // 2, activation=None)(l1)
    l1 = BatchNormalization()(l1)
    l1 = Activation("relu")(l1)
    l1 = Dropout(drop_frac)(l1)

    l2 = Dense(l_hidden // 4, activation=None)(l1)
    l2 = BatchNormalization()(l2)
    l2 = Activation("relu")(l2)
    l2 = Dropout(drop_frac / 2)(l2)

    out = Dense(n_class, activation="softmax")(l2)

    return out


def build_model(x_fc_train, x_scaled_train, y_train, x_fc_valid, x_scaled_valid, y_valid, config, wtable, i_fold):
    # remove flux_err
    passband_list = [0, 1, 2, 3, 4, 5]
    for i in passband_list:
        x_scaled_train[i] = x_scaled_train[i][:, :, :2]
        x_scaled_valid[i] = x_scaled_valid[i][:, :, :2]

    # make RNN main_input: flux_diff + time_diff
    main_input_0 = Input(shape=(x_scaled_train[0].shape[1], x_scaled_train[0].shape[-1]), name='main_input_0')
    main_input_1 = Input(shape=(x_scaled_train[1].shape[1], x_scaled_train[1].shape[-1]), name='main_input_1')
    main_input_2 = Input(shape=(x_scaled_train[2].shape[1], x_scaled_train[2].shape[-1]), name='main_input_2')
    main_input_3 = Input(shape=(x_scaled_train[3].shape[1], x_scaled_train[3].shape[-1]), name='main_input_3')
    main_input_4 = Input(shape=(x_scaled_train[4].shape[1], x_scaled_train[4].shape[-1]), name='main_input_4')
    main_input_5 = Input(shape=(x_scaled_train[5].shape[1], x_scaled_train[5].shape[-1]), name='main_input_5')
    fc_input = Input(shape=(x_fc_train.shape[1],), name='fc_input')

    model_input = [
        main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5, fc_input
    ]

    # construct RNN AutoEncoder
    rnn = RNN_model(main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5, fc_input, config)
    model = Model(model_input, rnn)
    print(model.summary())

    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
    def mywloss(y_true, y_pred):
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
        return loss

    lr = config["model"]["lr"]
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=mywloss)

    # set train and valid
    x_train_set = {
        'main_input_0': x_scaled_train[0],
        'main_input_1': x_scaled_train[1],
        'main_input_2': x_scaled_train[2],
        'main_input_3': x_scaled_train[3],
        'main_input_4': x_scaled_train[4],
        'main_input_5': x_scaled_train[5],
        'fc_input': x_fc_train
    }

    x_valid_set = {
        'main_input_0': x_scaled_valid[0],
        'main_input_1': x_scaled_valid[1],
        'main_input_2': x_scaled_valid[2],
        'main_input_3': x_scaled_valid[3],
        'main_input_4': x_scaled_valid[4],
        'main_input_5': x_scaled_valid[5],
        'fc_input': x_fc_valid
    }

    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    early_stopping_patience = config["model"]["early_stopping_patience"]
    tmp_model_path = "./data/output/rnn/tmp_rnn_model/"

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    chkpt = os.path.join(tmp_model_path, 'weights_.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=chkpt, verbose=1, save_best_only=True, monitor='val_loss')
    history = model.fit(
        x_train_set, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
        validation_data=(x_valid_set, y_valid), shuffle=True, callbacks=[early_stopping, checkpointer]
    )
    # get checkpoint model
    best_model = getNewestModel(model, tmp_model_path)

    val_loss = best_model.evaluate(x_valid_set, y_valid, batch_size=4096)
    val_pred = best_model.predict(x_valid_set, batch_size=4096)

    return best_model, val_loss, val_pred, mywloss

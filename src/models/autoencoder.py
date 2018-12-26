import os
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, LSTM, GRU, Dropout, concatenate,
                          Flatten, RepeatVector, Recurrent, Bidirectional, SimpleRNN, Masking)
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


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


def masked_mean_squared_error(y_true, y_pred):
    mask_value = 0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    n_data = K.sum(mask)   # if data, mask[i] = 1
    squared = K.square(y_pred * mask - y_true * mask)
    loss = K.sum(squared) / n_data
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


def encoder(main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5, config):
    num_layers = config["model"]["num_layers"]
    bidirectional = config["model"]["bidirectional"]
    n_hidden = config["model"]["n_hidden"]
    n_out = config["model"]["n_out"]
    drop_frac = config["model"]["drop_frac"]

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

    return concat_encode


def decoder(encode, aux_input_0, aux_input_1, aux_input_2, aux_input_3, aux_input_4, aux_input_5, config):
    num_layers = config["model"]["num_layers"]
    bidirectional = config["model"]["bidirectional"]
    drop_frac = config["model"]["drop_frac"]
    n_hidden = config["model"]["n_hidden"]
    passband = config["args"]["passband"]

    aux_input_list = [aux_input_0, aux_input_1, aux_input_2, aux_input_3, aux_input_4, aux_input_5]
    aux_input_list = [aux_input_list[passband]]   # only single passband

    decode_list = []
    for i_band, aux_input in enumerate(aux_input_list):
        n_step = aux_input.shape[1].value
        decode = RepeatVector(n_step, name=f'repeat_{i_band}')(encode)
        decode = concatenate([aux_input, decode])

        for i_layer in range(num_layers):
            if drop_frac > 0.0 and i_layer > 0:   # skip these for first layer for symmetry
                decode = Dropout(drop_frac, name=f'drop_decode_{i_band}_{i_layer}')(decode)
                wrapper = Bidirectional if bidirectional else lambda x: x
                decode = wrapper(GRU(n_hidden, name=f'decode_{i_band}_{i_layer}',
                                     return_sequences=True))(decode)

        decode = TimeDistributed(Dense(1, activation='linear'), name=f'time_dist_{i_band}')(decode)
        decode_list.append(decode)

    return decode_list


def build_model(x_scaled_list, train_ids, valid_ids, config):
    # remove flux_err
    passband_list = [0, 1, 2, 3, 4, 5]
    for i in passband_list:
        x_scaled_list[i] = x_scaled_list[i][:, :, :2]

    # target passband
    target_passband = config["args"]["passband"]

    # make main_input: flux_diff + time_diff
    main_input_0 = Input(shape=(x_scaled_list[0].shape[1], x_scaled_list[0].shape[-1]), name='main_input_0')
    main_input_1 = Input(shape=(x_scaled_list[1].shape[1], x_scaled_list[1].shape[-1]), name='main_input_1')
    main_input_2 = Input(shape=(x_scaled_list[2].shape[1], x_scaled_list[2].shape[-1]), name='main_input_2')
    main_input_3 = Input(shape=(x_scaled_list[3].shape[1], x_scaled_list[3].shape[-1]), name='main_input_3')
    main_input_4 = Input(shape=(x_scaled_list[4].shape[1], x_scaled_list[4].shape[-1]), name='main_input_4')
    main_input_5 = Input(shape=(x_scaled_list[5].shape[1], x_scaled_list[5].shape[-1]), name='main_input_5')

    # make aux_input: time_diff
    aux_input_0 = Input(shape=(x_scaled_list[0].shape[1], x_scaled_list[0].shape[-1] - 1), name='aux_input_0')
    aux_input_1 = Input(shape=(x_scaled_list[1].shape[1], x_scaled_list[1].shape[-1] - 1), name='aux_input_1')
    aux_input_2 = Input(shape=(x_scaled_list[2].shape[1], x_scaled_list[2].shape[-1] - 1), name='aux_input_2')
    aux_input_3 = Input(shape=(x_scaled_list[3].shape[1], x_scaled_list[3].shape[-1] - 1), name='aux_input_3')
    aux_input_4 = Input(shape=(x_scaled_list[4].shape[1], x_scaled_list[4].shape[-1] - 1), name='aux_input_4')
    aux_input_5 = Input(shape=(x_scaled_list[5].shape[1], x_scaled_list[5].shape[-1] - 1), name='aux_input_5')

    # merge main_input and aux_input
    model_input = [
        main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5,
        aux_input_0, aux_input_1, aux_input_2, aux_input_3, aux_input_4, aux_input_5
    ]

    # construct RNN AutoEncoder
    encode = encoder(main_input_0, main_input_1, main_input_2, main_input_3, main_input_4, main_input_5, config)
    decode = decoder(encode, aux_input_0, aux_input_1, aux_input_2, aux_input_3, aux_input_4, aux_input_5, config)
    model = Model(model_input, decode)
    print(model.summary())

    lr = config["model"]["lr"]
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=masked_mean_squared_error)

    # set train and valid
    x_train_0 = x_scaled_list[0][train_ids]
    x_train_1 = x_scaled_list[1][train_ids]
    x_train_2 = x_scaled_list[2][train_ids]
    x_train_3 = x_scaled_list[3][train_ids]
    x_train_4 = x_scaled_list[4][train_ids]
    x_train_5 = x_scaled_list[5][train_ids]

    x_valid_0 = x_scaled_list[0][valid_ids]
    x_valid_1 = x_scaled_list[1][valid_ids]
    x_valid_2 = x_scaled_list[2][valid_ids]
    x_valid_3 = x_scaled_list[3][valid_ids]
    x_valid_4 = x_scaled_list[4][valid_ids]
    x_valid_5 = x_scaled_list[5][valid_ids]

    x_train_set = {
        'main_input_0': x_train_0, 'aux_input_0': x_train_0[:, :, [0]],
        'main_input_1': x_train_1, 'aux_input_1': x_train_1[:, :, [0]],
        'main_input_2': x_train_2, 'aux_input_2': x_train_2[:, :, [0]],
        'main_input_3': x_train_3, 'aux_input_3': x_train_3[:, :, [0]],
        'main_input_4': x_train_4, 'aux_input_4': x_train_4[:, :, [0]],
        'main_input_5': x_train_5, 'aux_input_5': x_train_5[:, :, [0]],
    }
    y_train_list = [
        x_scaled_list[target_passband][train_ids][:, :, [1]]
    ]

    x_valid_set = {
        'main_input_0': x_valid_0, 'aux_input_0': x_valid_0[:, :, [0]],
        'main_input_1': x_valid_1, 'aux_input_1': x_valid_1[:, :, [0]],
        'main_input_2': x_valid_2, 'aux_input_2': x_valid_2[:, :, [0]],
        'main_input_3': x_valid_3, 'aux_input_3': x_valid_3[:, :, [0]],
        'main_input_4': x_valid_4, 'aux_input_4': x_valid_4[:, :, [0]],
        'main_input_5': x_valid_5, 'aux_input_5': x_valid_5[:, :, [0]],
    }
    y_valid_list = [
        x_scaled_list[target_passband][valid_ids][:, :, [1]]
    ]

    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    early_stopping_patience = config["model"]["early_stopping_patience"]
    tmp_model_path = "./data/output/rnn/tmp/"

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    chkpt = os.path.join(tmp_model_path, 'weights_.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=chkpt, verbose=1, save_best_only=True, monitor='val_loss')
    history = model.fit(
        x_train_set, y_train_list, epochs=epochs, batch_size=batch_size, verbose=1,
        validation_data=(x_valid_set, y_valid_list), shuffle=True, callbacks=[early_stopping, checkpointer]
    )

    # get checkpoint model
    best_model = getNewestModel(model, tmp_model_path)
    val_loss = best_model.evaluate(x_valid_set, y_valid_list, batch_size=batch_size)

    return best_model, history, val_loss

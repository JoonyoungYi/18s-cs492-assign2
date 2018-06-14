import os
import shutil
import time
import traceback
import random

import tensorflow as tf
import numpy as np

from extra_config import *
from extra_model import fc_model_fn

# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

timestamp = None


def _get_eval_set():
    eval_set = np.load('data/extra2-valid.npy')
    valid_x = eval_set[:, :INPUT_LAYER_SIZE].astype(np.float32) / 255
    valid_y = eval_set[:, INPUT_LAYER_SIZE].astype(np.int32)
    return valid_x, valid_y


def _get_last_time_msg():
    delta = time.time() - timestamp
    return '(%02d:%02d:%02d)' % (delta // 3600, (delta // 60) % 60, delta % 60)


def _get_msg_from_result(result):
    return 'acc={}% loss={}'.format(('%3.2f' % (result['accuracy'] * 100))[:4],
                                    '%.4f' % result['loss'])


def _log(i, train_results, eval_results, final=False):
    if final:
        msg = '\nTEST %4s' % ("{:,}".format(i))
    else:
        msg = 'ITER %4s' % ("{:,}".format(i))
    msg += _get_last_time_msg()
    msg += ' - TRAIN: '
    msg += _get_msg_from_result(train_results)
    msg += ', VALID: '
    msg += _get_msg_from_result(eval_results)
    # msg += ', TEST: '
    # msg += _get_test_result_msg(sess, models)
    print(msg)


def main(model_idx, refresh=False):
    global timestamp

    print('START MODEL:{}'.format(model_idx))

    if refresh:
        if os.path.exists(MODEL_FOLDER_NAME.format(model_idx)):
            shutil.rmtree(MODEL_FOLDER_NAME.format(model_idx))

    # init dataset and models
    train_set = np.load(
        'data/extra2-train.npy')  # 40000 by 3073 = 256 * 256 * 3 + 1
    test_data = np.load('data/extra2-test_img.npy').astype(
        np.float32) / 255  # 10000 by 3072

    # setting classifier
    classifier = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir=MODEL_FOLDER_NAME.format(model_idx))

    train_x = train_set[:, :INPUT_LAYER_SIZE].astype(np.float32) / 255
    train_y = train_set[:, INPUT_LAYER_SIZE].astype(np.int32)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=2000,
        num_epochs=1,
        shuffle=True)

    eval_x, eval_y = _get_eval_set()
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_x},
        y=eval_y,
        num_epochs=1,
        shuffle=False, )

    e = 0
    timestamp = time.time()
    for i in range(1, TRAIN_ITER_NUMBER + 1):
        # print('ITER', i)
        if i % BATCH_ITER_NUMBER == 1:
            # After 1 epoch. reinitilaize batch_x
            e = (i // BATCH_ITER_NUMBER)
            print('EPOCH', e + 1)
            batch_x = train_x
            # batch_x = train_x
            batch_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": batch_x},
                y=train_y,
                batch_size=BATCH_SIZE,
                num_epochs=1,
                shuffle=True)

        classifier.train(input_fn=batch_input_fn, steps=BATCH_SIZE)
        train_results = classifier.evaluate(input_fn=train_input_fn)

        # Eval the model. You can evaluate your trained model with validation data
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        _log(i, train_results, eval_results)

        # Early stop
        if train_results['loss'] < EARLY_STOP_TRAIN_LOSS:
            print('EARLY STOP!')
            break

    # ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data}, shuffle=False)
    pred_results = classifier.predict(input_fn=pred_input)
    pred_list = list(pred_results)
    result = np.asarray([list(x.values())[1] for x in pred_list])
    # ----------------------------------------- ##

    np.save('extra_20183453_network_{}.npy'.format(hidden_layer_number),
            result)


if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # 6: layer-3
    # 5: layer-5
    # 4: layer-7
    model_idx = 4
    main(model_idx, refresh=False)

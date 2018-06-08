import tensorflow as tf

from extra_1_config import *


def fc_model_fn(features, labels, mode):
    """
        Model function for PA2. Convolutional Neural Network.
    """
    # print(features["x"])
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    fc_layer_input_size = 28
    for i in range(hidden_layer_number - 2):
        conv_layer = tf.layers.conv2d(
            inputs=input_layer,
            filters=8 * (i + 2),
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        strides = 1
        fc_layer_input_size -= 1
        pooling_layer = tf.layers.max_pooling2d(
            inputs=conv_layer, pool_size=[2, 2], strides=strides)

        dropout_layer = tf.layers.dropout(
            inputs=pooling_layer,
            rate=DROPOUT_RATE,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        input_layer = dropout_layer

    pooling_layer_flat = tf.reshape(input_layer, [
        -1, fc_layer_input_size * fc_layer_input_size * 8 *
        (hidden_layer_number - 1)
    ])
    dense_layer = tf.layers.dense(
        inputs=pooling_layer_flat, units=2048, activation=tf.nn.relu)
    dropout_layer = tf.layers.dropout(
        inputs=dense_layer,
        rate=DROPOUT_RATE,
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    output_layer = tf.layers.dense(
        inputs=dropout_layer, units=OUTPUT_LAYER_SIZE)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=output_layer, axis=1),
        # Add `softmax_tensor` to the graph.
        # It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    # Also, prepare accuracy.
    loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])

    # Setting Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # This two lines are for bm.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={"accuracy": accuracy})
    else:
        # Setting evaluation metrics (for EVAL mode)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={"accuracy": accuracy}, )

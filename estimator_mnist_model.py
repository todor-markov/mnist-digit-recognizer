import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import train_test_split


def cnn_Nlayer(filters1,
               filters2,
               num_units_dense=1024,
               kernel_size=3,
               reg_strength=0.0,
               dropout_rate=0.4,
               use_batchnorm=False,
               decay_epochs=20,
               batch_size=100):

    assert len(filters1) == len(filters2), (
        'Filter arrays must have the same length')

    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=reg_strength)
    kernel = [kernel_size, kernel_size]

    # Architecture: N x Conv - Pool - N x Conv - Pool - Dense - Dropout -
    #               - Softmax
    def classifier(features, labels, mode):
        net = tf.reshape(features['x'], [-1, 28, 28, 1])

        for num_filters in filters1:
            net = tf.layers.conv2d(
                inputs=net,
                filters=num_filters,
                kernel_size=kernel,
                padding='same',
                activation=tf.nn.relu,
                kernel_regularizer=l2_regularizer)

            if use_batchnorm:
                net = tf.layers.batch_normalization(
                    inputs=net,
                    training=(mode == tf.estimator.ModeKeys.TRAIN))

        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

        for num_filters in filters2:
            net = tf.layers.conv2d(
                inputs=net,
                filters=num_filters,
                kernel_size=kernel,
                padding='same',
                activation=tf.nn.relu,
                kernel_regularizer=l2_regularizer)

            if use_batchnorm:
                net = tf.layers.batch_normalization(
                    inputs=net,
                    training=(mode == tf.estimator.ModeKeys.TRAIN))

        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

        net = tf.reshape(net, [-1, 7 * 7 * filters2[-1]])

        net = tf.layers.dense(
            inputs=net,
            units=num_units_dense,
            activation=tf.nn.relu,
            kernel_regularizer=l2_regularizer)

        net = tf.layers.dropout(
            inputs=net,
            rate=dropout_rate,
            training=(mode == tf.estimator.ModeKeys.TRAIN))

        logits = tf.layers.dense(inputs=net, units=10)

        predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])
        }

        tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

        regularization_loss = tf.losses.get_regularization_losses(),
        base_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        loss = base_loss + tf.reduce_sum(regularization_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return classifier


def train(classifier, X, y, batch_size, hooks=None):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X},
        y=y,
        batch_size=batch_size,
        num_epochs=200,
        shuffle=True)

    classifier.train(
        input_fn=train_input_fn,
        hooks=hooks)


def eval(classifier, X, y):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X},
        y=y,
        num_epochs=1,
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print()
    print(eval_results)
    print()


def predict(classifier, X):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X}, shuffle=False)

    generator = classifier.predict(
        input_fn=predict_input_fn, predict_keys='classes')

    return np.asarray([i['classes'] for i in list(generator)])


if __name__ == '__main__':
    mode = sys.argv[1]
    train_type = sys.argv[2]
    model_dir = sys.argv[3]

    if train_type == 'crossval':
        [train_data, val_data] = train_test_split(
            np.loadtxt('data/train.csv', dtype=int, delimiter=',', skiprows=1),
            test_size=12000,
            shuffle=False)

    if train_type == 'full_data':
        train_data = np.loadtxt(
            'data/train.csv', dtype=int, delimiter=',', skiprows=1)

        val_data = train_data

    X_train = train_data[:, 1:785].astype(np.float32)
    y_train = train_data[:, 0].astype(np.int32)

    X_val = val_data[:, 1:785].astype(np.float32)
    y_val = val_data[:, 0].astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_Nlayer([64, 128], [192, 128]),
        model_dir=model_dir)

    if mode == 'train':
        batch_size = 100
        tensors_to_log = {'probabilities': 'softmax_tensor'}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        train(mnist_classifier, X_train, y_train, batch_size, [logging_hook])

    if mode == 'eval':
        eval(mnist_classifier, X_train, y_train)
        if (train_type == 'full_data'):
            quit()

        eval(mnist_classifier, X_val, y_val)

    if mode == 'predict':
        submission_name = sys.argv[4]
        test = np.loadtxt(
            'data/test.csv', dtype=np.float32, delimiter=',', skiprows=1)

        predictions = predict(mnist_classifier, test)
        np.savetxt(
            submission_name,
            np.column_stack([np.array(range(1, 28001)), predictions]),
            fmt='%d',
            delimiter=',',
            header='ImageId,Label',
            comments='')

import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import train_test_split


# Architecture: N x Conv - Pool - N x Conv - Pool - Dense - Dropout - Softmax

def cnn_Nlayer(features, labels, mode, params):
    filters1 = params['filters1']
    filters2 = params['filters2']
    num_units_dense = params['num_units_dense']
    kernel_size = params['kernel_size']
    reg_strength = params['reg_strength']
    dropout_rate = params['dropout_rate']
    use_batchnorm = params['use_batchnorm']
    learning_rate = params['learning_rate']

    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=reg_strength)
    kernel = [kernel_size, kernel_size]

    net = tf.reshape(features['X'], [-1, 28, 28, 1])

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
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train(model_fn,
          model_dir,
          params,
          train_data,
          val_data,
          batch_size,
          hooks=None):

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=params)

    val_accuracy = 0
    for _ in range(30):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'X': train_data['X']},
            y=train_data['y'],
            batch_size=batch_size,
            num_epochs=5,
            shuffle=True)

        classifier.train(
            input_fn=train_input_fn,
            hooks=hooks)

        def get_eval_input(data):
            return tf.estimator.inputs.numpy_input_fn(
                x={'X': data['X']},
                y=data['y'],
                num_epochs=1,
                shuffle=False)

        train_results = classifier.evaluate(
            input_fn=get_eval_input(train_data))

        val_results = classifier.evaluate(input_fn=get_eval_input(val_data))
        if val_results['accuracy'] < val_accuracy:
            params['learning_rate'] /= 2
            classifier = tf.estimator.Estimator(
                model_fn=model_fn,
                model_dir=model_dir,
                params=params)

            print()
            print(params['learning_rate'])

        val_accuracy = val_results['accuracy']

        print()
        print(train_results)
        print()
        print(val_results)
        print()


def eval(model_fn, model_dir, params, data):
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=params)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': data['X']},
        y=data['y'],
        num_epochs=1,
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print()
    print(eval_results)
    print()


def predict(model_fn, model_dir, params, data):
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=params)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': data['X']}, shuffle=False)

    generator = classifier.predict(
        input_fn=predict_input_fn, predict_keys='classes')

    return np.asarray([i['classes'] for i in list(generator)])


if __name__ == '__main__':
    mode = sys.argv[1]
    validation_set_size = int(sys.argv[2])
    model_dir = sys.argv[3]

    [train_data, val_data] = train_test_split(
        np.loadtxt('data/train.csv', dtype=int, delimiter=',', skiprows=1),
        test_size=validation_set_size,
        shuffle=False)

    _train = {
        'X': train_data[:, 1:785].astype(np.float32),
        'y': train_data[:, 0].astype(np.int32)
    }

    _val = {
        'X': val_data[:, 1:785].astype(np.float32),
        'y': val_data[:, 0].astype(np.int32)
    }

    params = {
        'filters1': [64, 128],
        'filters2': [192, 128],
        'num_units_dense': 1024,
        'kernel_size': 3,
        'reg_strength': 0.0,
        'dropout_rate': 0.4,
        'use_batchnorm': False,
        'learning_rate': 0.001
    }

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_Nlayer,
        model_dir=model_dir,
        params=params)

    if mode == 'train':
        batch_size = 100
        tensors_to_log = {'probabilities': 'softmax_tensor'}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        train(
            cnn_Nlayer,
            model_dir,
            params,
            _train,
            _val,
            batch_size,
            [logging_hook])

    if mode == 'eval':
        eval(cnn_Nlayer, model_dir, params, _train)
        eval(cnn_Nlayer, model_dir, params, _val)

    if mode == 'predict':
        submission_name = sys.argv[4]
        _test = {
            'X': np.loadtxt(
                'data/test.csv', dtype=np.float32, delimiter=',', skiprows=1)
        }

        predictions = predict(cnn_Nlayer, model_dir, params, _test)
        np.savetxt(
            submission_name,
            np.column_stack([np.array(range(1, 28001)), predictions]),
            fmt='%d',
            delimiter=',',
            header='ImageId,Label',
            comments='')

import os
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

flags = tf.app.flags

#
# Snippet for distributed learning
#
try:
    config = os.environ['TF_CONFIG']
    config = json.loads(config)
    task = config['task']['type']
    task_index = config['task']['index']

    local_ip = 'localhost:' + config['cluster'][task][task_index].split(':')[1]
    config['cluster'][task][task_index] = local_ip
    if task == 'chief' or task == 'master':
        config['cluster']['worker'][task_index] = local_ip
    os.environ['TF_CONFIG'] = json.dumps(config)
except:
    pass
#
#
#

flags.DEFINE_string("log_dir",
                    get_logs_path(
                        os.path.expanduser('~/Documents/Scratch/cluster1_experiments/logs')
                    ),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")
FLAGS = flags.FLAGS

def make_model(features, labels, mode, params=None, config=None):
    #
    # Keras model
    #
    model_inp = tf.keras.layers.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(model_inp)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    model_out = tf.keras.layers.Dense(10, activation=None)(x)
    model = tf.keras.models.Model(model_inp, model_out)

    logits = model(features)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.softmax_cross_entropy(labels, logits)
    # Create training op with exponentially decaying learning rate.
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.1,
            global_step=global_step,
            decay_steps=100,
            decay_rate=0.001
        )
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        #
        tf.summary.scalar('optim_learning_rate', learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=tf.argmax(labels, axis=1),
                predictions=predicted_classes
            )
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return None

def make_data(num_epochs=None, shuffle=True, batch_size=32):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    def _input_fn(x, y, num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size):
        data_x = tf.data.Dataset.from_tensor_slices(x)
        data_y = tf.data.Dataset.from_tensor_slices(y)
        data_y = data_y.map(lambda z: tf.one_hot(z, 10, dtype=tf.int32))

        data = tf.data.Dataset.zip((data_x, data_y))
        data = data.batch(batch_size)
        data = data.repeat(num_epochs)
        if shuffle:
            data = data.shuffle(buffer_size=batch_size * 10)

        iterator = data.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels

    train_input = lambda: _input_fn(
        x_train,
        y_train,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    val_input = lambda: _input_fn(
        x_test,
        y_test,
        shuffle=False
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input,
        max_steps=1e6
    )
    val_spec = tf.estimator.EvalSpec(
        input_fn=val_input,
        steps=None,
        start_delay_secs=0,
        throttle_secs=1
    )

    return train_spec, val_spec

def main(_):
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.log_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        keep_checkpoint_max=5,
        log_step_count_steps=50
    )

    classifier = tf.estimator.Estimator(model_fn=make_model, config=config)
    train_spec, val_spec = make_data(num_epochs=None, batch_size=64, shuffle=True)
    tf.estimator.train_and_evaluate(classifier, train_spec, val_spec)

if __name__ == '__main__':
    tf.app.run()
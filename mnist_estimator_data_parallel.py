import os
import json
import time
import tensorflow as tf
import numpy as np
from clusterone import get_data_path, get_logs_path

flags = tf.app.flags

#
# Snippet for distributed learning
#
tf.logging.set_verbosity(tf.logging.INFO)

try:
    task_type = os.environ['JOB_NAME']
    task_index = int(os.environ['TASK_INDEX'])
    ps_hosts = os.environ['PS_HOSTS'].split(',')
    worker_hosts = os.environ['WORKER_HOSTS'].split(',')
    TF_CONFIG = {
        'task': {'type': task_type, 'index': task_index},
        'cluster': {
            'chief': [worker_hosts[0]],
            'worker': worker_hosts,
            'ps': ps_hosts
        },
        'environment': 'cloud'
    }

    local_ip = 'localhost:' + TF_CONFIG['cluster'][task_type][task_index].split(':')[1]
    TF_CONFIG['cluster'][task_type][task_index] = local_ip
    if (task_type in ('chief', 'master')) or (task_type == 'worker' and task_index == 0):
        TF_CONFIG['cluster']['worker'][task_index] = local_ip
        TF_CONFIG['task']['type'] = 'chief'

    os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
except KeyError as ex:
    print(ex)
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None


flags.DEFINE_string("log_dir",
                    get_logs_path(
                        os.path.expanduser('~/Documents/Scratch/cluster1_experiments/logs')
                    ),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")
tf.flags.DEFINE_integer('n_gpus', 1, 'number of gpus to utilize')

FLAGS = flags.FLAGS

def make_model():
    model_inp = tf.keras.layers.Input(shape=(28, 28,), name='input')
    x = tf.keras.layers.Flatten()(model_inp)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    model_out = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
    model = tf.keras.models.Model(model_inp, model_out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)

def get_inputs(x, y, batch_size=64, shuffle=True):
    """
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        """
        """
        x_placeholder = tf.placeholder(tf.float32, x.shape)
        y_placeholder = tf.placeholder(tf.float32, y.shape)
        data_x = tf.data.Dataset.from_tensor_slices(x_placeholder)
        data_y = tf.data.Dataset.from_tensor_slices(y_placeholder)
        data = tf.data.Dataset.zip((data_x, data_y))
        data = data.batch(batch_size).repeat(count=None)

        if shuffle:
            data = data.shuffle(buffer_size=batch_size * 10)

        data = data.prefetch(8)

        iterator = data.make_initializable_iterator()
        next_example, next_label = iterator.get_next()

        # Set runhook to initialize iterator
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
            iterator.initializer,
            feed_dict={x_placeholder: x, y_placeholder: y}
        )

        return next_example, next_label

    return inputs, iterator_initializer_hook


#def make_custom_model(features, labels, mode, params=None, config=None):
#    #
#    # Keras model
#    #
#    model_inp = tf.keras.layers.Input(shape=(28, 28))
#    x = tf.keras.layers.Flatten()(model_inp)
#    x = tf.keras.layers.Dense(128, activation='relu')(x)
#    x = tf.keras.layers.Dense(128, activation='relu')(x)
#    model_out = tf.keras.layers.Dense(10, activation=None)(x)
#    model = tf.keras.models.Model(model_inp, model_out)
#
#    logits = model(features)
#
#    # Compute predictions.
#    predicted_classes = tf.argmax(logits, 1)
#    if mode == tf.estimator.ModeKeys.PREDICT:
#        predictions = {
#            'class': predicted_classes,
#            'prob': logits
#        }
#        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
#
#    # Compute loss.
#    loss = tf.losses.softmax_cross_entropy(labels, logits)
#    # Create training op with exponentially decaying learning rate.
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        global_step = tf.train.get_global_step()
#        learning_rate = tf.train.exponential_decay(
#            learning_rate=0.1,
#            global_step=global_step,
#            decay_steps=100,
#            decay_rate=0.001
#        )
#        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
#        tf.summary.scalar('optim_learning_rate', learning_rate)
#
#        train_op = optimizer.minimize(loss, global_step=global_step)
#        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
#
#    # Compute evaluation metrics.
#    if mode == tf.estimator.ModeKeys.EVAL:
#        eval_metric_ops = {
#            'accuracy': tf.metrics.accuracy(
#                labels=tf.argmax(labels, axis=1),
#                predictions=predicted_classes
#            )
#        }
#        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#    return None

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)

def main(_):
    model = make_model()

    time_hist = TimeHistory()

  #  strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.n_gpus)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.log_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        keep_checkpoint_max=3,
        log_step_count_steps=50
    )

    classifier = tf.keras.estimator.model_to_estimator(
        model,
        config=run_config
    )

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    train_input_fn, train_iter_hook = get_inputs(
        x_train,
        tf.keras.utils.to_categorical(y_train, 10).astype(np.float32)
    )
    test_input_fn, test_iter_hook = get_inputs(
        x_test,
        tf.keras.utils.to_categorical(y_test, 10).astype(np.float32),
        shuffle=False
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=1e4,
        hooks=[train_iter_hook, time_hist]
    )
    val_spec = tf.estimator.EvalSpec(
        input_fn=test_input_fn,
        steps=None,
        start_delay_secs=10,
        throttle_secs=30,
        hooks=[test_iter_hook]
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, val_spec)

    # job metrics
    total_time = sum(time_hist.times)
    print('Total Training Time: {0} on {1} GPUs'.format(total_time, FLAGS.n_gpus))

if __name__ == '__main__':
    tf.app.run()
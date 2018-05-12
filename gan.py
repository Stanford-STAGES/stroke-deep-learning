import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from six.moves import xrange
from tensorflow.contrib import predictor
tfgan = tf.contrib.gan

tfgan = tf.contrib.gan
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

# https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb
# https://github.com/soumith/ganhacks
# https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
# https://www.tensorflow.org/api_docs/python/tf/contrib/gan/gan_loss
# Consider: adding classification penalization and tensor_pool_fn

class GANModel:
    def __init__(self, params):
        self.n_epoch_samples = params['n_epoch_samples']
        self.verbose_shapes = params['verbose_shapes']
        self.logdir = params['logdir']
        self.batch_size = params['batch_size']['config.py']
        self.fs = params['fs']

    def _conditional_generator_fn(self, inputs, weight_decay=2.5e-5, is_training=True):
        """Generator to produce MNIST images.

        Args:
            inputs: A 2-tuple of Tensors (noise, one_hot_labels).
            weight_decay: The value of the l2 weight decay.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population
                statistics.

        Returns:
            A generated image in the range [-1, 1].
        """
        noise, one_hot_labels = inputs

        with framework.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm,
                weights_regularizer=layers.l2_regularizer(weight_decay)), \
             framework.arg_scope([layers.batch_norm], is_training=is_training,
                                 zero_debias_moving_mean=True):
            # net = layers.fully_connected(noise, 1024)
            net = tf.reshape(noise, [8, 4, 2, -1])
            net = layers.conv2d_transpose(net, 256, [3, 2], stride=[2, 1])
            net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
            net = layers.conv2d_transpose(net, 128, [3, 2], stride=[2, 1])
            net = layers.conv2d_transpose(net, 64, [3, 2], stride=[2, 1])
            net = layers.conv2d_transpose(net, 32, [3, 2], stride=[2, 1])
            net = layers.conv2d_transpose(net, 16, [3, 2], stride=[2, 1])
            net = layers.conv2d_transpose(net, 16, [3, 2], stride=[2, 1])
            net = layers.conv2d_transpose(net, 16, [3, 2], stride=[2, 1])
            net = layers.conv2d_transpose(net, 16, [3, 2], stride=[2, 1])
            net = layers.conv2d(net, 1, 16, normalizer_fn=None, activation_fn=tf.tanh)
            net = net[:, 100:100 + self.n_epoch_samples, 0:2, :]
            if self.verbose_shapes: print('Generator output shape: ' + str(net.shape))
            return net

    def _conditional_discriminator_fn(self, input, conditioning, weight_decay=2.5e-5):
        """Conditional discriminator network on MNIST digits.

        Args:
            img: Real or generated MNIST digits. Should be in the range [-1, 1].
            conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
            weight_decay: The L2 weight decay.

        Returns:
            Logits for the probability that the image is real.
        """
        _, one_hot_labels = conditioning
        with framework.arg_scope(
                [layers.conv2d, layers.fully_connected],
                activation_fn=leaky_relu, normalizer_fn=None,
                weights_regularizer=layers.l2_regularizer(weight_decay),
                biases_regularizer=layers.l2_regularizer(weight_decay)):
            net = layers.conv2d(input, 256, [3, 2], stride=2)
            net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
            net = layers.conv2d(net, 128, [3, 2], stride=2)
            net = layers.avg_pool2d(net, kernel_size=[4, 1])
            net = layers.conv2d(net, 64, [3, 2], stride=2)
            net = layers.conv2d(net, 32, [3, 2], stride=2)
            net = layers.avg_pool2d(net, kernel_size=[4, 1])
            net = layers.conv2d(net, 16, [3, 2], stride=2)
            net = layers.flatten(net)
            if self.verbose_shapes: print('Discriminator output shape: ' + str(net.shape))
            net = layers.fully_connected(net, 256, normalizer_fn=layers.batch_norm)
            return layers.linear(net, 1)


    def _visualize_training_generator(self, fig, ax, signals, labels, fs, real_signals=False):
        """Visualize generator outputs during training.
        """
        grp = np.argmax(labels, axis=1)
        grp_names = ['Control\nNorm. EEG amp.', 'Experimental\nNorm. EEG amp.']
        chn_names = ['Channel 1', 'Channel 2']
        colour = ['g', 'r']
        n_examples = 4
        n_channels = signals.shape[2]
        t = np.arange(0, signals.shape[1] / fs, 1 / fs)
        for i in range(n_examples):
            for c in range(n_channels):
                plt.axes(ax[i, c])
                plt.cla()
                if i == 0: plt.title(chn_names[c])
                if real_signals:
                    if c == 0: plt.ylabel('Real: ' + str(i + 1) + '\n' + grp_names[grp[i]])
                else:
                    if c == 0: plt.ylabel('Generated: ' + str(i + 1) + '\n' + grp_names[grp[i]])
                if i == n_examples - 1: plt.xlabel('Time [s]')
                plt.ylim([-1, 1])
                ax[i, c].plot(t, signals[i, :, c], colour[grp[i]])
                ax[i, c].spines['top'].set_visible(False)
                ax[i, c].spines['right'].set_visible(False)
                if i != n_examples - 1: ax[i, c].spines['bottom'].set_visible(False)
                if i != n_examples - 1: ax[i, c].get_xaxis().set_ticks([])
                if c != 0: ax[i, c].spines['left'].set_visible(False)
                if c != 0: ax[i, c].get_yaxis().set_ticks([])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)

    def __call__(self, real_signals_generator, gan_train = False):
        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def __init__(self, outer_self, fs):
                self.fs = fs
                self.outer_self = outer_self

            def begin(self):
                self._step = -1
                self._start_time = time.time()
                self.fig, self.ax = plt.subplots(4, 2)
                self.fig_real, self.ax_real = plt.subplots(4, 2)

            def before_run(self, run_context):
                self._step += 1
                gen_loss_np = gan_loss.generator_loss
                dis_loss_np = gan_loss.discriminator_loss
                return tf.train.SessionRunArgs(
                    [gen_loss_np, dis_loss_np, eval_signals, one_hot_labels, real_images, labels])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % 200 == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    returns = run_values.results
                    examples_per_sec = 10 * 8 / duration
                    sec_per_batch = float(duration / 10)
                    predict_fn = predictor.from_saved_model(export_dir)
                    predictions = predict_fn(
                        {"input": returns[2]})
                    y_pred = np.transpose(predictions['classes'][:, 0])
                    y_est = np.argmax(returns[3], axis=1)
                    cur_acc = sum(y_pred == y_est) / len(y_est) * 100
                    self.outer_self._visualize_training_generator(self.fig, self.ax, returns[2], returns[3], fs = self.fs)
                    self.outer_self._visualize_training_generator(self.fig_real, self.ax_real, returns[4], returns[5], fs = self.fs, real_signals=True)

                    format_str = ('%s: step %d, loss = (g: %.2f, d: %.2f) (%.1f examples/sec; %.3f '
                                  'sec/batch), accuracy: %.2f')
                    print(format_str % (datetime.now(), self._step, returns[0], returns[1],
                                        examples_per_sec, sec_per_batch, cur_acc))

        exports = os.listdir(self.logdir)
        exports = [int(e) for e in exports if e.isdigit()]
        export_dir = os.path.abspath(self.logdir + str(exports[np.argmax(exports)]))
        gan_model_dir = self.logdir + 'gan/'

        signal, labels = real_signals_generator
        labels = tf.reshape(labels, [8, 2])
        real_images = tf.divide(signal, tf.reduce_max(tf.abs(signal)))
        real_images = tf.expand_dims(real_images, [3])
        real_images = tf.reshape(real_images, shape=[8, self.n_epoch_samples, 2, 1])

        '''GANModel Tuple'''
        noise_dims = 2048
        conditional_gan_model = tfgan.gan_model(
            generator_fn=self._conditional_generator_fn,
            discriminator_fn=self._conditional_discriminator_fn,
            real_data=real_images,
            generator_inputs=(tf.random_normal([self.batch_size, noise_dims]),
                              labels))
        '''Losses'''
        with tf.name_scope('GAN_loss'):
            gan_loss = tfgan.gan_loss(
                conditional_gan_model,
                generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
                discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
                gradient_penalty_weight=1.0)

        '''Train Ops'''
        with tf.name_scope('GAN_train'):
            generator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.98)
            discriminator_optimizer = tf.train.AdamOptimizer(0.00001, beta1=0.5, beta2=0.98)
            gan_train_ops = tfgan.gan_train_ops(
                conditional_gan_model,
                gan_loss,
                generator_optimizer,
                discriminator_optimizer)

        '''Eval'''
        # Set up class-conditional visualization. We feed class labels to the generator
        # so that the the first column is `0`, the second column is `1`, etc.
        n_classes = 2
        signals_to_eval = 8
        random_noise = tf.random_normal([signals_to_eval, noise_dims])
        one_hot_labels = tf.one_hot(
            [i for _ in xrange(signals_to_eval // n_classes) for i in xrange(n_classes)], depth=n_classes)
        with tf.variable_scope('Generator', reuse=True):
            eval_signals = conditional_gan_model.generator_fn(
                (random_noise, one_hot_labels), is_training=False)
            eval_signals = eval_signals[:, :, :, 0]

        '''Train'''
        global_step = tf.train.get_or_create_global_step()
        train_step_fn = tfgan.get_sequential_train_steps()
        predict_fn = predictor.from_saved_model(export_dir)

        if gan_train:
            tfgan.gan_train(
                gan_train_ops,
                gan_model_dir,
                hooks=[_LoggerHook(self, self.fs)])

        # ADAM variables are causing the checkpoint reload to choke, so omit them when
        # doing variable remapping.
        var_dict = {x.op.name: x for x in
                    tf.contrib.framework.get_variables('Generator/')
                    if 'Adam' not in x.name}
        ckpt = tf.train.get_checkpoint_state(gan_model_dir)
        tf.contrib.framework.init_from_checkpoint(
            ckpt.model_checkpoint_path, var_dict)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x, y = sess.run([eval_signals, one_hot_labels])
        fig, ax = plt.subplots(4, 2)
        self._visualize_training_generator(fig, ax, x, y)

        plt.show()


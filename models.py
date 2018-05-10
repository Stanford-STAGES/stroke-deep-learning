import tensorflow as tf
from datahandler import DataHandler

class SimpleCRNN:
    def __init__(self, name, params):
        self.name = name
        self.temporal_kernel_size = params['temporal_kernel_size']
        self.n_channels = params['n_channels']
        self.n_epoch_samples = params['n_epoch_samples']
        self.dropout_pct = params['dropout_pct']
        self.pooling_size = params['pooling_size']
        self.n_units_dense = params['n_units_dense']
        self.n_classes = params['n_classes']
        self.batch_norm = params['batch_norm']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.kernel_regularizer = params['kernel_regularizer']
        self.kernel_initializer = params['kernel_initializer']
        self.optimizer = params['optimizer']
        self.learning_rate = params['learning_rate']
        self.n_time_steps = params['time_steps']
        self.n_filters = params['n_filters']
        self.n_sub_epoch_samples = self.n_epoch_samples//self.n_time_steps # todo consider case where these to does not yield integer
        self.rnn_size = params['rnn_size']
        self.regularization = params['regularization']
        self.verbose_shapes = params['verbose_shapes']
        self.pool_stride = params['pool_stride']
        self.n_layers = params['n_layers']
        self.rnn_layer = params['rnn_layer']
        self.dense_layer = params['dense_layer']
        self.one_output_per_epoch = params['one_output_per_epoch']
        self.training_hook_n_iter = params['training_hook_n_iter']

    def __conv_layer(self, input, n_filters):
        conv = tf.layers.conv3d(inputs=input,
                                 filters=n_filters,
                                 kernel_size=[1, 1, self.temporal_kernel_size],
                                 data_format='channels_first',
                                 use_bias=False,  # DUE TO ERROR WHEN NOT HAVING
                                 padding='valid',
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer)
        return conv

    def __pool_layer(self, input):
        pool = tf.layers.max_pooling3d(inputs=input,
                                        data_format='channels_first',
                                        pool_size=[1, 1, self.pooling_size],
                                        padding='valid',
                                        strides=(1, 1, self.pool_stride))
        return pool

    def __conv_block(self, input, n_filters, scope='test'):
        conv1 = self.__conv_layer(input, n_filters)
        bn1 = tf.layers.batch_normalization(inputs=conv1, axis=1, fused=True) if self.batch_norm else conv1
        do1 = tf.layers.dropout(inputs=bn1, noise_shape=(1, n_filters, 1, 1, 1), rate=self.dropout_pct,
                                training=self.training) if self.dropout else bn1
        conv2 = self.__conv_layer(do1, n_filters)
        bn2 = tf.layers.batch_normalization(inputs=conv2, axis=1, fused=True) if self.batch_norm else conv2
        do2 = tf.layers.dropout(inputs=bn2, noise_shape=(1, n_filters, 1, 1, 1), rate=self.dropout_pct,
                               training=self.training) if self.dropout else bn2

        pool = self.__pool_layer(do2)
        if self.verbose_shapes:
            print('{}/conv1: {}'.format(scope,conv1.shape))
            print('{}/conv2: {}'.format(scope,conv2.shape))
            print('{}/pool: {}'.format(scope,pool.shape))
        return pool

    def __network(self, input, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            with tf.name_scope('conv_net'):
                #conv_out = tf.contrib.layers.repeat(input, self.n_layers, self.__conv_block, scope='conv')
                n = self.n_sub_epoch_samples
                stack_args = []
                for i in range(self.n_layers):
                    n = (n - 2*self.temporal_kernel_size) // self.pool_stride
                    stack_args.append(self.n_filters//(2**i))
                conv_out = tf.contrib.layers.stack(input, self.__conv_block, stack_args, scope='conv_block')

            with tf.name_scope('rnn'):
                if self.rnn_layer:
                    rnn_input = tf.transpose(conv_out[:, :, :, 0, :], [0, 2, 1, 3])
                    rnn_input = tf.reshape(rnn_input,
                                           [-1, self.n_time_steps, stack_args[-1] * (n+1)])
                    if self.verbose_shapes: print('rnn_input: {}'.format(rnn_input.shape))
                    lstm_fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.rnn_size)
                    lstm_bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.rnn_size)
                    rnn_states, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell,
                                                                    rnn_input,
                                                                    dtype=tf.float32)
                    rnn_output = tf.concat(rnn_states, axis=2)
                    if self.verbose_shapes: print('rnn_output: {}'.format(rnn_output.shape))
                    dense_input = rnn_output
                else:
                    dense_input = tf.transpose(conv_out, [0, 2, 1, 3, 4])
                    #dense_input = tf.reduce_max(dense_input, axis=4)
                    #dense_input = tf.reduce_mean(dense_input, axis=4)
                    #print(dense_input.shape)
                    dense_input = tf.reshape(dense_input, [-1, self.n_time_steps, 5*16])
                    #dense_input = tf.reshape(dense_input, [-1, self.n_time_steps, self.n_filters * n])
                if self.verbose_shapes: print('dense_input: {}'.format(dense_input.shape))

            with tf.name_scope('output_layer'):
                if self.dense_layer:
                    output = tf.layers.dense(inputs=dense_input,
                                                  units=self.n_units_dense,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  activation=self.activation)
                else:
                    output = dense_input
            if self.verbose_shapes: print('network output: {}'.format(output.shape))
        return output

    def __output_layer(self, input, reuse = False):
        with tf.variable_scope(self.name+'_output') as scope:
            if reuse:
                scope.reuse_variables()
            logits = tf.layers.dense(inputs=input,
                                     units=self.n_classes)
            return logits

    def __sensitivity_analysis(self,x,y):
        with tf.name_scope('sensitivity_analysis'):
            g = [xx * tf.gradients(yy, xx)[0] for (yy, xx) in zip(y, x)]
            # g = [tf.gradients(yy, xx)[0] for (yy,xx) in zip(yy,xx)]
            # g = [tf.square(tf.gradients(yy, xx)[0]) for (yy,xx) in zip(y,x)]
            g = tf.convert_to_tensor(g)
        return tf.transpose(g, [1, 2, 0, 3, 4])

    def __call__(self, features, labels, mode, reuse=False):
        self.training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        with tf.name_scope('classifer_input'):
            inputs = tf.identity(features, name="input")
            bs = tf.shape(features)[0]
            channels_first = tf.transpose(inputs, [0,2,1]) # channels first
            subepoched = tf.reshape(channels_first, [bs, self.n_channels, self.n_time_steps, 1, self.n_sub_epoch_samples])
            xlist = [subepoched[:,:,time,:,:] for time in range(self.n_time_steps)]
            X = tf.convert_to_tensor(xlist)
            X = tf.transpose(X, [1,2,0,3,4])
            if self.verbose_shapes: print('X: {}'.format(X.shape))

        extracted_features = self.__network(input=X, reuse=reuse)
        if self.one_output_per_epoch:
            extracted_features = tf.contrib.layers.flatten(extracted_features)

        with tf.name_scope('classifier_output'):
            logits = self.__output_layer(input = extracted_features, reuse=reuse)

            if self.verbose_shapes: print('logits: {}'.format(logits.shape))
            self.verbose_shapes = False

            if self.one_output_per_epoch:
                classes = tf.argmax(input=logits, axis=1)
            else:
                classes = tf.argmax(input=logits, axis=2)
            probabilities = tf.nn.softmax(logits, name="softmax_tensor")

            predictions = {}
            predictions["classes"] = classes
            predictions["probabilities"] = probabilities
            predictions["features"] = extracted_features

            if self.one_output_per_epoch:
                l = tf.expand_dims(logits, 1)

                ylist = [l[:, time, 0] for time in range(1)]
                control_sensitivity = self.__sensitivity_analysis(xlist, ylist)
                ylist = [l[:, time, 1] for time in range(1)]
                experimental_sensitivity = self.__sensitivity_analysis(xlist, ylist)

                predictions["experimental_sensitivity"] = experimental_sensitivity
                predictions["control_sensitivity"] = control_sensitivity
            else:
                ylist = [logits[:, time, 0] for time in range(self.n_time_steps)]
                control_sensitivity = self.__sensitivity_analysis(xlist, ylist)

                ylist = [logits[:, time, 1] for time in range(self.n_time_steps)]
                experimental_sensitivity = self.__sensitivity_analysis(xlist, ylist)
                predictions["experimental_sensitivity"] = experimental_sensitivity
                predictions["control_sensitivity"] = control_sensitivity

            export_outputs = {'out': tf.estimator.export.PredictOutput(predictions)}

        '''
        with tf.name_scope('activation_maximization'):
            X_p = tf.get_variable('X_p',
                shape = [self.n_classes, self.n_channels, self.n_time_steps, 1, self.n_sub_epoch_samples],
                initializer=tf.random_normal_initializer(0))
            if self.verbose_shapes: print('X_p: {}'.format(X_p.shape))
            X_mean = tf.placeholder(tf.float32,
                shape = [self.n_classes, self.n_channels, self.n_time_steps, 1, self.n_sub_epoch_samples],
                                    name='X_mean')
            Spectra = tf.placeholder(tf.float32,
                shape = [self.n_classes, self.n_channels, self.n_sub_epoch_samples//2+1],
                                    name='Spectra')

            Spectra_p = tf.squeeze( tf.square( tf.abs( tf.spectral.rfft(X_p ) ) ) )

            lambda_p = tf.placeholder_with_default(0.00001, shape=[], name='lambda_p')
            theta_p = tf.placeholder_with_default(0.00001, shape=[], name='theta_p')
            ypsilon_p = tf.placeholder_with_default(0.00001, shape=[], name='ypsilon_p')

            extracted_features_p = self.__network(input=X_p, reuse=True)
            logits_p = self.__output_layer(input = extracted_features_p, reuse=True)

            y_p = tf.one_hot(tf.cast(tf.lin_space(0., self.n_classes-1, self.n_classes), tf.int32), depth=self.n_classes)

            cost_p = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_p,
                                                        labels=y_p))
            cost_p += lambda_p * tf.reduce_sum(tf.abs(X_p))
            cost_p += theta_p * tf.nn.l2_loss(X_p - X_mean)
            cost_p += ypsilon_p * tf.nn.l2_loss(Spectra_p - Spectra)

            lr_p = tf.placeholder_with_default(0.01, shape=[], name='lr_p')
            opt_p = self.optimizer(learning_rate=lr_p).minimize(cost_p,var_list=[X_p])

            tf.add_to_collection('prototype', X_p)
            tf.add_to_collection('prototype', y_p)
            tf.add_to_collection('prototype', logits_p)
            tf.add_to_collection('prototype', cost_p)
            tf.add_to_collection('prototype', opt_p)
            tf.add_to_collection('prototype', lr_p)
            tf.add_to_collection('prototype', lambda_p)
            tf.add_to_collection('prototype', theta_p)
            tf.add_to_collection('prototype', X_mean)
            tf.add_to_collection('prototype', Spectra)
            tf.add_to_collection('prototype', ypsilon_p)
        '''

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)

        with tf.name_scope('classifier_evaluation'):
            if not self.one_output_per_epoch:
                y = tf.reshape(tf.tile(labels, (1, self.n_time_steps)), (-1, 2))
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(logits,[-1,2]), labels=y))
                true_classes = tf.argmax(y, 1)
            else:
                y = labels
                true_classes = tf.argmax(y,1)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

            l2 = self.regularization * sum(
                tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
            )
            loss += l2

            #est_classes = tf.reshape(predictions['classes'],[-1])
            est_classes = predictions['classes']
	    correct_prediction = tf.equal(est_classes, true_classes)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            cost_summary = tf.summary.scalar('Cost', loss)
            accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
            probabilities_summary = tf.summary.histogram('Probabilities', predictions['probabilities'])
            summary = tf.summary.merge_all()

            # Allow logging
            tensors_to_log = {'step': tf.train.get_global_step(),
                              'loss': loss,
                              'accuracy': accuracy,
                              'summary': summary}

            def formatter(d):
                return 'Mode: {}, Step: {:04}, loss: {:.4f}, accuracy: {:.2f}'.format(mode,
                                                                                      d['step'],
                                                                                      d['loss'],
                                                                                      d['accuracy'])

            training_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=self.training_hook_n_iter, formatter=formatter)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            opt = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            train_op = opt
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks= [training_hook])
        # Add evaluation metrics (for EVAL mode)
        with tf.name_scope('evaluation'):
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=true_classes, predictions=est_classes),
                "fp": tf.metrics.false_positives(labels=true_classes, predictions=est_classes),
                "fn": tf.metrics.false_negatives(labels=true_classes, predictions=est_classes),
                "tp": tf.metrics.true_positives(labels=true_classes, predictions=est_classes),
                "tn": tf.metrics.true_negatives(labels=true_classes, predictions=est_classes),
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=None)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

def input_fn(mode,params):
    if mode == 'train' or mode == 'val':
        gen = DataHandler(mode).generate_batches
    elif mode =='test_sequence':
        gen = DataHandler('test').generate_sequence
    elif mode =='test_batch':
        gen = DataHandler('test').generate_batches
    ds = tf.data.Dataset.from_generator(gen,
            output_types=(tf.float32, tf.int32))
    ds = ds.prefetch(buffer_size=params['buffer_size'])
    # todo implement reading and indexing going on in generator as map and use map_and_batch
    return ds.make_one_shot_iterator().get_next()


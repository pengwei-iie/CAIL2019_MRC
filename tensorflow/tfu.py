# coding:utf-8
import tensorflow as tf
import tensorflow.contrib as tfctr
import numpy as np
import re

INF = 1e20


class OptimizerWrapper(object):

    def __init__(self, optimizer, grad_clip=None, decay=None, exclude=None):
        self._opt = optimizer
        self._grad_clip = grad_clip or 5
        self._decay_param = 7e-7 if decay is None else decay
        self._exclude = set(exclude) if exclude is not None else None

    def compute_gradients(self, loss):
        grads = self._opt.compute_gradients(loss)
        gradients, variables = zip(*grads)
        if self._grad_clip > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, self._grad_clip)

        return zip(gradients, variables)

    def _get_decay_var_list(self):
        if self._exclude is None:
            var_list = tf.trainable_variables()
        else:
            var_list = []
            for var in tf.trainable_variables():
                is_in = True
                for kx in self._exclude:
                    if kx in var.name:
                        is_in = False
                        break
                if is_in:
                    var_list.append(var)
        return var_list

    def apply_gradients(self, grads_and_vars, global_step):
        train_op = self._opt.apply_gradients(grads_and_vars, global_step=global_step)
        var_list = self._get_decay_var_list()
        l2_loss = tf.add_n([tf.nn.l2_loss(ix) for ix in var_list]) * self._decay_param / 0.5
        self._l2_loss = l2_loss
        decay_opt = tf.train.GradientDescentOptimizer(1)
        decay_op = decay_opt.minimize(l2_loss, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(train_op, decay_op, update_ops)
        return train_op

    def minimize(self, loss, global_step):
        a = self.compute_gradients(loss)
        return self.apply_gradients(a, global_step)

    @property
    def l2_loss(self):
        return self._l2_loss


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """
    A basic Adam optimizer that includes "correct" L2 weight decay.
    Copy from the [bert](https://github.com/google-research/bert?from=timeline&isappinstalled=0)
    """

    def __init__(self, learning_rate, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, exclude_from_weight_decay=None, name="AdamWeightDecayOptimizer"):
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(name=param_name + "/adam_m", shape=param.shape.as_list(),
                                dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer())
            v = tf.get_variable(name=param_name + "/adam_v", shape=param.shape.as_list(),
                                dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class CudnnGRU(object):

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self._scope = scope
        with tf.variable_scope(self._scope or 'gru'):
            for layer in range(num_layers):
                input_size_ = input_size if layer == 0 else 2 * num_units
                gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
                gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
                init_fw = tf.tile(tf.get_variable('init_fw_%d' % layer, dtype=tf.float32, shape=[1, 1, num_units],
                                                  initializer=tf.zeros_initializer), [1, batch_size, 1])
                init_bw = tf.tile(tf.get_variable('init_bw_%d' % layer, dtype=tf.float32, shape=[1, 1, num_units],
                                                  initializer=tf.zeros_initializer), [1, batch_size, 1])
                mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                  keep_prob=keep_prob, is_train=is_train)
                mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                  keep_prob=keep_prob, is_train=is_train)
                self.grus.append((gru_fw, gru_bw,))
                self.inits.append((init_fw, init_bw,))
                self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        output_states = []
        with tf.variable_scope(self._scope or 'gru'):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = gru_fw(outputs[-1] * mask_fw, initial_state=(init_fw,))
                    out_tt = tf.reverse_sequence(out_fw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw,))
                    out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
                output_states.append(tf.concat([out_tt[0], out_bw[0]], axis=1))
            if concat_layers:
                res = tf.concat(outputs[1:], axis=2)
                output_states = tf.concat(output_states, axis=1)
            else:
                res = outputs[-1]
                output_states = output_states[-1]
            res = tf.transpose(res, [1, 0, 2])
        return res, output_states


class NativeGRU(object):

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        with tf.variable_scope(self.scope or 'native_gru'):
            for layer in range(num_layers):
                input_size_ = input_size if layer == 0 else 2 * num_units
                gru_fw = tfctr.rnn.GRUBlockCellV2(num_units)
                gru_bw = tfctr.rnn.GRUBlockCellV2(num_units)
                init_fw = tf.get_variable('init_fw_%d' % layer, dtype=tf.float32, shape=[1, num_units],
                                          initializer=tf.zeros_initializer)
                init_bw = tf.get_variable('init_bw_%d' % layer, dtype=tf.float32, shape=[1, num_units],
                                          initializer=tf.zeros_initializer)
                init_fw = tf.tile(init_fw, [batch_size, 1])
                init_bw = tf.tile(init_bw, [batch_size, 1])
                mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                  keep_prob=keep_prob, is_train=is_train)
                mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                  keep_prob=keep_prob, is_train=is_train)
                self.grus.append((gru_fw, gru_bw,))
                self.inits.append((init_fw, init_bw,))
                self.dropout_mask.append((mask_fw, mask_bw,))

    def __call__(self, inputs, seq_len, concat_layers=True):
        outputs = [inputs]
        output_states = []
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, state_fw = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, state_bw = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
                output_states.append(tf.concat([state_fw, state_bw], axis=1))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
            output_states = tf.concat(output_states, axis=1)
        else:
            res = outputs[-1]
            output_states = output_states[-1]
        return res, output_states


class SimpleCNN(object):

    def __init__(self, num_filters=10, filter_size=(2, 3, 4, 5), keep_prob=1.0, is_train=None, scope=None,
                 activation=None, bias=True, mode='SAME'):
        self._is_bias = bias
        self._mode = mode
        self._filter_sizes = filter_size
        self._kprob = keep_prob
        self._is_train = is_train
        self._scope = scope or 'simple_cnn'
        self._activation = activation
        if isinstance(num_filters, int):
            self._num_filter = [num_filters] * len(filter_size)
        elif isinstance(num_filters, (tuple, list, np.ndarray)):
            self._num_filter = num_filters
        assert len(self._num_filter) == len(self._filter_sizes)

    def __call__(self, inputs, concat_layers=True, reuse=False):
        outputs = []
        with tf.variable_scope(self._scope, reuse=reuse):
            for fil_size, num_fil in zip(self._filter_sizes, self._num_filter):
                masked_inputs = dropout(inputs, self._kprob, self._is_train)
                res = convolution(masked_inputs, num_fil, kernel_size=fil_size, scope='conv_%d' % fil_size,
                                  bias=self._is_bias, mode=self._mode)
                if self._activation is not None:
                    res = self._activation(res)
                outputs.append(res)
            outputs = tf.concat(outputs[1:], axis=-1) if concat_layers else outputs[-1]
        return outputs


class _Transformer(object):

    def __init__(self, hidden, layers, heads, ffd_hidden, ffd_fn=None, keep_prob=1.0,
                 is_train=None, scope='transformer'):
        self._hidden = hidden
        self._layer = layers
        self._heads = heads
        self._ffd_hidden = ffd_hidden
        self._ffd_fn = ffd_fn or gelu
        self._kprob = keep_prob
        self._is_train = is_train
        self._scope = scope

        if hidden % heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden, heads))
        self._att_hidden = hidden // heads

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TransformerEncoder(_Transformer):

    def __init__(self, hidden, layers, heads, ffd_hidden, ffd_fn=None, keep_prob=1.0,
                 is_train=None, scope='transformer_encoder'):
        super(TransformerEncoder, self).__init__(hidden, layers, heads, ffd_hidden, ffd_fn, keep_prob,
                                                 is_train, scope)

    def __call__(self, inputs, mask, all_layer=False, reuse=False, **kwargs):
        with tf.variable_scope(self._scope, reuse=reuse):
            hidden = inputs.shape.as_list()[-1]
            if self._hidden != hidden:
                # raise ValueError("The width of the input tensor (%d) != hidden size (%d) due to the residuals" %
                #                  (hidden, self._hidden))
                inputs = self._ffd_fn(dense(inputs, self._hidden, use_bias=False, scope='input_proj'))

            outputs = [inputs]
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 2)  # [batch, 1, 1, m_length]
            for layer in range(self._layer):
                with tf.variable_scope('layer_%d' % layer):
                    now_out = self._layer_call(outputs[-1], mask)
                    outputs.append(now_out)
            return outputs[1:] if all_layer else outputs[-1]

    def _layer_call(self, inputs, mask):
        att_res = multi_head_attention(inputs, inputs, self._heads, self._att_hidden, is_train=self._is_train,
                                       mem_mask=mask, keep_prob=self._kprob, scope='self_attention')
        # att_res = dense(att_res, self._hidden, scope='compress')
        att_res = dropout(att_res, self._kprob, self._is_train)
        att_res = layer_norm(att_res + inputs, 'att')

        res = self._ffd_fn(dense(att_res, self._ffd_hidden, scope='ffd_w0'))
        res = dense(res, self._hidden, scope='ffd_w1')
        res = dropout(res, self._kprob, self._is_train)
        res = layer_norm(res + att_res, scope='ffd')
        return res


class TransformerDecoder(_Transformer):

    def __init__(self, hidden, layers, heads, ffd_hidden, ffd_fn=None, keep_prob=1.0,
                 is_train=None, scope='transformer_decoder'):
        super(TransformerDecoder, self).__init__(hidden, layers, heads, ffd_hidden, ffd_fn, keep_prob,
                                                 is_train, scope)
        # for decoder step
        self._step_memory = None
        self._step_mem_mask = None
        self._batch = None
        self._att_prob = None

    @property
    def attention_prob(self):
        return self._att_prob

    @property
    def before_input_shape(self):
        before_shape = {'layer_{}'.format(ix): tf.TensorShape([None, None, None])
                        for ix in range(-1, self._layer)}
        before_shape['is_start'] = tf.TensorShape([])
        return before_shape

    @property
    def before_init(self):
        before = {'layer_{}'.format(ix): tf.zeros((self._batch, 1, self._hidden), dtype=tf.float32)
                  for ix in range(-1, self._layer)}
        before['is_start'] = tf.constant(True)
        return before

    def _train_self_att_block(self, inputs, input_mask):
        with tf.variable_scope('self_att'):
            att_res = multi_head_attention(inputs, inputs, self._heads, self._att_hidden, is_train=self._is_train,
                                           mem_mask=input_mask, keep_prob=self._kprob, scope='self_attention')
            # att_res = dense(att_res, self._hidden, scope='compress')
            att_res = dropout(att_res, self._kprob, self._is_train)
            att_res = layer_norm(att_res + inputs, 'att')
            return att_res

    def _train_memory_att_block(self, att_res, memory, mem_mask):
        with tf.variable_scope('mem_att'):
            enc_att, prob = multi_head_attention(
                att_res, memory, self._heads, self._att_hidden, is_train=self._is_train, mem_mask=mem_mask,
                keep_prob=self._kprob, scope='attention', is_prob=True)
            self._att_prob = prob
            # enc_att = dense(enc_att, self._hidden, scope='compress')
            enc_att = dropout(enc_att, self._kprob, self._is_train)
            enc_att = layer_norm(enc_att + att_res, 'enc_att')
            return enc_att

    def _train_ffd_block(self, enc_att):
        with tf.variable_scope('ffd'):
            res = self._ffd_fn(dense(enc_att, self._ffd_hidden, scope='ffd_w0'))
            res = dense(res, self._hidden, scope='ffd_w1')
            res = dropout(res, self._kprob, self._is_train)
            res = layer_norm(res + enc_att, scope='ffd')
            return res

    def _train_layer_call(self, inputs, input_mask, memory, mem_mask):
        att_res = self._train_self_att_block(inputs, input_mask)
        enc_att = self._train_memory_att_block(att_res, memory, mem_mask)
        res = self._train_ffd_block(enc_att)
        return res

    def _train_input_mask(self, input_mask):
        # input_mask: [batch, length]
        # need: [batch, head, length, length]
        batch = tf.shape(input_mask)[0]
        length = tf.shape(input_mask)[1]
        lower_triangle = tf.matrix_band_part(tf.ones([length, length], dtype=tf.int32), -1, 0)
        lower_triangle = tf.reshape(lower_triangle, [1, 1, length, length])
        lower_triangle = tf.tile(lower_triangle, [batch, self._heads, 1, 1])

        input_mask = tf.expand_dims(tf.expand_dims(input_mask, 1), 2)
        input_mask = tf.tile(input_mask, [1, self._heads, length, 1])
        return input_mask * lower_triangle

    def assign_memory(self, memory, mask):
        """联合模型中，word进行decode时的batch与训练时不同（decode出的pattern长度不同）"""
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            # memory_hidden = memory.shape.as_list()[-1]
            # if self._hidden != memory_hidden:
            #     memory = self._ffd_fn(dense(memory, self._hidden, use_bias=False, scope='input_proj_mem'))
            self._step_memory = memory

            memory_mask = tf.expand_dims(tf.expand_dims(mask, 1), 2)  # [batch, 1, 1, length]
            memory_mask = tf.tile(memory_mask, [1, self._heads, 1, 1])
            self._step_mem_mask = memory_mask
            self._batch = tf.shape(self._step_memory)[0]

    def __call__(self, inputs, memory, input_mask, memory_mask, all_layer=False, reuse=False):
        self._batch = tf.shape(inputs)[0]
        with tf.variable_scope(self._scope, reuse=reuse):
            hidden = inputs.shape.as_list()[-1]
            if self._hidden != hidden:
                inputs = self._ffd_fn(dense(inputs, self._hidden, use_bias=False, scope='input_proj'))
            # if self._hidden != memory_hidden:
            #     memory = self._ffd_fn(dense(memory, self._hidden, use_bias=False, scope='input_proj_mem'))

            input_mask = tf.to_int32(input_mask)
            input_mask = self._train_input_mask(input_mask)
            self.assign_memory(memory, memory_mask)

            outputs = [inputs]

            for layer in range(self._layer):
                with tf.variable_scope('layer_%d' % layer):
                    now_out = self._train_layer_call(outputs[-1], input_mask, self._step_memory, self._step_mem_mask)
                    outputs.append(now_out)
            return outputs[1:] if all_layer else outputs[-1]

    def _step_self_att_block(self, inputs, input_before):
        with tf.variable_scope('self_att'):
            att_res = multi_head_attention(inputs, input_before, self._heads, self._att_hidden, is_train=self._is_train,
                                           mem_mask=None, keep_prob=self._kprob, scope='self_attention')
            # att_res = dense(att_res, self._hidden, scope='compress')
            att_res = dropout(att_res, self._kprob, self._is_train)
            att_res = layer_norm(att_res + inputs, 'att')
            return att_res

    def _step_memory_att_block(self, att_res, memory, mem_mask, **kwargs):
        with tf.variable_scope('mem_att'):
            enc_att, prob = multi_head_attention(
                att_res, memory, self._heads, self._att_hidden, is_train=self._is_train,
                mem_mask=mem_mask, keep_prob=self._kprob, scope='attention', is_prob=True)
            self._att_prob = prob
            # enc_att = dense(enc_att, self._hidden, scope='compress')
            enc_att = dropout(enc_att, self._kprob, self._is_train)
            enc_att = layer_norm(enc_att + att_res, 'enc_att')
            return enc_att

    def _step_ffd_block(self, enc_att):
        with tf.variable_scope('ffd'):
            res = self._ffd_fn(dense(enc_att, self._ffd_hidden, scope='ffd_w0'))
            res = dense(res, self._hidden, scope='ffd_w1')
            res = dropout(res, self._kprob, self._is_train)
            res = layer_norm(res + enc_att, scope='ffd')
            return res

    def _layer_step(self, inputs, memory, mem_mask, input_before, **kwargs):
        att_res = self._step_self_att_block(inputs, input_before)
        enc_att = self._step_memory_att_block(att_res, memory, mem_mask, **kwargs)
        res = self._step_ffd_block(enc_att)
        return res

    def step(self, single_input, before_input, all_layer=False, reuse=True):
        """
        before_input 是一个字典，需要记录每一层的之前的输出
        """
        with tf.variable_scope(self._scope, reuse=reuse):
            hidden = single_input.shape.as_list()[-1]
            if self._hidden != hidden:
                single_input = self._ffd_fn(dense(single_input, self._hidden, use_bias=False,
                                                  scope='input_proj', reuse=reuse))

            before_input['layer_-1'] = tf.cond(
                before_input['is_start'], lambda: single_input,
                lambda: tf.concat([before_input['layer_-1'], single_input], axis=1)
            )

            outputs = [single_input]
            for layer in range(self._layer):
                now_key = 'layer_{}'.format(layer)
                pre_key = 'layer_{}'.format(layer - 1)
                with tf.variable_scope('layer_%d' % layer):
                    now_out = self._layer_step(outputs[-1], self._step_memory, self._step_mem_mask,
                                               before_input[pre_key])
                    outputs.append(now_out)
                    before_input[now_key] = tf.cond(
                        before_input['is_start'], lambda: now_out,
                        lambda: tf.concat([before_input[now_key], now_out], axis=1)
                    )

            before_input['is_start'] = tf.constant(False)
            return outputs[1:] if all_layer else outputs[-1], before_input


class TransformerDecoderCoverage(TransformerDecoder):

    def __init__(self, hidden, layers, heads, ffd_hidden, ffd_fn=None, keep_prob=1.0,
                 is_train=None, scope='transformer_decoder'):
        super(TransformerDecoderCoverage, self).__init__(hidden, layers, heads, ffd_hidden, ffd_fn, keep_prob,
                                                         is_train, scope)
        self._collect_loss = tf.constant(0.0, dtype=tf.float32)

    @property
    def loss(self):
        return self._collect_loss

    @property
    def before_input_shape(self):
        before_shape = {'layer_{}_output'.format(ix): tf.TensorShape([None, None, None])
                        for ix in range(-1, self._layer)}
        before_shape.update(
            {'layer_{}_coverage'.format(ix): tf.TensorShape([None, self._heads, None, None])
             for ix in range(0, self._layer)}
        )
        before_shape['is_start'] = tf.TensorShape([])
        return before_shape

    @property
    def before_init(self):
        before = {'layer_{}_output'.format(ix): tf.zeros((self._batch, 1, self._hidden), dtype=tf.float32)
                  for ix in range(-1, self._layer)}
        before.update({
            'layer_{}_coverage'.format(ix): tf.zeros((self._batch, self._heads, 1, 1), dtype=tf.float32)
            for ix in range(0, self._layer)
        })
        before['is_start'] = tf.constant(True)
        return before

    def _train_memory_att_block(self, att_res, memory, mem_mask):
        with tf.variable_scope('mem_att'):
            enc_att, _, loss = multi_head_attention_with_coverage(
                att_res, memory, self._heads, self._att_hidden, is_train=self._is_train,
                mem_mask=mem_mask, keep_prob=self._kprob, scope='attention')
            # enc_att = dense(enc_att, self._hidden, scope='compress')
            enc_att = dropout(enc_att, self._kprob, self._is_train)
            enc_att = layer_norm(enc_att + att_res, 'enc_att')
            self._collect_loss += loss
            return enc_att

    def _step_memory_att_block(self, att_res, memory, mem_mask, coverage_tile_now):
        with tf.variable_scope('mem_att'):
            enc_att, coverage_tile_now, _ = multi_head_attention_with_coverage(
                att_res, memory, self._heads, self._att_hidden, is_train=self._is_train,
                mem_mask=mem_mask, keep_prob=self._kprob, scope='attention', coverage=coverage_tile_now,
            )
            # enc_att = dense(enc_att, self._hidden, scope='compress')
            enc_att = dropout(enc_att, self._kprob, self._is_train)
            enc_att = layer_norm(enc_att + att_res, 'enc_att')
            return enc_att, coverage_tile_now

    def _layer_step_coverage(self, inputs, memory, mem_mask, input_before, coverage_tile_now):
        att_res = self._step_self_att_block(inputs, input_before)
        enc_att, coverage_tile_now = self._step_memory_att_block(att_res, memory, mem_mask,
                                                                 coverage_tile_now=coverage_tile_now)
        res = self._step_ffd_block(enc_att)
        return res, coverage_tile_now

    def step(self, single_input, before_input, all_layer=False, reuse=True):
        """
        before_input 是一个字典，需要记录每一层的之前的输出
        """
        with tf.variable_scope(self._scope, reuse=reuse):
            hidden = single_input.shape.as_list()[-1]
            if self._hidden != hidden:
                single_input = self._ffd_fn(dense(single_input, self._hidden, use_bias=False,
                                                  scope='input_proj', reuse=reuse))

            before_input['layer_-1_output'] = tf.cond(
                before_input['is_start'], lambda: single_input,
                lambda: tf.concat([before_input['layer_-1_output'], single_input], axis=1)
            )

            mem_length = tf.shape(self._step_memory)[1]
            before_input['layer_0_coverage'] = tf.cond(
                before_input['is_start'], lambda: tf.zeros((self._batch, self._heads, mem_length, 1), dtype=tf.float32),
                lambda: before_input['layer_0_coverage']
            )

            outputs = [single_input]
            for layer in range(self._layer):
                now_key = 'layer_{}_'.format(layer)
                pre_key = 'layer_{}_'.format(layer - 1)
                with tf.variable_scope('layer_{}'.format(layer)):
                    now_out, coverage = self._layer_step_coverage(
                        outputs[-1], self._step_memory, self._step_mem_mask,
                        before_input[pre_key + 'output'], before_input[now_key + 'coverage']
                    )
                    outputs.append(now_out)
                    before_input[now_key + 'output'] = tf.cond(
                        before_input['is_start'], lambda: now_out,
                        lambda: tf.concat([before_input[now_key + 'output'], now_out], axis=1)
                    )
                    before_input[now_key + 'coverage'] = coverage

            before_input['is_start'] = tf.constant(False)
            return outputs[1:] if all_layer else outputs[-1], before_input


class TransformerDecoderCLow(TransformerDecoderCoverage):

    def _train_memory_att_block(self, att_res, memory, mem_mask):
        with tf.variable_scope('mem_att'):
            enc_att, _, loss = multi_head_attention_coverage_low(
                att_res, memory, self._heads, self._att_hidden, is_train=self._is_train,
                mem_mask=mem_mask, keep_prob=self._kprob, scope='attention')
            # enc_att = dense(enc_att, self._hidden, scope='compress')
            enc_att = dropout(enc_att, self._kprob, self._is_train)
            enc_att = layer_norm(enc_att + att_res, 'enc_att')
            self._collect_loss += loss
            return enc_att

    def _step_memory_att_block(self, att_res, memory, mem_mask, coverage_tile_now):
        with tf.variable_scope('mem_att'):
            enc_att, coverage_tile_now, _ = multi_head_attention_coverage_low(
                att_res, memory, self._heads, self._att_hidden, is_train=self._is_train,
                mem_mask=mem_mask, keep_prob=self._kprob, scope='attention', coverage=coverage_tile_now,
            )
            # enc_att = dense(enc_att, self._hidden, scope='compress')
            enc_att = dropout(enc_att, self._kprob, self._is_train)
            enc_att = layer_norm(enc_att + att_res, 'enc_att')
            return enc_att, coverage_tile_now


def convolution(inputs, output_size, bias=True, kernel_size=1, mode='SAME', scope="conv"):
    with tf.variable_scope(scope):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_", filter_shape, dtype=tf.float32, )
        outputs = conv_func(inputs, kernel_, strides, mode)
        if bias:
            outputs += tf.get_variable("bias_", bias_shape, initializer=tf.zeros_initializer())
        return outputs


def dropout(inputs, keep_prob, is_train):
    if keep_prob < 1.0:
        inputs = tf.cond(is_train, lambda: tf.nn.dropout(inputs, keep_prob), lambda: inputs)
    return inputs


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ", is_prob=False):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        if not is_prob:
            return res
        else:
            return res, a


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention", is_prob=False):
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            if mask is not None:
                mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
                outputs = softmax_mask(outputs, mask)
            logits = tf.nn.softmax(outputs)
            outputs = tf.matmul(logits, memory)
            return outputs, logits
        # res = tf.concat([inputs, outputs], axis=2)
        # with tf.variable_scope("gate"):
        #     dim = res.get_shape().as_list()[-1]
        #     d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
        #     gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
        #     return res * gate


def dense(inputs, hidden, use_bias=True, scope="dense", reuse=False):
    return tf.layers.dense(inputs, hidden, use_bias=use_bias, name=scope, reuse=reuse)


def batch_norm(x, is_train, scope='batch_norm'):
    with tf.variable_scope(scope):
        res = tf.layers.batch_normalization(x, training=is_train)
        return res


def layer_norm(x, scope='', reuse=False, epsilon=1e-6):
    # return tf.contrib.layers.layer_norm(
    #     inputs=x, begin_norm_axis=-1, begin_params_axis=-1, scope='layer_norm_' + scope, reuse=reuse)
    filters = x.shape.as_list()[-1]
    with tf.variable_scope('layer_norm_' + scope, reuse=reuse):
        scale = tf.get_variable(
            "scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "bias", [filters], initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias


def semantic_fusion_net(state, others: list, hidden=100, scope='fusion'):
    state_hidden = state.shape.as_list()[-1]
    with tf.variable_scope(scope):
        gate_can = tf.concat([state] + others, -1)
        with tf.variable_scope('candidate'):
            new_ = gelu(dense(gelu(dense(gate_can, hidden, scope='w1')),
                              state_hidden, scope='w2'))
        with tf.variable_scope('gate'):
            gate_ = tf.nn.sigmoid(dense(gelu(dense(gate_can, hidden, scope='w1')),
                                        state_hidden, scope='w2'))
        return new_ * gate_ + (1 - gate_) * state


def highway_net(x, hidden, layers=1, activation=tf.nn.relu, scope='highway_net'):
    with tf.variable_scope(scope):
        if x.shape.as_list()[-1] != hidden:
            x = dense(x, hidden, use_bias=False, scope='input_proj')
        for ll in range(layers):
            gl, nl = tf.split(dense(x, 2 * hidden, scope='kerel_%d' % ll), 2, axis=-1)
            gate = tf.nn.sigmoid(gl)
            new = activation(nl)
            x = gate * x + new * (1.0 - gate)
        return x


def res_net(x, hidden, layers=1, scope='res_net'):
    x_hidden = x.shape.as_list()[-1]
    with tf.variable_scope(scope):
        x_new = x
        for ll in range(layers - 1):
            x_new = tf.nn.relu(dense(x_new, hidden, scope='layer_%s' % ll))
        x_new = dense(x_new, x_hidden, scope='layer_%s' % layers)
        return x_new + x


def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x = tf.pad(x, [[0, 0], [0, tf.mod(x, 2)]])
    x, x_h = tf.split(x, 2, axis=-1)
    return tf.nn.sigmoid(x) * x_h


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def add_timing_signal(x, position=None, min_timescale=1.0, max_timescale=1.0e4):
    """Attention is all you need from https://arxiv.org/abs/1706.03762"""
    '''
    input shape [batch, length, channels]
    channels是input的维度
    '''
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    num_timescales = channels // 2
    log_timescale_increment = np.log(max_timescale / min_timescale) / (tf.to_float(num_timescales) - 1 + 1e-20)
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

    if position is None:
        position = tf.to_float(tf.range(length))
    else:
        position = tf.to_float(tf.convert_to_tensor([position]))
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return x + signal


def batch_gather(params, indices, name=None):
    """
    copy from tensorflow 1.12
    """

    with tf.name_scope(name):
        indices_shape = tf.shape(indices)
        params_shape = tf.shape(params)
        ndims = indices.shape.ndims
        if ndims is None:
            raise ValueError("batch_gather does not allow indices with unknown "
                             "shape.")
        batch_indices = indices
        accum_dim_value = 1
        for dim in range(ndims - 1, 0, -1):
            dim_value = params_shape[dim - 1]
            accum_dim_value *= params_shape[dim]
            dim_indices = tf.range(0, dim_value, 1)
            dim_indices *= accum_dim_value
            dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] * (ndims - dim),
                                 axis=0)
            batch_indices += tf.reshape(dim_indices, dim_shape)

        flat_indices = tf.reshape(batch_indices, [-1])
        outer_shape = params_shape[ndims:]
        flat_inner_shape = tf.reduce_prod(
            params_shape[:ndims], [0], False)

        flat_params = tf.reshape(
            params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
        flat_result = tf.gather(flat_params, flat_indices)
        result = tf.reshape(flat_result, tf.concat([indices_shape, outer_shape], axis=0))
        final_shape = indices.get_shape()[:ndims - 1].merge_with(
            params.get_shape()[:ndims - 1])
        final_shape = final_shape.concatenate(indices.get_shape()[ndims - 1])
        final_shape = final_shape.concatenate(params.get_shape()[ndims:])
        result.set_shape(final_shape)
        return result


def batch_scatter(shape, indices, updates, name=None):
    """modify from tensorflow 1.12  `tf.batch_scatter_update`"""
    with tf.name_scope(name):
        indices_shape = tf.shape(indices)
        indices_dimensions = indices.get_shape().ndims

        if indices_dimensions is None:
            raise ValueError("batch_gather does not allow indices with unknown "
                             "shape.")

        nd_indices = tf.expand_dims(indices, axis=-1)
        nd_indices_list = []

        for dimension in range(indices_dimensions - 1):
            dimension_size = indices_shape[dimension]
            shape_to_broadcast = [1] * (indices_dimensions + 1)
            shape_to_broadcast[dimension] = dimension_size
            dimension_range = tf.reshape(
                tf.range(0, dimension_size, 1), shape_to_broadcast)
            if dimension_range.dtype.base_dtype != nd_indices.dtype:
                dimension_range = tf.cast(dimension_range, nd_indices.dtype)
            nd_indices_list.append(
                dimension_range * tf.ones_like(nd_indices))
        nd_indices_list.append(nd_indices)
        final_indices = tf.concat(nd_indices_list, axis=-1)
        return tf.scatter_nd(final_indices, updates, shape)


def tensor_shape(tensor, message=''):
    return tf.Print(tensor, [tf.shape(tensor)], message=message, summarize=1000)


def get_act_fn(name):
    fn_dict = {
        'selu': tf.nn.selu, 'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid,
        'glu': glu, 'gelu': gelu, 'softmax': tf.nn.softmax, 'elu': tf.nn.elu, 'relu6': tf.nn.relu6
    }
    if name not in fn_dict:
        raise NotImplementedError('Do not support activation `%s`' % name)
    return fn_dict[name]


def multi_head_attention(query, memory, heads, hidden, mem_mask=None, keep_prob=1.0, is_train=None,
                         scope='multi_head_att', q_act=None, k_act=None, v_act=None, is_prob=False):
    def _change_to_multi_head(inputs, fn=None, scope='dense', is_transpose=False):
        inputs = dense(inputs, hidden * heads, scope=scope, use_bias=True)
        if fn is not None:
            inputs = fn(inputs)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], heads, hidden])
        if not is_transpose:
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
        else:
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        inputs = dropout(inputs, keep_prob, is_train)
        return inputs

    with tf.variable_scope(scope):
        q_ = _change_to_multi_head(query, q_act, 'q_')
        k_ = _change_to_multi_head(memory, k_act, 'k_', True)
        v_ = _change_to_multi_head(memory, v_act, 'v_')
        # 此处的矩阵转置开销了一定的时间，直接在上面函数转置
        sim = tf.matmul(q_, k_) / (hidden ** 0.5)
        if mem_mask is not None:
            sim = softmax_mask(sim, mem_mask)
        prob = dropout(tf.nn.softmax(sim), keep_prob, is_train)

        output = tf.matmul(prob, v_)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [tf.shape(query)[0], tf.shape(query)[1], heads * hidden])
        if is_prob:
            return output, tf.reduce_mean(prob, axis=1)
        else:
            return output


def multi_head_attention_with_coverage(query, memory, heads, hidden, mem_mask=None, keep_prob=1.0, is_train=None,
                                       scope='multi_head_att', q_act=None, k_act=None, v_act=None, coverage=None):
    """
    仅在**decoder阶段**使用，仅在非self-attention中使用
    需要计算累计的attention权重
        Get To The Point: Summarization with Pointer-Generator Networks
    采用如下公式作为取代
        $ score = (q_^T (k_ + w_c c_i^t)) / hidden ** 0.5 $

    """

    def _change_to_multi_head(inputs, fn=None, scope='dense'):
        inputs = dense(inputs, hidden * heads, scope=scope, use_bias=True)
        if fn is not None:
            inputs = fn(inputs)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], heads, hidden])
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        inputs = dropout(inputs, keep_prob, is_train)
        return inputs

    with tf.variable_scope(scope):
        batch, q_length = tf.shape(query)[0], tf.shape(query)[1]
        memory_length = tf.shape(memory)[1]

        q_ = _change_to_multi_head(query, q_act, 'q_')  # q_ [batch, heads, q_len, hidden]
        k_ = _change_to_multi_head(memory, k_act, 'k_')  # k_ [batch, heads, m_len, hidden]
        v_ = _change_to_multi_head(memory, v_act, 'v_')
        wc_ = tf.get_variable('wc', shape=[1, heads, 1, hidden], dtype=tf.float32)  # 每个head不同
        wc_ = tf.tile(wc_, [batch, 1, 1, 1])

        if coverage is None:
            # 需要计算累积coverage，需要把query进行串行计算(loop)
            coverage = tf.zeros((batch, heads, memory_length, 1), dtype=tf.float32)
            prob = tf.zeros((batch, heads, q_length, memory_length))
            loop_id = 0
            loss = tf.zeros([], dtype=tf.float32)

            def _loop_cond(loop_id, prob, coverage, loss):
                return tf.less(loop_id, q_length)

            def _loop_body(loop_id, prob, coverage, loss):
                q_l = q_[:, :, loop_id: loop_id + 1, :]
                k_l = k_ + tf.nn.tanh(tf.matmul(coverage, wc_))
                sim = tf.matmul(q_l, k_l, transpose_b=True) / (hidden ** 0.5)

                if mem_mask is not None:
                    sim = softmax_mask(sim, mem_mask)  # 传入的mask维度为 [batch, head, 1, m_length]
                prob_l = dropout(tf.nn.softmax(sim), keep_prob, is_train)
                prob = tf.cond(tf.equal(loop_id, 0), lambda: prob_l, lambda: tf.concat([prob, prob_l], 2))
                # 计算 loss
                probx = tf.transpose(prob_l, [0, 1, 3, 2])
                probx = tf.reduce_min(tf.concat([coverage, probx], axis=3), axis=3)
                if mem_mask is not None:
                    probx = probx * tf.to_float(tf.squeeze(mem_mask, 2))
                loss_now = tf.reduce_mean(tf.reduce_sum(probx, axis=2))
                loss += loss_now
                coverage += tf.transpose(prob_l, [0, 1, 3, 2])
                loop_id += 1
                return loop_id, prob, coverage, loss

            _, prob, coverage, loss = tf.while_loop(
                _loop_cond, _loop_body, loop_vars=(loop_id, prob, coverage, loss),
                shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, heads, None, None]),
                                  tf.TensorShape([None, heads, None, None]), tf.TensorShape([]))
            )
            loss = loss / (tf.to_float(q_length) + 1e-17)
        else:
            k_ = k_ + tf.nn.tanh(tf.matmul(coverage, wc_))
            sim = tf.matmul(q_, k_, transpose_b=True) / (hidden ** 0.5)
            if mem_mask is not None:
                sim = softmax_mask(sim, mem_mask)
            prob = dropout(tf.nn.softmax(sim), keep_prob, is_train)
            coverage += tf.transpose(prob, [0, 1, 3, 2])
            loss = tf.constant(0.0, dtype=tf.float32)

        output = tf.matmul(prob, v_)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [tf.shape(query)[0], tf.shape(query)[1], heads * hidden])
        return output, coverage, loss


def multi_head_attention_coverage_low(query, memory, heads, hidden, mem_mask=None, keep_prob=1.0, is_train=None,
                                      scope='multi_head_att', q_act=None, k_act=None, v_act=None, coverage=None):
    def _change_to_multi_head(inputs, fn=None, scope='dense'):
        inputs = dense(inputs, hidden * heads, scope=scope, use_bias=True)
        if fn is not None:
            inputs = fn(inputs)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], heads, hidden])
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        inputs = dropout(inputs, keep_prob, is_train)
        return inputs

    with tf.variable_scope(scope):
        batch, q_length = tf.shape(query)[0], tf.shape(query)[1]
        memory_length = tf.shape(memory)[1]

        q_ = _change_to_multi_head(query, q_act, 'q_')
        k_ = _change_to_multi_head(memory, k_act, 'k_')
        v_ = _change_to_multi_head(memory, v_act, 'v_')

        wc_ = tf.get_variable('wc', shape=[1, heads, 1, hidden], dtype=tf.float32)
        wc_ = tf.tile(wc_, [batch, 1, 1, 1])
        q_wc = tf.matmul(q_, wc_, transpose_b=True) / (hidden ** 0.5)

        sim = tf.matmul(q_, k_, transpose_b=True) / (hidden ** 0.5)
        if mem_mask is not None:
            sim = softmax_mask(sim, mem_mask)
            q_wc = softmax_mask(q_wc, mem_mask)

        if coverage is None:
            # 思路类似于 GRU ==> SRU, 使recurrent部分中减少矩阵运算
            # score = (q_ k_^T) / hidden ** 0.5  [batch, head, q_len, m_len]
            # q_ wc: [batch, head, q_len, 1]
            # prob_i = softmax(score + q_wc_i * coverage_i )
            # score_i: [batch, head, 1, m_len], coverage_i: [batch, head, m_len, 1]
            # coverage = coverage + prob

            coverage = tf.zeros((batch, heads, memory_length, 1), dtype=tf.float32)
            prob = tf.zeros((batch, heads, q_length, memory_length))
            loop_id = 0
            loss = tf.zeros([], dtype=tf.float32)

            def _loop_cond(loop_id, prob, coverage, loss):
                return tf.less(loop_id, q_length)

            def _loop_body(loop_id, prob, coverage, loss):
                sim_l = sim[:, :, loop_id: loop_id + 1, :]
                qwc_l = q_wc[:, :, loop_id: loop_id + 1, :]
                real_sim_l = sim_l + tf.nn.tanh(qwc_l * tf.transpose(coverage, [0, 1, 3, 2]))
                prob_l = dropout(tf.nn.softmax(real_sim_l), keep_prob, is_train)
                prob = tf.cond(tf.equal(loop_id, 0), lambda: prob_l, lambda: tf.concat([prob, prob_l], 2))
                # 计算 loss
                probx = tf.transpose(prob_l, [0, 1, 3, 2])
                probx = tf.reduce_min(tf.concat([coverage, probx], axis=3), axis=3)
                if mem_mask is not None:
                    probx = probx * tf.to_float(tf.squeeze(mem_mask, 2))
                loss_now = tf.reduce_mean(tf.reduce_sum(probx, axis=2))
                loss += loss_now
                coverage += tf.transpose(prob_l, [0, 1, 3, 2])
                loop_id += 1
                return loop_id, prob, coverage, loss

            _, prob, coverage, loss = tf.while_loop(
                _loop_cond, _loop_body, loop_vars=(loop_id, prob, coverage, loss),
                shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, heads, None, None]),
                                  tf.TensorShape([None, heads, None, None]), tf.TensorShape([]))
            )
            loss = loss / (tf.to_float(q_length) + 1e-17)
        else:
            real_sim = sim + tf.nn.tanh(q_wc * tf.transpose(coverage, [0, 1, 3, 2]))
            prob = dropout(tf.nn.softmax(real_sim), keep_prob, is_train)
            coverage += tf.transpose(prob, [0, 1, 3, 2])
            loss = tf.constant(0.0, dtype=tf.float32)

        output = tf.matmul(prob, v_)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [tf.shape(query)[0], tf.shape(query)[1], heads * hidden])
        return output, coverage, loss


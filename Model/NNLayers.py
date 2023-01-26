import tensorflow as tf
import numpy as np


def positional_encoding(dim, sentence_length, batch_size):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    encoded_vec = encoded_vec.reshape([sentence_length, dim])
    return [encoded_vec * batch_size]


def normalize(inputs,
              epsilon=1e-8,
              scope="ln_gen",
              reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              dimension,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding_gen",
              with_t=False,
              reuse=None):
    """Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      dimension: An int. Embedding dimension size .
      l2_reg:
      with_t:
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs are multiplied by sqrt dimension.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than input's. The last dimensionality
        should be `dimension`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable('lookup_table_gen',
                                                 dtype=tf.float32,
                                                 shape=[vocab_size, dimension],
                                                 # initializer=tf.contrib.layers.xavier_initializer(),
                                                 regularizer=tf.keras.regularizers.l2(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((lookup_table[:-1, :], tf.zeros(shape=[1, dimension])), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (dimension ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def node_level_dynamic_graph_attention(inputs,
                                       n_heads,
                                       masked_relations,
                                       is_training,
                                       scope="node_level_dynamic_graph_attention",
                                       drop_out=0.6,
                                       leaky_relu_negative_slope=0.2,
                                       share_weights=True,
                                       reuse=None
                                       ):
    """
    Dynamic graph attention
    Args:
        inputs: A 3d tensor with shape [items_num, dimension]
        n_heads: Number of head attentions
        masked_relations: Relation matrix of shape [relation_nums, items_num]
        is_training: Boolean.
        drop_out: Dropout probability
        leaky_relu_negative_slope: Negative slope for leaky relu activation
        share_weights: If set to True , the same matrix will be applied to the source and the target node of every edge
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
    Return:
        A tensor of shape [ relation_nums, items_num, dimension]

    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        itms_num = inputs.shape[0]
        output_dimension = inputs.shape[1]
        assert output_dimension % n_heads == 0
        n_hidden = output_dimension // n_heads
        inputs = tf.expand_dims(inputs, axis=0)
        relation_nums = masked_relations.shape[0]
        masked_relations = tf.expand_dims(masked_relations, axis=2)
        inputs = inputs * masked_relations
        g_l = tf.compat.v1.layers.dense(inputs, n_hidden * n_heads, use_bias=False, trainable=is_training)
        g_l = tf.reshape(g_l, [relation_nums, itms_num, n_heads, n_hidden])
        if share_weights:
            g_r = g_l
        else:
            g_r = tf.compat.v1.layers.dense(inputs, n_hidden * n_heads, use_bias=False, trainable=is_training)
            g_r = tf.reshape(g_r, [relation_nums, itms_num, n_heads, n_hidden])
        g_l_repeat = tf.repeat(g_l, repeats=[itms_num] * itms_num, axis=1)
        g_r_repeat_interleave = tf.tile(g_r, [1, itms_num, 1, 1])
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = tf.reshape(g_sum, [relation_nums, itms_num, itms_num, n_heads, n_hidden])
        e = tf.compat.v1.layers.dense(tf.nn.leaky_relu(g_sum, alpha=leaky_relu_negative_slope), 1, use_bias=False,
                                      trainable=is_training)
        e = tf.squeeze(e)
        e = tf.cast(e, tf.float32)
        e = tf.where(e == 0, e, -np.inf)
        a = tf.nn.softmax(e)
        a = tf.compat.v1.layers.dropout(a, rate=drop_out, training=is_training)
        attn_res = tf.einsum('mijh,mjhf->mihf', a, g_r)
        attn_res = tf.reshape(attn_res, [relation_nums, itms_num, n_heads * n_hidden])
    return attn_res


def semantic_level_attention(inputs,
                             is_training,
                             scope="semantic_level_attention",
                             reuse=None):
    """
    Semantic level attention
    Args:
        inputs: A tensor of shape [relation_nums, itms_num, dimension]
        is_training:
        scope:
        reuse:
    Return:
         A tensor of shape [itms_num, dimension]
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        dimension = inputs.shape[-1]
        attention_vector = tf.compat.v1.get_variable('semantic_atention_vec_gen',
                                                     dtype=tf.float32,
                                                     shape=[dimension],
                                                     # initializer=tf.contrib.layers.xavier_initializer(),
                                                     regularizer=tf.keras.regularizers.l2())
        e = tf.transpose(attention_vector) @ tf.compat.v1.layers.dense(inputs, dimension, tf.math.tanh,
                                                                       trainable=is_training,
                                                                       kernel_regularizer=tf.keras.regularizers.l2(),
                                                                       bias_regularizer=tf.keras.regularizers.l2())
        e = tf.squeeze(e)
        e = tf.reduce_mean(e, axis=-1, keepdims=False)
        a = tf.nn.softmax(e)
        a = tf.expand_dims(a, axis=0)
        a = tf.expand_dims(a, axis=2)
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        attn_res = tf.reduce_sum(a * inputs, axis=1, keepdims=False)
    return attn_res


def node_level_seq_dynamic_graph_attention(inputs,
                                           n_heads,
                                           masked_relations,
                                           is_training,
                                           scope="node_level_seq_dynamic_graph_attention",
                                           drop_out=0.6,
                                           leaky_relu_negative_slope=0.2,
                                           share_weights=True,
                                           reuse=None
                                           ):
    """
    Dynamic graph attention
    Args:
        inputs: A 3d tensor with shape [batch_size, seq_length, dimension]
        n_heads: Number of head attentions
        masked_relations: Relation matrix of shape [batch_size, relation_nums, seq_length]
        is_training: Boolean.
        drop_out: Dropout probability
        leaky_relu_negative_slope: Negative slope for leaky relu activation
        share_weights: If set to True , the same matrix will be applied to the source and the target node of every edge
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
    Return:
        A tensor of shape [batch_size, relation_nums, seq_length, dimension]

    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        output_dimension = inputs.shape[2]
        assert output_dimension % n_heads == 0
        n_hidden = output_dimension // n_heads
        inputs = tf.expand_dims(inputs, axis=1)
        relation_nums = masked_relations.shape[1]
        masked_relations = tf.expand_dims(masked_relations, axis=3)
        inputs = inputs * masked_relations
        g_l = tf.compat.v1.layers.dense(inputs, n_hidden * n_heads, use_bias=False, trainable=is_training)
        g_l = tf.reshape(g_l, [batch_size, relation_nums, seq_length, n_heads, n_hidden])
        if share_weights:
            g_r = g_l
        else:
            g_r = tf.compat.v1.layers.dense(inputs, n_hidden * n_heads, use_bias=False, trainable=is_training)
            g_r = tf.reshape(g_r, [batch_size, relation_nums, seq_length, n_heads, n_hidden])
        g_l_repeat = tf.repeat(g_l, repeats=[seq_length] * seq_length, axis=2)
        g_r_repeat_interleave = tf.tile(g_r, [1, 1, seq_length, 1, 1])
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = tf.reshape(g_sum, [batch_size, relation_nums, seq_length, seq_length, n_heads, n_hidden])
        e = tf.compat.v1.layers.dense(tf.nn.leaky_relu(g_sum, alpha=leaky_relu_negative_slope), 1, use_bias=False,
                                      trainable=is_training)
        e = tf.squeeze(e)
        e = tf.cast(e, tf.float32)
        e = tf.where(e == 0, e, -np.inf)
        a = tf.nn.softmax(e)
        a = tf.compat.v1.layers.dropout(a, rate=drop_out, training=is_training)
        attn_res = tf.einsum('mnijh,mnjhf->mnihf', a, g_r)
        attn_res = tf.reshape(attn_res, [batch_size, relation_nums, seq_length, n_heads * n_hidden])
    return attn_res


def semantic_level_seq_attention(inputs,
                                 is_training,
                                 scope="semantic_level_seq_attention",
                                 reuse=None):
    """
    Semantic level attention
    Args:
        inputs: A tensor of shape [batch_size, relation_nums, seq_length, dimension]
        is_training:
        scope:
        reuse:
    Return:
         A tensor of shape [batch_size, seq_length, dimension]
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        dimension = inputs.shape[-1]
        attention_vector = tf.compat.v1.get_variable('semantic_atention_vec_gen',
                                                     dtype=tf.float32,
                                                     shape=[dimension],
                                                     # initializer=tf.contrib.layers.xavier_initializer(),
                                                     regularizer=tf.keras.regularizers.l2())
        e = tf.transpose(attention_vector) @ tf.compat.v1.layers.dense(inputs, dimension, tf.math.tanh,
                                                                       trainable=is_training,
                                                                       kernel_regularizer=tf.keras.regularizers.l2(),
                                                                       bias_regularizer=tf.keras.regularizers.l2())
        e = tf.squeeze(e)
        e = tf.reduce_mean(e, axis=-1, keepdims=False)
        a = tf.nn.softmax(e)
        a = tf.expand_dims(a, axis=1)
        a = tf.expand_dims(a, axis=3)
        inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
        attn_res = tf.reduce_sum(a * inputs, axis=2, keepdims=False)
    return attn_res


def multihead_attention(inputs,
                        scale_size,
                        mask,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention_gen",
                        reuse=None
                        ):
    """Apply scaled-multihead attention.

    Args:
      inputs: A 3d tensor with shape of [batch_size, seq_length, dimension]
      scale_size: A int number.
      dropout_rate: A floating point number.
      is_training: Boolean.
      mask: Padding masking
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer.
        by the same name.

    Returns
      A 3d tensor with shape of [batch_size, seq_length, dimension]
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fallback option for dimension
        seq_length = inputs.shape[1]
        dimension = inputs.shape[2]
        batch_size = inputs.shape[0]
        assert dimension % num_heads == 0
        assert seq_length % scale_size == 0

        # Linear projections
        Q = tf.compat.v1.layers.dense(inputs, dimension, activation=None, trainable=is_training)  # (N, T_q, C)
        K = tf.compat.v1.layers.dense(inputs, dimension, activation=None, trainable=is_training)  # (N, T_k, C)
        V = tf.compat.v1.layers.dense(inputs, dimension, activation=None, trainable=is_training)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.reshape(Q, [batch_size, seq_length, num_heads, dimension / num_heads])
        K_ = tf.reshape(K, [batch_size, seq_length, num_heads, dimension / num_heads])
        V_ = tf.reshape(V, [batch_size, seq_length, num_heads, dimension / num_heads])

        # Scale
        Q_ = tf.reshape(Q_, [batch_size, seq_length / scale_size, scale_size, num_heads, dimension / num_heads])
        K_ = tf.reshape(K_, [batch_size, seq_length / scale_size, scale_size, num_heads, dimension / num_heads])
        V_ = tf.reshape(V_, [batch_size, seq_length / scale_size, scale_size, num_heads, dimension / num_heads])
        # Q_ = tf.transpose(Q_, [0, 1, 3, 2, 4])
        # K_ = tf.transpose(K_, [0, 1, 3, 2, 4])
        # V_ = tf.transpose(V_, [0, 1, 3, 2, 4])
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 1, 2, 4, 3]))

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.reshape(mask, [batch_size, seq_length / scale_size, scale_size])
        key_masks = tf.expand_dims(key_masks, axis=-1)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)

        # Restore shape
        outputs = tf.reshape(outputs, [batch_size, seq_length / scale_size, scale_size, dimension])
        outputs = tf.reshape(outputs, [batch_size, seq_length, dimension])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)
        return outputs


def multiscale_attention(inputs,
                         scale_list,
                         mask,
                         num_head,
                         dropout_rate=0,
                         is_training=True,
                         scope="multiscale_attention_gen",
                         reuse=None
                         ):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        output = inputs
        scaled_outputs = []
        for scale in scale_list:
            scaled_output = multihead_attention(output, scale, mask, num_head, dropout_rate,
                                                reuse=reuse, is_training=is_training)
            scaled_outputs.append(scaled_output)
        scaled_outputs = tf.reduce_mean(tf.convert_to_tensor(scaled_outputs), axis=0)
        output = feedforward(scaled_outputs, is_training=is_training)
        return output


def feedforward(inputs,
                scope="multihead_attention_gen",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    """Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        dimension = inputs.shape(-1)
        # Inner layer
        outputs = tf.compat.v1.layers.dense(inputs, dimension, activation=tf.nn.relu, use_bias=True,
                                            trainable=is_training)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, trainable=is_training)

        # Residual connection
        outputs = outputs * 0.5 + inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs

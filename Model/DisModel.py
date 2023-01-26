from NNLayers import *


class Dis:
    def __int__(self, item_num, cat_num, args, reuse):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.usr_beh = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_itm = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_itm_pos = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.category_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.beh_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.itms = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.label = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
        mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.usr_itm_pos, 0)), -1)

        with tf.compat.v1.variable_scope("Discriminator", reuse=reuse):
            self.beh_masks = tf.transpose(self.beh_masks, perm=[1, 0, 2])
            self.category_masks = tf.transpose(self.category_masks, perm=[1, 0, 2])
            self.itms_emb, self.itm_emb_table = embedding(self.itms,
                                                          vocab_size=item_num + 1,
                                                          dimension=args.dimension,
                                                          zero_pad=False,
                                                          scale=True,
                                                          l2_reg=args.l2_reg,
                                                          scope="itm_embedding_dis",
                                                          with_t=True,
                                                          reuse=reuse)
            # Node level aggregation
            self.itm_cat_agg = node_level_dynamic_graph_attention(self.itms_emb,
                                                                  n_heads=args.head_attention_num,
                                                                  is_training=self.is_training,
                                                                  masked_relations=self.category_masks,
                                                                  scope="cat_node_level_dynamic_graph_attention_dis",
                                                                  reuse=reuse)
            self.itm_beh_agg = node_level_dynamic_graph_attention(self.itms_emb,
                                                                  n_heads=args.head_attention_num,
                                                                  is_training=self.is_training,
                                                                  masked_relations=self.beh_masks,
                                                                  scope="beh_node_level_dynamic_graph_attention_dis",
                                                                  reuse=reuse)
            # Semantic level aggregation
            self.itms_emb = tf.concat([self.itm_cat_agg, self.itm_beh_agg], axis=0)
            self.itms_emb = semantic_level_attention(self.itms_emb, is_training=self.is_training,
                                                     scope="semantic_level_attention_dis", reuse=reuse)

            # Update new itm emb
            indices = tf.expand_dims(self.itms, axis=-1)
            tmp_itm_emb_table = tf.tensor_scatter_nd_update(self.itm_emb_table, indices, self.itms_emb)
            self.itm_emb_table = tf.compat.v1.assign(self.itm_emb_table, tmp_itm_emb_table)

            self.beh_emb_seq, self.beh_emb_table = embedding(self.usr_beh,
                                                             vocab_size=args.beh_num + 1,
                                                             dimension=args.dimension,
                                                             zero_pad=True,
                                                             scale=True,
                                                             l2_reg=args.l2_reg,
                                                             scope="beh_embedding_dis",
                                                             with_t=True,
                                                             reuse=reuse)
            self.input_seq = tf.nn.embedding_lookup(self.itm_emb_table, self.usr_itm)
            self.pos_emb_seq = tf.convert_to_tensor(positional_encoding(args.dimension, args.max_len,
                                                                        self.usr_itm.shape[0]).reshape(
                [args.dis_batch_size, args.max_len, args.dimension]), dtype=tf.float32)
            self.input_seq = self.input_seq + self.beh_emb_seq

            self.input_seq += self.pos_emb_seq
            self.input_seq *= mask
            for i in range(args.dis_multiscale_layer):
                with tf.variable_scope("num_blocks_dis_pop_%d" % i):
                    self.input_seq = multiscale_attention(self.input_seq,
                                                          scale_list=args.scale_list,
                                                          is_training=self.is_training,
                                                          dropout_rate=args.dis_dropout_rate,
                                                          scope="multiscale_attention_dis",
                                                          mask=mask, num_head=args.dis_multiscale_head)
            self.input_seq = normalize(self.input_seq)
            self.input_seq = self.input_seq[:, -1, :]
            l2_reg_lambda = 0.2
            l2_loss = tf.constant(0.0)
            with tf.name_scope("output"):
                w1 = tf.compat.v1.Variable(tf.random.truncated_normal([args.dimension, 2], stddev=0.1), name="W1")
                b1 = tf.compat.v1.Variable(tf.constant(0.1, shape=[2]), name="b1")
                l2_loss += tf.nn.l2_loss(w1)
                l2_loss += tf.nn.l2_loss(b1)
                self.scores = tf.compat.v1.nn.xw_plus_b(self.input_seq, w1, b1, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)
                self.loss = loss + l2_reg_lambda * l2_loss

        if reuse is None:
            self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
            self.global_step = tf.compat.v1.Variable(0, name='global_step_dis', trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.dis_lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

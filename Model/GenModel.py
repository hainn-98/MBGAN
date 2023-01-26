from NNLayers import *


class Gen:
    def __int__(self, item_num, cat_num, args, reuse=None):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.usr_beh = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_itm = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_itm_pos = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_masked_itm_pos = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_masked_beh = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.usr_masked_itm = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.category_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.beh_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.beh_seq_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, None, args.max_len])
        self.itms = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.rewards = tf.compat.v1.placeholder(tf.float32, shape=[None, args.maxlen])
        self.usr_neg_itm = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.usr_pos_itm = tf.compat.v1.placeholder(tf.int32, shape=[None, args.max_len])
        self.test_set = tf.compat.v1.placeholder(tf.int32, shape=[None, None])

        with tf.compat.v1.variable_scope("Generator", reuse=reuse):
            self.beh_masks = tf.transpose(self.beh_masks, perm=[1, 0, 2])
            self.category_masks = tf.transpose(self.category_masks, perm=[1, 0, 2])
            self.itms_emb, self.itm_emb_table = embedding(self.itms,
                                                          vocab_size=item_num + 1,
                                                          dimension=args.dimension,
                                                          zero_pad=False,
                                                          scale=True,
                                                          l2_reg=args.l2_reg,
                                                          scope="itm_embedding_gen",
                                                          with_t=True,
                                                          reuse=reuse)
            # Node level aggregation
            self.itm_cat_agg = node_level_dynamic_graph_attention(self.itms_emb,
                                                                  n_heads=args.head_attention_num,
                                                                  is_training=self.is_training,
                                                                  masked_relations=self.category_masks,
                                                                  scope="cat_node_level_dynamic_graph_attention_gen",
                                                                  reuse=reuse)
            self.itm_beh_agg = node_level_dynamic_graph_attention(self.itms_emb,
                                                                  n_heads=args.head_attention_num,
                                                                  is_training=self.is_training,
                                                                  masked_relations=self.beh_masks,
                                                                  scope="beh_node_level_dynamic_graph_attention_gen",
                                                                  reuse=reuse)
            # Semantic level aggregation
            self.itms_emb = tf.concat([self.itm_cat_agg, self.itm_beh_agg], axis=0)
            self.itms_emb = semantic_level_attention(self.itms_emb, is_training=self.is_training,
                                                     scope="semantic_level_attention_gen", reuse=reuse)

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
                                                             scope="beh_embedding_gen",
                                                             with_t=True, reuse=reuse)
            self.input_seq = tf.nn.embedding_lookup(self.itm_emb_table, self.usr_itm)
            self.pos_emb_seq = tf.convert_to_tensor(positional_encoding(args.dimension, args.max_len,
                                                                        self.usr_itm.shape[0]).reshape(
                [args.gen_batch_size, args.max_len, args.dimension]), dtype=tf.float32)
            # Node level aggregation
            self.beh_agg_seq = node_level_seq_dynamic_graph_attention(self.input_seq,
                                                                      n_heads=args.head_attention_num,
                                                                      is_training=self.is_training,
                                                                      masked_relations=self.beh_seq_masks,
                                                                      scope="beh_node_level_seq_dynamic_graph_attention_gen",
                                                                      reuse=reuse)

            # self.cat_agg_seq = node_level_dynamic_graph_attention(self.input_seq,
            #                                                       n_heads=args.head_attention_num,
            #                                                       masked_relations=self.category_masks,
            #                                                       scope="cat_node_level_dynamic_graph_attention_gen",
            #                                                       reuse=reuse)
            # Semantic level aggregation
            # self.input_seq = semantic_level_seq_attention(self.beh_agg_seq, scope="semantic_level_attention_gen", reuse=reuse)

            # Replace masked token with beh-aggregated token
            for i in range(4):
                usr_masked_beh = tf.where(tf.equal(self.usr_masked_beh, i), 1, 0)
                usr_masked_beh = tf.expand_dims(usr_masked_beh, axis=-1)
                beh_agg = self.beh_agg_seq[:, i, :, :]
                beh_agg = tf.squeeze(beh_agg, axis=1)
                beh_agg = beh_agg * usr_masked_beh
                beh_agg = tf.where(tf.equal(self.usr_masked_itm_pos, 1), beh_agg, 0)
                self.input_seq = tf.where(tf.equal(self.usr_masked_itm_pos, 1), 0, self.input_seq)
                self.input_seq += beh_agg
            self.input_seq += self.beh_emb_seq
            self.input_seq += self.pos_emb_seq
            padding_mask = tf.where(tf.equal(self.usr_itm_pos, 0), 0, 1)
            for i in range(args.gen_multiscale_layer):
                with tf.variable_scope("num_blocks_gen_%d" % i):
                    self.input_seq = multiscale_attention(self.input_seq, scale_list=args.scale_list,
                                                          is_training=self.is_training,
                                                          dropout_rate=args.gen_dropout_rate,
                                                          mask=padding_mask, num_head=args.gen_multiscale_head)

        self.usr_pos_itm = tf.reshape(self.usr_pos_itm, [tf.shape(self.usr_pos_itm)[0] * args.max_len])
        self.usr_neg_itm = tf.reshape(self.usr_pos_itm, [tf.shape(self.usr_neg_itm)[0] * args.max_len])
        pos_emb = tf.nn.embedding_lookup(self.itm_emb_table, self.usr_pos_itm)
        neg_emb = tf.nn.embedding_lookup(self.itm_emb_table, self.usr_neg_itm)
        seq_emb = tf.reshape(self.input_seq, [tf.shape(self.input_seq)[0] * args.max_len, args.dimension])

        last_itm_emb = self.input_seq[:, -1, :]
        self.last_itm_logits = tf.linalg.matmul(last_itm_emb, self.itm_emb_table, transpose_b=True)

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # Test
        test_set_emb = tf.nn.embedding_lookup(self.itm_emb_table, self.test_set)
        self.test_logits = tf.matmul(last_itm_emb, tf.transpose(test_set_emb))

        # ignore padding items (0)
        istarget = tf.reshape(tf.compat.v1.to_float(tf.not_equal(self.usr_itm_pos, 0)),
                              [tf.shape(self.input_seq)[0] * args.maxlen])
        self.pre_loss = tf.reduce_sum(
            - tf.math.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.math.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.pre_loss += sum(reg_losses)

        self.pre_global_step = tf.compat.v1.Variable(0, name='global_step_gen', trainable=False)
        self.pre_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.gen_lr, beta2=0.98)
        self.pre_train_op = self.pre_optimizer.minimize(self.pre_loss, global_step=self.pre_global_step,
                                                        var_list=tf.compat.v1.get_collection(
                                                            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope="Generator"))

        self.gen_loss = tf.reduce_sum(
            (- tf.math.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget * self.rewards -
             tf.math.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget)
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.gen_loss += sum(reg_losses)

        self.gen_global_step = tf.Variable(0, name='global_step_gen', trainable=False)
        self.gen_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.gen_lr, beta2=0.98)
        self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss, global_step=self.gen_global_step,
                                                        var_list=tf.compat.v1.get_collection(
                                                            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope="Generator")
                                                        )

    def predict(self, sess, usr_beh, usr_itm, usr_itm_pos, usr_masked_itm_pos, usr_masked_beh, category_masks,
                beh_masks, beh_seq_masks, test_set, itms):
        logits = sess.run(self.test_logits,
                          {self.usr_beh: usr_beh, self.usr_itm: usr_itm, self.usr_itm_pos: usr_itm_pos,
                           self.usr_masked_beh: usr_masked_beh, self.usr_masked_itm_pos: usr_masked_itm_pos,
                           self.category_masks: category_masks, self.beh_masks: beh_masks,
                           self.beh_seq_masks: beh_seq_masks, self.test_set: test_set, self.itms: itms}
                          )
        return logits

    def generate_last_itm(self, sess, usr_beh, usr_itm, usr_itm_pos, usr_masked_itm_pos, usr_masked_beh, category_masks,
                          beh_masks, beh_seq_masks, itms):
        top_itms = np.zeros([usr_beh.shape[0]])
        logits = sess.run(self.last_itm_logits,
                          {self.usr_beh: usr_beh, self.usr_itm: usr_itm, self.usr_itm_pos: usr_itm_pos,
                           self.usr_masked_beh: usr_masked_beh, self.usr_masked_itm_pos: usr_masked_itm_pos,
                           self.category_masks: category_masks, self.beh_masks: beh_masks,
                           self.beh_seq_masks: beh_seq_masks,
                           self.itms: itms})
        logits = -logits
        index = logits.argsort()
        for i in range(len(index)):
            top_itms[i] = index[i][0]
        return top_itms

    def generate_top_k(self, sess, usr_beh, usr_itm, usr_itm_pos, usr_masked_itm_pos, usr_masked_beh, category_masks,
                       beh_masks, beh_seq_masks, itms, k, is_test=False):
        top_itms = []
        logits = sess.run(self.last_itm_logits,
                          {self.usr_beh: usr_beh, self.usr_itm: usr_itm, self.usr_itm_pos: usr_itm_pos,
                           self.usr_masked_beh: usr_masked_beh, self.usr_masked_itm_pos: usr_masked_itm_pos,
                           self.category_masks: category_masks, self.beh_masks: beh_masks,
                           self.beh_seq_masks: beh_seq_masks,
                           self.itms: itms})
        logits = -logits
        index = logits.argsort()
        for i in range(len(index)):
            top_itms.append([0] * k)
            count = 0
            for rank in range(k):
                if (index[i][count] == usr_itm[i]) and is_test == False:
                    count += 1
                else:
                    top_itms[i][rank] = index[i][count]
        return top_itms

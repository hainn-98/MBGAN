import tensorflow as tf
from Utils.Params import args
import pickle
from Utils.Logger import log
from GenModel import Gen
from DisModel import Dis
import numpy as np
import random


def dis_expand_k(usr_itm, usr_beh, usr_itm_pos, usr_masked_beh, usr_masked_itm, sample_k):
    label = []
    usr_beh[:, -1] = usr_masked_beh
    usr_beh = tf.repeat(usr_beh, repeats=[args.top_k + 1] * args.dis_batch_size, axis=0)
    usr_itm_pos = tf.repeat(usr_itm_pos, repeats=[args.top_k + 1] * args.dis_batch_size, axis=0)
    usr_itm = tf.repeat(usr_itm, repeats=[args.top_k + 1] * args.dis_batch_size, axis=0)
    for i in range(usr_itm.shape[0]):
        uid = i // (args.top_k + 1)
        if i % (args.top_k + 1) == 0:
            usr_itm[i, -1] = usr_masked_itm[uid]
            label.append([1, 0])
        else:
            usr_itm[i, -1] = sample_k[uid][i % (args.top_k + 1)]
            label.append([0, 1])
    return usr_beh, usr_itm, usr_itm_pos, label


class MBGAN:

    def __int__(self, sess, handler):
        self.sess = sess
        self.handler = handler
        args.beh_num = len(self.handler.trn_mats)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        self.itm_num = self.handler.itm_num
        self.usr_trn = self.handler.usr_trn
        # self.cat_num = len(self.handler.categories)

    def run(self):
        self.prepare_model()
        print("--------------BEGIN--------------")
        # Pre-train generator
        variables_to_restore = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
        gen_saver = tf.compat.v1.train.Saver(variables_to_restore)
        if args.pre_train_g:
            for epoch in range(args.num_pre_generator):
                print('Pre-training generator epoch: %d' % epoch)
                all_ids = random.shuffle(self.usr_trn)
                num_gen_batch = round(len(all_ids) / args.gen_batch_size)
                for step in range(num_gen_batch):
                    st = step * args.gen_batch_size
                    end = min((st + 1) * args.gen_batch_size, self.usr_trn)
                    usr_ids = all_ids[st: end]
                    usr, usr_beh, usr_itm, usr_neg_itm, usr_masked_itm, usr_itm_pos, usr_masked_itm_pos, usr_masked_beh, \
                        itms, itms_2_beh, itms_2_cat, categories = self.handler.load_data(usr_ids)
                    usr_pos_itm = np.array(usr_itm)
                    usr_pos_itm[:, -1] = usr_masked_itm
                    category_masks = []
                    for category in categories:
                        category_mask = np.zeros_like(itms_2_cat)
                        category_mask = np.where(itms_2_cat != category, category_mask, 1)
                        category_masks.append(category_mask)
                    beh_masks = []
                    for i in range(4):
                        beh_mask = itms_2_beh[i]
                        beh_masks.append(beh_mask)

                    beh_seq_masks = []
                    for i in range(4):
                        beh_seq_mask = np.zeros_like(usr_beh)
                        beh_seq_mask = np.where(usr_beh != i, beh_seq_mask, 1)
                        beh_seq_masks.append(beh_seq_mask)

                    loss, _ = self.sess.run([self.gen_model.pre_loss, self.gen_model.gen_train_op],
                                            {self.gen_model.usr_beh: usr_beh, self.gen_model.usr_itm: usr_itm,
                                             self.gen_model.usr_itm_pos: usr_itm_pos,
                                             self.gen_model.usr_masked_beh: usr_masked_beh,
                                             self.gen_model.category_masks: category_masks,
                                             self.gen_model.beh_masks: beh_masks,
                                             self.gen_model.usr_pos_itm: usr_pos_itm,
                                             self.gen_model.usr_neg_itm: usr_neg_itm,
                                             self.gen_model.is_training: True
                                             })

                if epoch % 100 == 0:
                    print('Evaluating pre-train process', )
                    t_test = self.evaluate()
                    print(
                        'epoch:%d, NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f'
                        % (epoch, t_test['NDCG'], t_test['HR'], t_test['MRR']))
                    # f.write(str(t_test) + '\n')
                    # f.flush()
            gen_saver.save(self.sess, "models_" + args.dataset + "/generator")
        else:
            gen_saver.restore(self.sess, "./models_" + args.dataset + "/generator")
            print('Evaluating pre-train process', )
            t_test = self.evaluate()
            print(
                'NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f'
                % (t_test['NDCG'], t_test['HR'], t_test['MRR']))
            # f.write(str(t_test) + '\n')
            # f.flush()

        # Pre-train discriminator
        print('Sampling......')
        for epoch in range(args.num_pre_discriminator):
            print('Pre-training discriminator epoch: %d' % epoch)
            all_ids = random.shuffle(self.usr_trn)
            num_dis_batch = round(len(all_ids) / args.dis_batch_size)
            tot_loss = 0
            for step in range(num_dis_batch):
                st = step * args.gen_batch_size
                end = min((st + 1) * args.gen_batch_size, self.usr_trn)
                usr_ids = all_ids[st: end]
                usr, usr_beh, usr_itm, usr_itm_pos, usr_masked_itm, usr_masked_itm_pos, usr_masked_beh, \
                    itms, itms_2_beh, itms_2_cat, categories = self.handler.load_data(usr_ids)
                category_masks = []
                for category in categories:
                    category_mask = np.zeros_like(itms_2_cat)
                    category_mask = np.where(itms_2_cat != category, category_mask, 1)
                    category_masks.append(category_mask)
                beh_masks = []
                for i in range(4):
                    beh_mask = np.zeros_like(itms_2_beh)
                    beh_mask = np.where(itms_2_beh == i, beh_mask, 1)
                    beh_masks.append(beh_mask)
                beh_seq_masks = []
                for i in range(4):
                    beh_seq_mask = np.zeros_like(usr_beh)
                    beh_seq_mask = np.where(usr_beh != i, beh_seq_mask, 1)
                    beh_seq_masks.append(beh_seq_mask)
                sample_k = self.gen_model.generate_top_k(sess=self.sess,
                                                         usr_beh=usr_beh,
                                                         usr_itm=usr_itm,
                                                         usr_itm_pos=usr_itm_pos,
                                                         usr_masked_itm_pos=usr_masked_itm_pos,
                                                         usr_masked_beh=usr_masked_beh,
                                                         category_masks=category_masks,
                                                         beh_masks=beh_masks,
                                                         beh_seq_masks=beh_seq_masks,
                                                         itms=itms,
                                                         k=args.top_k)
                usr_beh, usr_itm, usr_itm_pos, label = dis_expand_k(usr_itm, usr_beh, usr_itm_pos, usr_masked_beh,
                                                                    usr_masked_itm, sample_k)
                loss, _ = self.sess.run([self.dis_model.loss, self.dis_model.train_op],
                                        {self.dis_model.usr_beh: usr_beh, self.dis_model.usr_itm: usr_itm,
                                         self.dis_model.usr_itm_pos: usr_itm_pos,
                                         self.dis_model.category_masks: category_masks,
                                         self.dis_model.beh_masks: beh_masks, self.dis_model.itms: itms,
                                         self.dis_model.label: label, self.dis_model.is_training: True})
                tot_loss += loss

        # Adversarial training
        loss_tot = []
        cnt_loss = -1
        for turn in range(args.gan_epoch_num):
            # Train the generator
            for epoch in range(args.generator_train_num):
                all_ids = random.shuffle(self.usr_trn)
                num_gen_batch = round(len(all_ids) / args.gen_batch_size)
                cnt_loss += 1
                loss_tot.append(0)
                for step in range(num_gen_batch):
                    st = step * args.gen_batch_size
                    end = min((st + 1) * args.gen_batch_size, self.usr_trn)
                    usr_ids = all_ids[st: end]
                    usr, usr_beh, usr_itm, usr_neg_itm, usr_masked_itm, usr_itm_pos, usr_masked_itm_pos, usr_masked_beh, \
                        itms, itms_2_beh, itms_2_cat, categories = self.handler.load_data(usr_ids)
                    usr_pos_itm = np.array(usr_itm)
                    usr_pos_itm[:, -1] = usr_masked_itm
                    category_masks = []
                    for category in categories:
                        category_mask = np.zeros_like(itms_2_cat)
                        category_mask = np.where(itms_2_cat != category, category_mask, 1)
                        category_masks.append(category_mask)
                    beh_masks = []
                    for i in range(4):
                        beh_mask = itms_2_beh[i]
                        beh_masks.append(beh_mask)

                    beh_seq_masks = []
                    for i in range(4):
                        beh_seq_mask = np.zeros_like(usr_beh)
                        beh_seq_mask = np.where(usr_beh != i, beh_seq_mask, 1)
                        beh_seq_masks.append(beh_seq_mask)
                    predicts = self.gen_model.generate_last_itm(sess=self.sess,
                                                                usr_beh=usr_beh,
                                                                usr_itm=usr_itm,
                                                                usr_itm_pos=usr_itm_pos,
                                                                usr_masked_itm_pos=usr_masked_itm_pos,
                                                                usr_masked_beh=usr_masked_beh,
                                                                category_masks=category_masks,
                                                                beh_masks=beh_masks,
                                                                beh_seq_masks=beh_seq_masks,
                                                                itms=itms)
                    usr_itm_dis = np.array(usr_itm)
                    usr_itm_dis[:, -1] = predicts
                    usr_beh_dis = np.array(usr_beh)
                    usr_beh_dis[:, -1] = usr_masked_beh
                    rewards = self.sess.run(self.dis_model.ypred_for_auc,
                                            {self.dis_model.usr_beh: usr_beh_dis, self.dis_model.usr_itm: usr_itm_dis,
                                             self.dis_model.usr_itm_pos: usr_itm_pos,
                                             self.dis_model.category_masks: category_masks,
                                             self.dis_model.beh_masks: beh_masks, self.dis_model.itms: itms,
                                             self.dis_model.is_training: False})
                    rewards = np.array([item[1] for item in rewards])  # batch_size * 1
                    rewards = np.reshape(np.repeat(np.expand_dims(rewards, axis=1), args.max_len, axis=1),
                                         [len(args.gen_batch_size) * args.max_len])
                    loss, _ = self.sess.run([self.gen_model.gen_loss, self.gen_model.gen_train_op],
                                            {self.gen_model.usr_beh: usr_beh, self.gen_model.usr_itm: usr_itm,
                                             self.gen_model.usr_itm_pos: usr_itm_pos,
                                             self.gen_model.usr_masked_beh: usr_masked_beh,
                                             self.gen_model.category_masks: category_masks,
                                             self.gen_model.beh_masks: beh_masks,
                                             self.gen_model.usr_pos_itm: usr_pos_itm,
                                             self.gen_model.usr_neg_itm: usr_neg_itm,
                                             self.gen_model.rewards: rewards,
                                             self.gen_model.is_training: True
                                             })
                    loss_tot[cnt_loss] += loss
                loss_tot[cnt_loss] /= num_gen_batch

                if epoch % 10 == 0:
                    print('Evaluating adversarial process', )
                    t_test = self.evaluate()
                    print(
                        'epoch:%d, NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f'
                        % (epoch, t_test['NDCG'], t_test['HR'], t_test['MRR']))
                    # f.write('turn: ' + str(turn) + ',  ' + str(t_test) + '\n')
                    # f.flush()

            # Train the discriminator
            for epoch in range(args.discriminator_train_num):
                all_ids = random.shuffle(self.usr_trn)
                print('Training turn %d generator epoch: %d' % (turn, epoch))
                num_dis_batch = round(len(all_ids) / args.dis_batch_size)
                tot_loss = 0
                for step in range(num_dis_batch):
                    st = step * args.gen_batch_size
                    end = min((st + 1) * args.gen_batch_size, self.usr_trn)
                    usr_ids = all_ids[st: end]
                    usr, usr_beh, usr_itm, usr_itm_pos, usr_masked_itm, usr_masked_itm_pos, usr_masked_beh, \
                        itms, itms_2_beh, itms_2_cat, categories = self.handler.load_data(usr_ids)
                    category_masks = []
                    for category in categories:
                        category_mask = np.zeros_like(itms_2_cat)
                        category_mask = np.where(itms_2_cat != category, category_mask, 1)
                        category_masks.append(category_mask)
                    beh_masks = []
                    for i in range(4):
                        beh_mask = np.zeros_like(itms_2_beh)
                        beh_mask = np.where(itms_2_beh == i, beh_mask, 1)
                        beh_masks.append(beh_mask)
                    beh_seq_masks = []
                    for i in range(4):
                        beh_seq_mask = np.zeros_like(usr_beh)
                        beh_seq_mask = np.where(usr_beh != i, beh_seq_mask, 1)
                        beh_seq_masks.append(beh_seq_mask)
                    sample_k = self.gen_model.generate_top_k(sess=self.sess,
                                                             usr_beh=usr_beh,
                                                             usr_itm=usr_itm,
                                                             usr_itm_pos=usr_itm_pos,
                                                             usr_masked_itm_pos=usr_masked_itm_pos,
                                                             usr_masked_beh=usr_masked_beh,
                                                             category_masks=category_masks,
                                                             beh_masks=beh_masks,
                                                             beh_seq_masks=beh_seq_masks,
                                                             itms=itms,
                                                             k=args.top_k)
                    usr_beh, usr_itm, usr_itm_pos, label = dis_expand_k(usr_itm, usr_beh, usr_itm_pos, usr_masked_beh,
                                                                        usr_masked_itm, sample_k)
                    loss, _ = self.sess.run([self.dis_model.loss, self.dis_model.train_op],
                                            {self.dis_model.usr_beh: usr_beh, self.dis_model.usr_itm: usr_itm,
                                             self.dis_model.usr_itm_pos: usr_itm_pos,
                                             self.dis_model.category_masks: category_masks,
                                             self.dis_model.beh_masks: beh_masks, self.dis_model.itms: itms,
                                             self.dis_model.label: label, self.dis_model.is_training: True})
                    tot_loss += loss

    def evaluate(self):
        epoch_hit, epoch_ndcg, epoch_mrr = [0] * 3
        all_ids = random.shuffle(self.handler.user_test)
        num_test_batch = round(len(all_ids) / args.test_batch_size)
        for step in range(num_test_batch):
            st = step * args.test_batch_size
            end = min((st + 1) * args.test_batch_size, self.usr_trn)
            usr_ids = all_ids[st: end]
            usr, usr_beh, usr_itm, usr_neg_itm, usr_masked_itm, usr_itm_pos, usr_masked_itm_pos, usr_masked_beh, \
                itms, itms_2_beh, itms_2_cat, categories = self.handler.load_data(usr_ids, is_test=True)
            category_masks = []
            for category in categories:
                category_mask = np.zeros_like(itms_2_cat)
                category_mask = np.where(itms_2_cat != category, category_mask, 1)
                category_masks.append(category_mask)
            beh_masks = []
            for i in range(4):
                beh_mask = np.zeros_like(itms_2_beh)
                beh_mask = np.where(itms_2_beh == i, beh_mask, 1)
                beh_masks.append(beh_mask)
            beh_seq_masks = []
            for i in range(4):
                beh_seq_mask = np.zeros_like(usr_beh)
                beh_seq_mask = np.where(usr_beh != i, beh_seq_mask, 1)
                beh_seq_masks.append(beh_seq_mask)
            test_set = np.concatenate((usr_neg_itm, np.array([usr_masked_itm]).T), axis=1)
            predict = self.gen_model.predict(sess=self.sess,
                                             usr_beh=usr_beh,
                                             usr_itm=usr_itm,
                                             usr_itm_pos=usr_itm_pos,
                                             usr_masked_itm_pos=usr_masked_itm_pos,
                                             usr_masked_beh=usr_masked_beh,
                                             category_masks=category_masks,
                                             beh_masks=beh_masks,
                                             beh_seq_masks=beh_seq_masks,
                                             test_set=test_set,
                                             itms=itms)
            hit, ndcg, mrr = self.calc_res(predict, usr_masked_itm, test_set)
            epoch_hit += hit
            epoch_ndcg += ndcg
            epoch_mrr += mrr
        ret = dict()
        ret['HR'] = epoch_hit / len(all_ids)
        ret['NDCG'] = epoch_ndcg / len(all_ids)
        ret['MRR'] = epoch_mrr / len(all_ids)
        return ret

    def calc_res(self, predict, test_itm, test_set):
        hit = 0
        ndcg = 0
        mrr = 0
        for j in range(predict.shape[0]):
            predvals = list(zip(predict[j], test_set[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if test_itm[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(test_itm[j]) + 2))
                mrr += np.reciprocal(shoot.index(test_itm[j]) + 1)
        return hit, ndcg, mrr

    def prepare_model(self):
        self.gen_model = Gen(self.itm_num, args)
        self.dis_model = Dis(self.itm_num, args)
        self.sess.run(tf.compat.v1.global_variables_initializer())

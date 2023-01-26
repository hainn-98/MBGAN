import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Utils.Params import args
import scipy.sparse as sp


def transpose(mat):
    coomat = sp.coo_matrix(mat)
    return csr_matrix(coomat.transpose())


def random_neq(l, r, s):
    if l == 1:
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t
    else:
        result = []
        for i in range(l):
            t = np.random.randint(1, r)
            while t in s:
                t = np.random.randint(1, r)
            result.append(t)
        return result


class DataHandler:
    def __init__(self):
        self.trn_mats = None
        self.test_mats = None
        self.max_len = args.max_len
        if args.data == 'taobao':
            predir = './Datasets/Taobao/'
            behs = ['pv', 'fav', 'cart', 'buy']
        elif args.data == 'ijcai':
            predir = './Datasets/ijcai/'
            behs = ['click', 'fav', 'cart', 'buy']
        elif args.data == 'jd':
            predir = './Datasets/JD2021/'
            behs = ['browse', 'review', 'buy']

        self.predir = predir
        self.behs = behs

    def prepare_data(self):
        trn_mats = list()
        test_mats = list()
        for i in range(len(self.behs)):
            beh = self.behs[i]
            trn_path = self.predir + beh + '_trn.npz'
            test_path = self.predir + beh + '_test.npz'
            with open(trn_path, 'rb') as fs:
                trn_mat = sp.load_npz(fs).tocsr()
            trn_mats.append(trn_mat)

            with open(test_path, 'rb') as fs:
                test_mat = sp.load_npz(fs).tocsr()
            test_mats.append(test_mat)
        self.trn_mats = trn_mats
        self.test_mats = test_mats
        self.itm_num = self.trn_mats[0].shape[1]
        self.user_trn = list(set(trn_mats[0].tocoo().row))
        self.user_test = list(set(test_mats[0].tocoo().row))
        self.mask_token = self.itm_num
        self.mask_beh = len(self.trn_mats)
        with open(self.predir + 'itm2cat.pkl', 'rb') as fp:
            self.itm_to_cat = pickle.load(fp)
        with open(self.predir + 'categories.pkl', 'rb') as fp:
            self.categories = pickle.load(fp)

    def load_data(self, users, is_test=False, top_k=args.shoot):

        def item_to_cat(itm):
            return self.itm_to_cat(itm)

        TIME, BEH, ITM = [0, 1, 2]
        if is_test:
            adjs = self.test_mats
        else:
            adjs = self.trn_mats
        pck_adjs = []
        for i in range(len(adjs)):
            pckU = adjs[i][users]
            pck_adjs.append(sp.coo_matrix(pckU))
        usr, usr_beh, usr_itm, usr_neg_itm, usr_itm_pos, usr_masked_itm, usr_masked_itm_pos, usr_masked_beh = [list()
                                                                                                               for i in
                                                                                                               range(8)]
        categories = set()
        itms = set()

        datas = [list() for i in range(len(users))]
        neg_datas = [list() for i in range(len(users))]
        for j in range(len(self.behs)):
            row = pck_adjs[j].row
            col = pck_adjs[j].col
            itms.update(col)
            data = pck_adjs[j].data
            for k in range(len(row)):
                datas[row[k]].append([data[k], j, int(col[k])])
                neg_datas[row[k]].append([random_neq(1, self.itm_num, set(data)), j, int(col[k])])

        itms = np.array(list(itms))
        itms_2_beh = [np.zeros_like(itms) for i in range(len(self.behs))]

        for i in range(len(users)):
            data = datas[i]
            neg_data = neg_datas[i]
            data.sort(key=lambda x: x[TIME], reverse=True)
            neg_data.sort(key=lambda x: x[TIME], reverse=True)
            usr_beh.append([0] * args.max_len)
            usr_itm.append([0] * args.max_len)
            if not is_test:
                usr_neg_itm.append([0] * args.max_len)
            usr_itm_pos.append([0] * args.max_len)
            usr_masked_itm_pos.append([0] * args.max_len)
            masked_idx = 0
            for i in range(len(data)):
                if i == masked_idx:
                    # usr_masked_itm[-1][-1 - i] = data[i][ITM]
                    # usr_masked_beh[-1][-1 - i] = data[i][BEH]
                    usr_masked_itm.append(data[i][ITM])
                    usr_masked_beh.append(data[i][BEH])
                    usr_itm[-1][-1 - i] = self.mask_token
                    usr_beh[-1][-1 - i] = self.mask_beh
                    usr_masked_itm_pos[-1][-1 - i] = 1
                else:
                    usr_beh[-1][-1 - i] = data[i][BEH]
                    usr_itm[-1][-1 - i] = data[i][ITM]
                    itms_2_beh[data[i][BEH]][np.where(itms == data[i][ITM])] = 1
                usr_itm_pos[-1][-1 - i] = len(data) - i
                if not is_test:
                    usr_neg_itm[-1][-1 - i] = neg_data[i][ITM]
                categories.add(self.itm_to_cat(data[i][ITM]))
            if is_test:
                tmp_itms = set(usr_itm[-1])
                usr_neg_itm.append(random_neq(top_k + 1, self.itm_num, tmp_itms))

        itms_2_cat = list(map(item_to_cat, itms))

        return usr, usr_beh, usr_itm, usr_neg_itm, usr_masked_itm, usr_itm_pos, usr_masked_itm_pos, \
            usr_masked_beh, itms, itms_2_beh, itms_2_cat, categories

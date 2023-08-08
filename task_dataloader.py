import collections
import pdb
import random
import time
import dgl

from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os
import torch.nn as nn

USER_DICT = dict()
ITEM_DICT = dict()
USER_EMB = dict()
ITEM_EMB = dict()
DOMAIN_EMB = dict()
USER_ID_DICT = dict()  # {tasks: {user_o_ids: user_n_ids]}}
ITEM_ID_DICT = dict()
ID_USER_DICT = dict()
ID_ITEM_DICT = dict()  # {tasks: {item_n_ids: item_o_ids}}

TRAIN_RECORDS = dict()  # {tasks: {user_n_ids: [item_n_ids]}}
VAL_RECORDS = dict()
TEST_RECORDS = dict()

AFT_TRAIN_BHVR_RECORDS = dict()  # {tasks: {user_n_ids: item_aggr_emb}}

TRAIN_ORI_RECORDS = dict()  # {user_o_ids: {tasks: [item_n_ids]}}
VAL_ORI_RECORDS = dict()
TEST_ORI_RECORDS = dict()

USER_OVERLAP_LIST = list()  # [(domain1, id, domain2, id)]
USER_OVERLAP_DICT = dict()  # {(domain1, domain2): [[id1s], [id2s]]]

GCN_INIT_USER_EMB = torch.tensor(0)
GCN_INIT_ITEM_EMB = torch.tensor(0)

AFT_TOTAL_USER_ID_DICT = dict()
#############################   Use in single LightGCN to construct a large graph with all data #######################
TOTAL_TRAIN_RECORDS = collections.defaultdict(list)  # {user_n_ids: [item_n_ids]}
TOTAL_VAL_RECORDS = dict()  # {tasks: {user_n_ids: [item_n_ids]}}
TOTAL_TEST_RECORDS = dict()  # {tasks: {user_n_ids: [item_n_ids]}}
TOTAL_USER_ID_DICT = dict()  # {user_o_ids: [user_n_ids]}
TOTAL_ITEM_ID_DICT = dict()  # {user_o_ids: [user_n_ids]}
USER_DOMAIN_DICT = dict()
ITEM_DOMAIN_DICT = dict()

NEG_SAMPLE_SIZE = 10


class TaskDataset(Dataset):
    def __init__(self, task, mode, dataset='Amazon_5core'):
        f = open('./data/{}/{}_{}.csv'.format(dataset, task, mode), 'r')
        self.rating_list = f.readlines()
        self.task = task
        f.close()

    def __getitem__(self, i):
        item, user, _ = self.rating_list[i].strip().split(',')
        neg_item = random.choice(tuple(ITEM_DICT[self.task]))
        while ITEM_ID_DICT[self.task][neg_item] in TRAIN_RECORDS[self.task][USER_ID_DICT[self.task][user]]:
            neg_item = random.choice(tuple(ITEM_DICT[self.task]))
        return (USER_ID_DICT[self.task][user], ITEM_ID_DICT[self.task][item]), (
            USER_ID_DICT[self.task][user], ITEM_ID_DICT[self.task][neg_item])

    def __len__(self):
        return len(self.rating_list)


class GCNDataset(Dataset):
    def __init__(self, task, mode, dataset='Amazon_5core'):
        f = open('./data/{}/{}_{}.csv'.format(dataset, task, mode), 'r')
        self.rating_list = f.readlines()
        self.task = task
        f.close()

    def __getitem__(self, i):
        item, user, _ = self.rating_list[i].strip().split(',')
        neg_item = random.choice(tuple(ITEM_DICT[self.task]))
        return USER_ID_DICT[self.task][user], ITEM_ID_DICT[self.task][item], ITEM_ID_DICT[self.task][neg_item]

    def __len__(self):
        return len(self.rating_list)


def get_test_inputs(users, task):
    item_ids = torch.arange(len(ITEM_DICT[task]))
    testids = torch.cartesian_prod(users.cpu(), item_ids)
    return testids[:, 0], testids[:, 1]


def task_dataloader(dataset, tasks, batch_size):
    initialization(dataset, tasks)
    initial_for_total_gcn(dataset, tasks)
    return base_dataloaders(batch_size, dataset, tasks)


def base_dataloaders(batch_size, dataset, tasks):
    data_loaders = {}
    global TRAIN_ORI_RECORDS
    global AFT_TRAIN_BHVR_RECORDS
    for idx, d in enumerate(tasks):
        data_loaders[d] = {}
        modes = ['train', 'val', 'test']
        for mode in modes:
            task_dataset = GCNDataset(d, mode, dataset)
            # batch_size_t = len(task_dataset) // batch_size + 1
            batch_size_t = batch_size
            data_loaders[d][mode] = DataLoader(task_dataset,
                                               num_workers=0,
                                               pin_memory=True,
                                               batch_size=batch_size_t,
                                               shuffle=True if mode == 'train' else False)
    return data_loaders


def initial_for_total_gcn(dataset, tasks):
    uid = 0
    iid = 0
    user_domain_set = collections.defaultdict(set)
    item_domain_set = collections.defaultdict(set)
    for d in tasks:
        TOTAL_VAL_RECORDS[d] = collections.defaultdict(list)
        TOTAL_TEST_RECORDS[d] = collections.defaultdict(list)
        modes = ['train', 'val', 'test']
        for mode in modes:
            f = open('./data/{}/{}_{}.csv'.format(dataset, d, mode), 'r')
            for rating in f.readlines():
                item, user, _ = rating.strip().split(',')
                if user not in TOTAL_USER_ID_DICT:
                    TOTAL_USER_ID_DICT[user] = uid
                    uid += 1
                if item not in TOTAL_ITEM_ID_DICT:
                    TOTAL_ITEM_ID_DICT[item] = iid
                    iid += 1
                if mode == 'train':
                    TOTAL_TRAIN_RECORDS[TOTAL_USER_ID_DICT[user]].append(TOTAL_ITEM_ID_DICT[item])
                elif mode == 'val':
                    TOTAL_VAL_RECORDS[d][TOTAL_USER_ID_DICT[user]].append((TOTAL_ITEM_ID_DICT[item], 1))
                elif mode == 'test':
                    TOTAL_TEST_RECORDS[d][TOTAL_USER_ID_DICT[user]].append((TOTAL_ITEM_ID_DICT[item], 1))
                user_domain_set[d].add((USER_ID_DICT[d][user], TOTAL_USER_ID_DICT[user]))
                item_domain_set[d].add((ITEM_ID_DICT[d][item], TOTAL_ITEM_ID_DICT[item]))
            f.close()
    for t in tasks:
        for u in TOTAL_VAL_RECORDS[t]:
            for neg_item_id in random.choices(list(item_domain_set[t]),
                                              k=NEG_SAMPLE_SIZE * len(TOTAL_VAL_RECORDS[t][u])):
                TOTAL_VAL_RECORDS[t][u].append((neg_item_id[1], 0))
        for u in TOTAL_TEST_RECORDS[t]:
            for neg_item_id in random.choices(list(item_domain_set[t]),
                                              k=NEG_SAMPLE_SIZE * len(TOTAL_TEST_RECORDS[t][u])):
                TOTAL_TEST_RECORDS[t][u].append((neg_item_id[1], 0))
    for t in tasks:
        USER_DOMAIN_DICT[t] = torch.LongTensor([x for _, x in sorted(user_domain_set[t])])
        ITEM_DOMAIN_DICT[t] = torch.LongTensor([x for _, x in sorted(item_domain_set[t])])


def initialization(dataset, tasks):
    global USER_DICT
    global ITEM_DICT
    global TRAIN_RECORDS
    global VAL_RECORDS
    global TEST_RECORDS
    global TRAIN_ORI_RECORDS
    global VAL_ORI_RECORDS
    global TEST_ORI_RECORDS
    DOMAIN_EMB['all'] = torch.randn((len(tasks), 128))
    user_feats = None
    item_feats = None
    if os.path.exists('./data/{}/item_feats.pt'):
        item_feats = torch.load('./data/{}/item_feats.pt'.format(dataset), map_location='cpu')
    if os.path.exists('./data/{}/user_feats.pt'):
        user_feats = torch.load('./data/{}/user_feats.pt'.format(dataset), map_location='cpu')
    item_emb_size = 128
    user_emb_size = 128
    tuid = 0
    for d in tasks:
        modes = ['train', 'val', 'test']
        USER_DICT[d] = set()
        ITEM_DICT[d] = set()

        USER_ID_DICT[d] = dict()
        ITEM_ID_DICT[d] = dict()
        ID_USER_DICT[d] = dict()
        ID_ITEM_DICT[d] = dict()
        uid = 0
        iid = 0
        for mode in modes:
            f = open('./data/{}/{}_{}.csv'.format(dataset, d, mode), 'r')
            for rating in f.readlines():
                item, user, _ = rating.strip().split(',')
                if user not in AFT_TOTAL_USER_ID_DICT:
                    AFT_TOTAL_USER_ID_DICT[user] = tuid
                    tuid += 1
                if user not in USER_DICT[d]:
                    USER_ID_DICT[d][user] = uid
                    ID_USER_DICT[d][uid] = user
                    uid += 1
                    USER_DICT[d].add(user)
                if item not in ITEM_DICT[d]:
                    ITEM_ID_DICT[d][item] = iid
                    ID_ITEM_DICT[d][iid] = item

                    iid += 1
                    ITEM_DICT[d].add(item)
            f.close()
    for i, task in enumerate(tasks):
        USER_EMB[task] = torch.randn((len(USER_DICT[task]), user_emb_size))
        ITEM_EMB[task] = torch.randn((len(ITEM_DICT[task]), item_emb_size))
        nn.init.normal_(USER_EMB[task], std=0.1)
        nn.init.normal_(ITEM_EMB[task], std=0.1)
        DOMAIN_EMB[task] = DOMAIN_EMB['all'][i]
        if user_feats and item_feats:
            for user in USER_DICT[task]:
                if user in user_feats:
                    USER_EMB[task][USER_ID_DICT[task][user]] = user_feats[user]
            for item in ITEM_DICT[task]:
                if item in item_feats:
                    ITEM_EMB[task][ITEM_ID_DICT[task][item]] = item_feats[item]
    for d in tasks:
        modes = ['train', 'val', 'test']
        TRAIN_RECORDS[d] = collections.defaultdict(list)
        VAL_RECORDS[d] = collections.defaultdict(list)
        TEST_RECORDS[d] = collections.defaultdict(list)
        for mode in modes:
            with open('./data/{}/{}_{}.csv'.format(dataset, d, mode), 'r') as f:
                for line in f.readlines():
                    item, user, _ = line.strip().split(',')
                    if mode == 'train':
                        TRAIN_RECORDS[d][USER_ID_DICT[d][user]].append(ITEM_ID_DICT[d][item])
                        if user not in TRAIN_ORI_RECORDS:
                            TRAIN_ORI_RECORDS[AFT_TOTAL_USER_ID_DICT[user]] = {d: []}
                        if d not in TRAIN_ORI_RECORDS[AFT_TOTAL_USER_ID_DICT[user]]:
                            TRAIN_ORI_RECORDS[AFT_TOTAL_USER_ID_DICT[user]][d] = []
                        TRAIN_ORI_RECORDS[AFT_TOTAL_USER_ID_DICT[user]][d].append(ITEM_ID_DICT[d][item])
                    elif mode == 'val':
                        user_id = USER_ID_DICT[d][user]
                        VAL_RECORDS[d][user_id].append((ITEM_ID_DICT[d][item], 1))
                        for neg_item in random.sample(tuple(ITEM_DICT[d]), NEG_SAMPLE_SIZE):
                            neg_item_id = ITEM_ID_DICT[d][neg_item]
                            VAL_RECORDS[d][user_id].append((neg_item_id, 0))
                        if user not in VAL_ORI_RECORDS:
                            VAL_ORI_RECORDS[user] = {d: []}
                        if d not in VAL_ORI_RECORDS[user]:
                            VAL_ORI_RECORDS[user][d] = []
                        VAL_ORI_RECORDS[user][d].append(ITEM_ID_DICT[d][item])
                    else:
                        user_id = USER_ID_DICT[d][user]
                        TEST_RECORDS[d][user_id].append((ITEM_ID_DICT[d][item], 1))
                        for neg_item in random.sample(tuple(ITEM_DICT[d]), NEG_SAMPLE_SIZE):
                            neg_item_id = ITEM_ID_DICT[d][neg_item]
                            TEST_RECORDS[d][user_id].append((neg_item_id, 0))
                        if user not in TEST_ORI_RECORDS:
                            TEST_ORI_RECORDS[user] = {d: []}
                        if d not in TEST_ORI_RECORDS[user]:
                            TEST_ORI_RECORDS[user][d] = []
                        TEST_ORI_RECORDS[user][d].append(ITEM_ID_DICT[d][item])


def create_overlap_dict(tasks):
    global USER_OVERLAP_DICT
    for i, task1 in enumerate(tasks):
        for j in range(i + 1, len(tasks)):
            task2 = tasks[j]
            for user in USER_DICT[task1]:
                if user in USER_DICT[task2]:
                    USER_OVERLAP_LIST.append((task1, USER_ID_DICT[task1][user], task2, USER_ID_DICT[task2][user]))
                    if (task1, task2) not in USER_OVERLAP_DICT:
                        USER_OVERLAP_DICT[(task1, task2)] = [[], []]
                    USER_OVERLAP_DICT[(task1, task2)][0].append(USER_ID_DICT[task1][user])
                    USER_OVERLAP_DICT[(task1, task2)][1].append(USER_ID_DICT[task2][user])


def write_file():
    tasks = ['Software']
    for task in tasks:
        train_file = task + '_train.txt'
        test_file = task + '_test.txt'
        f_tr = open(train_file, 'w')
        f_te = open(test_file, 'w')
        for u in TRAIN_RECORDS[task]:
            f_tr.write(str(u) + ' ')
            for item in TRAIN_RECORDS[task][u]:
                f_tr.write(str(item) + ' ')
            f_tr.write('\n')
        for u in TEST_RECORDS[task]:
            f_te.write(str(u) + ' ')
            for item in TEST_RECORDS[task][u]:
                f_te.write(str(item) + ' ')
            f_te.write('\n')

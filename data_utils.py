import collections
import gzip
import json
import os
import pdb
import random

import torch
from tqdm import tqdm


def data_stat():
    user_sets = []
    item_sets = dict()
    domain_behv_map = collections.defaultdict(dict)
    # dataset = 'AliAd'
    # domains = ['d2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    # dataset = 'Alibaba'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    # dataset = 'AliCCP'
    # domains = ['d0', 'd1', 'd2']
    dataset = 'Amazon_5core'
    domains = ['Arts', 'Inst', 'Music', 'Pantry', 'Video', 'Luxury']
    # dataset = 'TaobaoAd'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    for domain in domains:
        file = open(f'data/{dataset}/{domain}_train.csv', 'r')
        u_s = set()
        i_s = set()
        for line in file.readlines():
            ele = line.strip().split(',')
            u_s.add(ele[1])
            i_s.add(ele[0])
            if ele[1] not in domain_behv_map[domain]:
                domain_behv_map[domain][ele[1]] = set()
            domain_behv_map[domain][ele[1]].add(ele[0])
        user_sets.append((domain, u_s))
        item_sets[domain] = i_s

    for i in range(len(user_sets)):
        for j in range(i + 1, len(user_sets)):
            d1, u1 = user_sets[i]
            d2, u2 = user_sets[j]
            print(
                f'{d1}  {d2} (u1 ∩ u2): {len(u1.intersection(u2))} (u1 ∪ u2): {len(u1.union(u2))} (u1 ∩ u2) / (u1 ∪ u2): {len(u1.intersection(u2)) / len(u1.union(u2))}')

    for d1, u1 in user_sets:
        vlp_users = set()
        t_beh = set()
        o_beh = set()
        for uu in u1:
            for ii in domain_behv_map[d1][uu]:
                t_beh.add((uu, ii))
        for d2, u2 in user_sets:
            if d1 == d2:
                continue
            v_us = u1.intersection(u2)
            vlp_users = vlp_users.union(v_us)
            for uu in v_us:
                for ii in domain_behv_map[d2][uu]:
                    o_beh.add((uu, ii))
        num_nodes = len(u1) + len(item_sets[d1])
        print(f'{d1}: #nodes: {num_nodes}  #node overlap: {len(vlp_users)} ratio: {len(vlp_users) / num_nodes:.4f}', end=' ')
        print(f'#this INT: {len(t_beh)}  #other INT: {len(o_beh)}  ratio: {len(o_beh) / len(t_beh):.4f}')


def data_pos_stat():
    # path = 'data/AliCCP/'
    # datasets = ['d0', 'd1', 'd2']
    path = 'data/Amazon_5core/'
    datasets = ['Arts', 'Inst', 'Music', 'Pantry', 'Video', 'Luxury']
    # path = 'data/AliAd/'
    # datasets = ['d2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    # path = 'data/Taobao/'
    # datasets = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    # path = 'data/TaobaoAd/'
    # datasets = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    total_u, total_i, total_inter_set = 0, 0, set()
    for dataset in datasets:
        file_name = path + dataset + '_train.csv'
        f = open(file_name, 'r')
        uset, iset, uiset = set(), set(), set()
        for line in f.readlines():
            if line.strip() == '':
                continue
            item, user, _ = line.strip().split(',')
            uset.add(user)
            iset.add(item)
            uiset.add((user, item))
        total_u += len(uset)
        total_i += len(iset)
        num_l = len(uiset)
        total_inter_set = total_inter_set.union(uiset)
        print(
            f'Dataset: {dataset}, #Interactions: {num_l}, #Users: {len(uset)}, #Items: {len(iset)}'
            f', avg (#Interactions / #Users): {num_l / len(uset):.4f}')
    print(f'Total #Users: {total_u}, Total #Items: {total_i}, Total #Interactions: {len(total_inter_set)}')


def split_data(path):
    dir_list = os.listdir(path)
    for file in dir_list:
        if '_' in file or '.gz' in file or 'feat' in file or os.path.isdir(path + file) or 'UserBehavior' in file:
            continue
        f = open(path + file, 'r')
        lines = f.readlines()
        random.shuffle(lines)
        train_num = int(len(lines) * 0.7)
        val_num = int(len(lines) * 0.1)

        train_lines = lines[:train_num]
        val_lines = lines[train_num: train_num + val_num]
        test_lines = lines[train_num + val_num:]
        modes = ['train', 'val', 'test']
        for mode in modes:
            filename = path + file[: -4] + '_' + mode + '.csv'
            fw = open(filename, 'w')
            writelines = train_lines
            if mode == 'val':
                writelines = val_lines
            elif mode == 'test':
                writelines = test_lines
            for lin in writelines:
                fw.write(lin)


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


def create_5_core_data():
    # datasets = ['Beauty', 'Fashion', 'Luxury', 'Music', 'Pantry']
    # datasets = ['App', 'Gift', 'Mag', 'Inst', 'Software']
    datasets = ['Video']
    for dataset in datasets:
        file_name = 'data/Amazon_5core/' + dataset + '.csv'
        fw = open(file_name, 'w')
        for review in parse('data/Amazon_5core/' + dataset + '.json.gz'):
            fw.write(review['asin'] + ',' + review['reviewerID'] + ',' + str(review['overall']) + '\n')


def create_feature_dict():
    datasets = ['Arts', 'Inst', 'Music', 'Pantry', 'Video', 'Luxury']
    fea_dict = collections.defaultdict(dict)
    for ds in datasets:
        for review in parse('data/Amazon_5core/meta/meta_' + ds + '.json.gz'):
            fea_dict[ds][review['asin']] = review
            if 'Shakuhachi' in str(review) and ds == 'Inst':
                print(review['asin'])
    return fea_dict


def create_feats():
    desc_dict = torch.load('./data/Amazon_5core/item_feats.pt', map_location='cpu')
    records = collections.defaultdict(list)
    user_emb = dict()
    dataset = ['Software', 'Gift', 'Arts', 'Pantry', 'Inst']
    for ds in tqdm(dataset):
        with open(f'data/Amazon_5core/{ds}_train.csv', 'r') as f:
            for l in f:
                tokens = l.strip().split(',')
                records[tokens[1]].append(desc_dict[tokens[0]])
                if tokens[1] == 'A2KIZOKNM0A1JG':
                    print(tokens[0])
    for user in tqdm(records):
        user_emb[user] = torch.mean(torch.stack(records[user]), dim=0)
    # print(user_emb)
    # torch.save(user_emb, './data/Amazon_5core/user_feats.pt')
    print(len(records), len(user_emb))


def clear_test_file():
    # dataset = 'Alibaba2'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    # dataset = 'Amazon_5core'
    # domains = ['Arts', 'Inst', 'Music', 'Pantry', 'Video']
    # dataset = 'TaobaoAd'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    dataset = 'AliCCP'
    domains = ['d0', 'd1', 'd2']
    for domain in domains:
        file = open(f'data/{dataset}/{domain}_train.csv', 'r')
        u_s = set()
        i_s = set()
        for line in file.readlines():
            ele = line.strip().split(',')
            u_s.add(ele[1])
            i_s.add(ele[0])
        file.close()
        for mode in ['val', 'test']:
            val_file = open(f'data/{dataset}/{domain}_{mode}.csv', 'r')
            val_lines = []
            for line in val_file.readlines():
                ele = line.strip().split(',')
                if ele[1] in u_s and ele[0] in i_s:
                    val_lines.append(line)
            val_file.close()
            val_file = open(f'data/{dataset}/{domain}_{mode}.csv', 'w')
            for line in val_lines:
                val_file.write(line)
            val_file.close()


def clear_ori_file():
    # dataset = 'Alibaba2'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    # dataset = 'Amazon_5core'
    # domains = ['Arts', 'Inst', 'Music', 'Pantry', 'Video']
    # dataset = 'TaobaoAd'
    # domains = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    dataset = 'AliCCP'
    domains = ['d0', 'd1', 'd2']
    user_records = collections.defaultdict(list)
    item_records = collections.defaultdict(list)
    for domain in domains:
        file = open(f'data/{dataset}/{domain}.csv', 'r')
        for line in file.readlines():
            ele = line.strip().split(',')
            user_records[ele[1]].append(ele[0])
            item_records[ele[0]].append(ele[1])
        file.close()

    for domain in domains:
        file = open(f'data/{dataset}/{domain}.csv', 'r')
        domain_records = collections.defaultdict(list)
        for line in file.readlines():
            ele = line.strip().split(',')
            if len(user_records[ele[1]]) > 10 and len(item_records[ele[0]]) > 10:
                domain_records[ele[1]].append(ele[0])
        file.close()
        file = open(f'data/{dataset}/{domain}.csv', 'w')
        for user in domain_records:
            for item in domain_records[user]:
                file.write(f'{item},{user},1\n')
        file.close()

if __name__ == '__main__':
    # clear_ori_file()
    # split_data('data/AliCCP/')
    # clear_test_file()
    data_stat()
    data_pos_stat()
    # # create_5_core_data()
    # split_data('data/TaobaoAd/')
    # create_feats()
    # clear_test_file()
    # fea_d = create_feature_dict()
    # pdb.set_trace()

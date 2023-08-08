import collections
import logging
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from LibMTL.architecture import MMoE
from LibMTL.loss import AbsLoss
from LibMTL.metrics import AbsMetric
from task_dataloader import *
from sklearn.metrics import roc_auc_score, accuracy_score


def parse_args(parser):
    parser.add_argument('--dataset', default='Amazon_5core', type=str, help='dataset')
    parser.add_argument('--bs', default=8192, type=int, help='batch size')
    parser.add_argument('--input_size', default=128, type=int, help='input size')
    parser.add_argument('--hidden_dim', default=128, type=int, help='hidden dim')
    parser.add_argument('--aggr_dim', default=64, type=int, help='aggr dim')
    parser.add_argument('--topk', default=1, type=int, help='topk')
    parser.add_argument('--epoch', default=5000, type=int, help='epoch')
    parser.add_argument('--coe_dis', default=0.3, type=float, help='coefficient of discrepancy loss')
    parser.add_argument('--edge_dropout', default=0.7, type=float, help='Edge dropout ratio for EDDA')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--ablation', default='/', type=str, help='ablation')
    parser.add_argument('--tasks', nargs='+', type=str,
                        default=['Arts', 'Inst', 'Music', 'Pantry', 'Video', 'Luxury'])
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help='ID/Path of pretrained multi-task learning model')
    parser.add_argument('--target', type=str, default=None,
                        help='Target task. Only test one domain in a multi-domain recommendation setting.')
    parser.add_argument('--att', action='store_true', help='Using attention in MMoE')
    parser.add_argument('--thres', default=-1, type=float, help='threshold for graph similarity. -1: Average Value')
    return parser.parse_args()


class AFTLoss(AbsLoss):
    r"""The Mean Square Error loss function.
    """

    def __init__(self):
        super(AFTLoss, self).__init__()

    def compute_loss(self, l_dg, l_mmd):
        r"""
        """
        loss = l_dg + l_mmd

        return loss


class BPRLoss(AbsLoss):
    r"""The Mean Square Error loss function.
    """

    def __init__(self):
        super(BPRLoss, self).__init__()

    def compute_loss(self, pos, neg):
        r"""
        """
        loss = torch.mean(F.softplus(neg - pos))

        return loss


class LightGCNLoss(AbsLoss):
    r"""The Mean Square Error loss function.
    """

    def __init__(self):
        super(LightGCNLoss, self).__init__()

    def compute_loss(self, bpr_loss, reg_loss):
        r"""
        """
        loss = bpr_loss + reg_loss

        return loss

    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(1)
        return loss


class MSELoss(AbsLoss):
    r"""The Mean Square Error loss function.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        r"""
        """
        pred = pred.squeeze(1)
        loss = self.loss_fn(pred, gt)
        return loss


class MAEMetric(AbsMetric):
    r"""Calculate the Mean Absolute Error.
    """

    def __init__(self):
        super(MAEMetric, self).__init__()

    def update_fun(self, preded, gt):
        r"""
        """
        pred = preded.clone()
        pred = pred.squeeze(1)
        # pred[pred < 1] = 1
        # pred[pred > 5] = 5
        self.record.append((pred - gt).abs().sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return [(sum(self.record) / sum(self.bs))]


class NumMetric(AbsMetric):
    r"""Calculate the Mean Absolute Error.
    """

    def __init__(self):
        super(NumMetric, self).__init__()

    def update_fun(self, preded, gt):
        r"""
        """
        self.record.append(preded)
        self.bs.append(gt)

    def score_fun(self):
        r"""
        """
        return [(sum(self.record) / sum(self.bs))]


class MSEMetric(AbsMetric):
    r"""Calculate the Mean Square Error.
    """

    def __init__(self):
        super(MSEMetric, self).__init__()

    def update_fun(self, preded, gt):
        r"""
        """
        pred = preded.clone()
        pred = pred.squeeze(1)
        # pred[pred < 1] = 1
        # pred[pred > 5] = 5
        self.record.append(((pred - gt) ** 2).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return [(sum(self.record) / sum(self.bs))]


class RMSEMetric(AbsMetric):
    r"""Calculate the Root Mean Square Error.
    """

    def __init__(self):
        super(RMSEMetric, self).__init__()

    def update_fun(self, preded, gt):
        r"""
        """
        pred = preded.clone()
        pred = pred.squeeze(1)
        # pred[pred < 1] = 1
        # pred[pred > 5] = 5
        self.record.append(((pred - gt) ** 2).sum().item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        r"""
        """
        return [(math.sqrt(sum(self.record) / sum(self.bs)))]


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def getUsersRating(batch_users, items, model, task=None, graph=None, arch='normal'):
    mode = 'dis_all' if arch.startswith('dis') else 'normal'
    return model.getUsersRating(batch_users, items, graph, task, mode=mode)


def test_model_ndcg(model, train_records, test_records, task, graph=None, arch='normal'):
    model.eval()
    iter_batch_size = 30000000 if 'gcn' in arch else 2000000
    topks = [10, 20]
    max_K = max(topks)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(test_records.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        u_batch_size = iter_batch_size // len(ITEM_DICT[task])
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            allPos = [train_records[u] for u in batch_users]
            groundTrue = [test_records[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(model.device)
            rating = getUsersRating(batch_users_gpu, model, task, graph, arch)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = 0
            _, rating_K = torch.topk(rating, k=max_K)
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, topks))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        logging.info(f'{task}: {results}')
        return results['ndcg'][1]


def test_model(model, train_records, test_records, task, graph=None, arch='normal'):
    model.eval()
    u_batch_size = 1024
    top_ks = [1, 3]
    with torch.no_grad():
        users = list(test_records.keys())
        label_list = []
        rating_list = []
        for batch_users in minibatch(users, batch_size=u_batch_size):
            test_users = torch.LongTensor([u for u in batch_users for _ in test_records[u]]).to(model.device)
            test_items = torch.LongTensor([r[0] for u in batch_users for r in test_records[u]]).to(model.device)
            labels = torch.LongTensor([r[1] for u in batch_users for r in test_records[u]]).to(model.device)
            rating = getUsersRating(test_users, test_items, model, task, graph, arch)
            label_list.append(labels)
            rating_list.append(rating)

        label_list = torch.cat(label_list)
        rating_list = torch.cat(rating_list)
        auc = roc_auc_score(label_list.cpu().numpy(), torch.sigmoid(rating_list).cpu().numpy())

        ratings = rating_list.reshape(-1, NEG_SAMPLE_SIZE + 1)
        recalls = []
        for top_k in top_ks:
            _, rating_k = torch.topk(ratings, k=top_k)
            recall = float(torch.sum(torch.min(rating_k, dim=1).values == 0) / ratings.shape[0])
            recalls.append(recall)

        return auc, recalls[0], recalls[1]


def test_aggr_model(model, train_records, test_records, task, graph=None):
    model.eval()
    iter_batch_size = 30000000
    topks = [10, 20]
    max_K = max(topks)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(test_records.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        u_batch_size = iter_batch_size // len(TOTAL_ITEM_ID_DICT)
        for batch_users in minibatch(users, batch_size=u_batch_size):
            users_list.append(batch_users)
            allPos = [train_records[u] for u in batch_users]
            groundTrue = [test_records[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(model.device)
            rating = getUsersRating(batch_users_gpu, model, graph=graph)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = 0
            rating[:, ITEM_DOMAIN_DICT[task]] += 1
            _, rating_K = torch.topk(rating, k=max_K)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, topks))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        logging.info(f'{task}: {results}')
        return results['ndcg'][1]


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 10000)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def demo_sample(i_num, play_records):
    users = [u for u in play_records for _ in play_records[u]]
    pos_items = [pos_i for u in play_records for pos_i in play_records[u]]
    play_num = sum(len(play_records[x]) for x in play_records)
    neg_items = np.random.randint(0, i_num, play_num)

    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)


def non_sampling(play_records):
    users = list(play_records.keys())
    data = list(play_records.values())
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:, None]
    items = np.full_like(mask, -1, dtype=np.long)
    items[mask] = np.concatenate(data)
    return torch.LongTensor(users), torch.LongTensor(items)


def UniformSample_original_python(u_num, i_num, play_records):
    """
    the original implement of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time.time()
    play_num = sum(len(play_records[x]) for x in play_records)
    users = np.random.randint(0, u_num, play_num)
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time.time()
        posForUser = play_records[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time.time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, i_num)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time.time()
        sample_time1 += end - start
    total = time.time() - total_start
    print('Sample time: ', total)
    return np.array(S)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

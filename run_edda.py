import collections
import os.path
import pdb
import pickle
import time

import torch

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from baseline.EDDA import EDDA
from data_utils import create_feature_dict
from util_functions import *
from task_dataloader import *

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def create_train_graph(tasks):
    graph_pairs = dict()
    for task in tasks:
        src = []
        dst = []
        u_i_pairs = set()
        for uid in TRAIN_RECORDS[task]:
            iids = TRAIN_RECORDS[task][uid]
            for iid in iids:
                if (uid, iid) not in u_i_pairs:
                    src.append(int(uid))
                    dst.append(int(iid))
                    u_i_pairs.add((uid, iid))
        u_num, i_num = len(USER_DICT[task]), len(ITEM_DICT[task])
        src_ids = torch.tensor(src)
        dst_ids = torch.tensor(dst) + u_num
        g = dgl.graph((src_ids, dst_ids), num_nodes=u_num + i_num)
        g = dgl.to_bidirected(g)
        graph_pairs[task] = g
    return graph_pairs


def create_aggr_graph():
    src = []
    dst = []
    u_i_pairs = set()
    for uid in TOTAL_TRAIN_RECORDS:
        iids = TOTAL_TRAIN_RECORDS[uid]
        for iid in iids:
            if (uid, iid) not in u_i_pairs:
                src.append(int(uid))
                dst.append(int(iid))
                u_i_pairs.add((uid, iid))
    u_num, i_num = len(TOTAL_USER_ID_DICT), len(TOTAL_ITEM_ID_DICT)
    src_ids = torch.tensor(src)
    dst_ids = torch.tensor(dst) + u_num
    g = dgl.graph((src_ids, dst_ids), num_nodes=u_num + i_num)
    g = dgl.to_bidirected(g)
    return g


def cal_sim(domains, train_graphs):
    cos = nn.CosineSimilarity(dim=1)
    tanh = nn.Tanh()
    sim_dict = collections.defaultdict(dict)
    rw_len = 4
    walk_times = 10000
    node_sample_num_max = 10000
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            # g1 = train_graphs[domains[i]]
            # g2 = train_graphs[domains[j]]
            g1 = dgl.add_self_loop(train_graphs[domains[i]])
            g2 = dgl.add_self_loop(train_graphs[domains[j]])
            indices = torch.tensor(USER_OVERLAP_DICT[(domains[i], domains[j])])
            sampling_size = min(node_sample_num_max, indices[0].shape[0])
            ppr_g1 = torch.zeros((sampling_size, g1.num_nodes()))
            ppr_g2 = torch.zeros((sampling_size, g2.num_nodes()))

            sampled_src = indices[:, random.sample(range(indices[0].shape[0]), sampling_size)]
            trace_g1, _ = dgl.sampling.random_walk(g1, sampled_src[0].repeat(walk_times), length=rw_len)
            trace_g2, _ = dgl.sampling.random_walk(g2, sampled_src[1].repeat(walk_times), length=rw_len)
            dst_g1 = trace_g1[:, -1].reshape(walk_times, sampling_size).T
            dst_g2 = trace_g2[:, -1].reshape(walk_times, sampling_size).T
            for m in range(sampling_size):
                ppr_g1[m] = torch.bincount(dst_g1[m], minlength=g1.num_nodes())
            # ppr_g1 = ppr_g1[:, indices[0]]
            ppr_g1_norm = F.normalize(ppr_g1[:, indices[0]], p=1)
            for m in range(sampling_size):
                ppr_g2[m] = torch.bincount(dst_g2[m], minlength=g2.num_nodes())
            # ppr_g2 = ppr_g2[:, indices[1]]
            ppr_g2_norm = F.normalize(ppr_g2[:, indices[1]], p=1)

            sim = cos(ppr_g1_norm, ppr_g2_norm)
            # sim = sigmoid(sim)
            sim = tanh(sim)
            # sim = elu(sim)
            sim = float(torch.mean(sim))
            sim_dict[domains[i]][domains[j]] = sim
            sim_dict[domains[j]][domains[i]] = sim

    for dom in domains:
        sim_dict[dom][dom] = 1

    for d1 in domains:
        print('|', end='')
        for d2 in domains:
            print(f' {sim_dict[d1][d2]:.3f} ', end='')
        print('|')
    return sim_dict


def batch(x, bs):
    x = list(range(x))
    return [x[i:i + bs] for i in range(0, len(x), bs)]


def cal_align(domains, train_graphs):
    u_rw_len = 4
    i_rw_len = 3
    rw_times = 500
    node_sample_num_max = 10000
    align_dict = dict()
    batch_size = 4196
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            g1 = train_graphs[domains[i]]
            g2 = train_graphs[domains[j]]
            indices = torch.tensor(USER_OVERLAP_DICT[(domains[i], domains[j])])
            sampling_size = min(node_sample_num_max, indices[0].shape[0])
            ppr_g1 = torch.zeros((g1.num_nodes(), g1.num_nodes()))
            ppr_g2 = torch.zeros((g2.num_nodes(), g2.num_nodes()))
            sampled_src = indices[:, random.sample(range(indices[0].shape[0]), sampling_size)]
            d_g1 = []
            for nodes in batch(g1.num_nodes(), batch_size):
                trace_g1_u, _ = dgl.sampling.random_walk(g1, torch.tensor(nodes).repeat(rw_times), length=u_rw_len)
                trace_g1_i, _ = dgl.sampling.random_walk(g1, torch.tensor(nodes).repeat(rw_times), length=i_rw_len)

                tr_g1_u_end = torch.where(trace_g1_u[:, -1] == -1, trace_g1_u[:, 0], trace_g1_u[:, -1])
                tr_g1_i_end = torch.where(trace_g1_i[:, -1] == -1, trace_g1_i[:, 0], trace_g1_i[:, -1])

                dst_g1_u = tr_g1_u_end.reshape(rw_times, len(nodes)).T
                dst_g1_i = tr_g1_i_end.reshape(rw_times, len(nodes)).T
                dst_g1 = torch.cat([dst_g1_u, dst_g1_i], dim=1)
                d_g1.append(dst_g1)
            dst_g1 = torch.cat(d_g1)

            d_g2 = []
            for nodes in batch(g2.num_nodes(), batch_size):
                trace_g2_u, _ = dgl.sampling.random_walk(g2, torch.tensor(nodes).repeat(rw_times), length=u_rw_len)
                trace_g2_i, _ = dgl.sampling.random_walk(g2, torch.tensor(nodes).repeat(rw_times), length=i_rw_len)

                tr_g2_u_end = torch.where(trace_g2_u[:, -1] == -1, trace_g2_u[:, 0], trace_g2_u[:, -1])
                tr_g2_i_end = torch.where(trace_g2_i[:, -1] == -1, trace_g2_i[:, 0], trace_g2_i[:, -1])

                dst_g2_u = tr_g2_u_end.reshape(rw_times, len(nodes)).T
                dst_g2_i = tr_g2_i_end.reshape(rw_times, len(nodes)).T
                dst_g2 = torch.cat([dst_g2_u, dst_g2_i], dim=1)
                d_g2.append(dst_g2)
            dst_g2 = torch.cat(d_g2)

            for m in range(g1.num_nodes()):
                ppr_g1[m] = torch.bincount(dst_g1[m], minlength=g1.num_nodes())
            # ppr_g1 = ppr_g1[:, indices[0]]
            ppr_g1_norm = F.normalize(ppr_g1[:, sampled_src[0]], p=2)
            for m in range(g2.num_nodes()):
                ppr_g2[m] = torch.bincount(dst_g2[m], minlength=g2.num_nodes())
            # ppr_g2 = ppr_g2[:, indices[1]]
            ppr_g2_norm = F.normalize(ppr_g2[:, sampled_src[1]], p=2)
            sims = torch.matmul(ppr_g1_norm, ppr_g2_norm.T)
            align_dict[(domains[i], domains[j])] = dict()
            align_dict[(domains[j], domains[i])] = dict()
            for topk in [1, 3]:
                # topk = 3
                _, indices1 = torch.topk(sims, topk)
                src1 = torch.arange(g1.num_nodes()).unsqueeze(1).expand(-1, topk).flatten()
                dst1 = indices1.flatten()

                src1, dst1 = torch.cat([src1, indices[0]]), torch.cat([dst1, indices[1]])

                align_dict[(domains[i], domains[j])][topk] = (src1, dst1)
                _, indices2 = torch.topk(sims.T, topk)
                src2 = torch.arange(g2.num_nodes()).unsqueeze(1).expand(-1, topk).flatten()
                dst2 = indices2.flatten()

                src2, dst2 = torch.cat([src2, indices[1]]), torch.cat([dst2, indices[0]])

                align_dict[(domains[j], domains[i])][topk] = (src2, dst2)

    return align_dict


def create_aggr_domain_graph(agg_domains, graph_dict):
    aggr_graph = graph_dict['aggr']
    u_num = len(TOTAL_USER_ID_DICT)
    aggr_dom_ids = dict()
    for domain in agg_domains:
        aggr_dom_ids[domain] = ITEM_DOMAIN_DICT[domain] + u_num
    for domain in agg_domains:
        rel_domains = agg_domains[domain]
        for rel_d in rel_domains:
            aggr_dom_ids[domain] = torch.cat([aggr_dom_ids[domain], aggr_dom_ids[rel_d]]).unique()
    for domain in agg_domains:
        graph_dict['aggr' + domain] = dgl.to_bidirected(dgl.in_subgraph(aggr_graph, aggr_dom_ids[domain]))


def demo_sample(i_num, train_records):
    users = [u for u in train_records for _ in train_records[u]]
    pos_items = [pos_i for u in train_records for pos_i in train_records[u]]
    play_num = sum(len(train_records[x]) for x in train_records)
    neg_items = np.random.randint(0, i_num, play_num)

    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)


def test_similar_node(align_dict, d1, d2, item_id_in_d1, fea_d):
    item_id_in_d2 = int(align_dict[(d1, d2)][1][1][len(USER_ID_DICT[d1]) + item_id_in_d1] - len(USER_ID_DICT[d2]))
    print(fea_d[d1][ID_ITEM_DICT[d1][item_id_in_d1]])
    print()
    print(fea_d[d2][ID_ITEM_DICT[d2][item_id_in_d2]])


def main(params):
    FORMAT = '%(asctime)s: %(message)s'
    logging.basicConfig(filename='test_edda.log', format=FORMAT, level=logging.DEBUG)
    kwargs, optim_param, scheduler_param = prepare_args(params)
    print(f'coe_dis: {params.coe_dis}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task_names = params.tasks
    task_dataloader(params.dataset, task_names, batch_size=params.bs)
    create_overlap_dict(task_names)
    target_task = params.target
    # define tasks
    task_dict = {task: {'metrics': ['NDCG20'],
                        'metrics_fn': NumMetric(),
                        'loss_fn': LightGCNLoss(),
                        'weight': [1 if target_task is None or task == target_task else 0]} for task in task_names}

    hidden_dim = params.hidden_dim
    kwargs['arch_args']['hidden_dim'] = hidden_dim
    meter = _PerformanceMeter(task_dict, True)
    train_graphs = create_train_graph(task_names)
    train_graphs['aggr'] = create_aggr_graph()

    # sim_dict = cal_sim(task_names, train_graphs)
    sim_dict = dict()
    if params.coe_dis != 0:
        if os.path.exists(f'align_dict/{params.dataset}_all.pth'):
            align_dict = torch.load(f'align_dict/{params.dataset}_all.pth')
        else:
            if not os.path.exists('align_dict'):
                os.mkdir('align_dict')
            align_dict = cal_align(task_names, train_graphs)
            torch.save(align_dict, f'align_dict/{params.dataset}_all.pth')
    else:
        align_dict = None

    for task in train_graphs:
        train_graphs[task] = train_graphs[task].to(device)

    edda_model = EDDA(task_names, params.hidden_dim, sim_dict=sim_dict,
                          graph_dict=train_graphs, edge_dropout=params.edge_dropout).to(device)
    optimizer = torch.optim.Adam(edda_model.parameters(), lr=optim_param['lr'])

    for epoch in range(params.epoch):
        train_edda(train_graphs, edda_model, optimizer, task_names, device, meter, epoch,
                   params, align_dict)
        if epoch % 2 == 0:
            test_edda(train_graphs, edda_model, task_names, meter, epoch, 'test')
        if epoch == meter.best_result['epoch']:
            if not os.path.exists('./checkpoints'):
                os.mkdir('checkpoints')
            torch.save(edda_model.state_dict(), f'checkpoints/DDALG_{params.dataset}.pth')
        if epoch - meter.best_result['epoch'] > 50:
            break
    meter.display_best_result()


def train_edda(graph, model, optimizer, domains, device, meter, epoch, params, align_dict=None):
    model.train()
    meter.record_time('begin')
    topk = params.topk
    for domain in domains:
        users, posItems, negItems = demo_sample(len(ITEM_DICT[domain]), TRAIN_RECORDS[domain])
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=params.bs):
            optimizer.zero_grad()
            user, pos, neg = user.to(device), pos.to(device), neg.to(device)
            bpr_loss, reg_loss = model.bpr_loss(graph[domain], user, pos, neg, domain, mode='dis_all')
            gcn_loss = bpr_loss + reg_loss * 1e-4
            gcn_loss.backward()
            optimizer.step()
            meter.losses[domain]._update_loss(bpr_loss, reg_loss * 1e-4)
            meter.update(0, 1, domain)

    if params.coe_dis != 0:
        for i in range(len(domains)):
            for j in range(len(domains)):
                if i != j:
                    d1, d2 = domains[i], domains[j]
                    optimizer.zero_grad()
                    align_loss = model.cal_align_loss(d1, d2, align_dict[(d1, d2)][topk]) * params.coe_dis
                    align_loss.backward()
                    optimizer.step()

    meter.record_time('end')
    meter.get_score('train')
    meter.display(epoch=epoch, mode='train')
    meter.reinit()


def test_edda(train_graph, model, domains, meter, epoch, mode):
    model.eval()
    if mode == 'val' and not meter.has_val:
        meter.has_val = True
    meter.record_time('begin')
    recall_dict = dict()
    for domain in domains:
        if mode == 'val':
            auc, recall1, recall3 = test_model(model, TRAIN_RECORDS[domain], VAL_RECORDS[domain], domain,
                                               train_graph[domain])
        else:
            auc, recall1, recall3 = test_model(model, TRAIN_RECORDS[domain], TEST_RECORDS[domain], domain,
                                               train_graph[domain])
        meter.update(auc, 1, domain)
        recall_dict[domain] = {1: recall1, 3: recall3}
    meter.record_time('end')
    meter.get_score(mode)
    print('Recall@1: ', end='')
    sum_r = 0
    for dom in recall_dict:
        sum_r += recall_dict[dom][1]
        print(f'|  {dom}: {recall_dict[dom][1]:.4f}  ', end='')
    print(f'|  AVG: {sum_r / len(recall_dict):.4f}  |')
    print('Recall@3: ', end='')
    sum_r = 0
    for dom in recall_dict:
        sum_r += recall_dict[dom][3]
        print(f'|  {dom}: {recall_dict[dom][3]:.4f}  ', end='')
    print(f'|  AVG: {sum_r / len(recall_dict):.4f}  |')
    meter.display(epoch=epoch, mode=mode)
    meter.reinit()


if __name__ == '__main__':
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)

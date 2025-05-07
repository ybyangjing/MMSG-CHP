import argparse
import os
import pickle
import warnings
import dgl
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score,accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import units
from model.mamba import Mambas
from units import get_all_input, last_value, collate_fn_old, MyData
from model.cao2 import GATRE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gat_in', type=int, default=48)
    parser.add_argument('--gat_hid', type=int, default=64)
    parser.add_argument('--gat_out', type=int, default=64)
    # parser.add_argument('--gat_head', type=int, default=4)
    parser.add_argument('--mamba_input', type=int, default=64)
    parser.add_argument('--mamba_d_model', type=int, default=64)
    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=2)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_output', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=1500)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--period', type=int, default=9)
    parser.add_argument('--label_path', type=str, default=r"/media/zy/UBUNTU 20_0/augcos_label_589.pkl")
    parser.add_argument('--graphs_path', type=str, default=r"/media/zy/UBUNTU 20_0/augcos_sub_net_589.pkl")
    parser.add_argument('--n_feat_path', type=str, default=r"/media/zy/UBUNTU 20_0/augcos_n_feat_589.pkl")
    
    parser.add_argument('--e_feat_path', type=str, default=r"/media/zy/UBUNTU 20_0/augcos_e_feat_589.pkl")
    args = parser.parse_args()
    k = args.period
    units.set_k(args.period)

    save_cate = '90'
    save_path = 'value_result/cate_' + save_cate + '_result'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(save_cate)

    #load data
    dgls = get_all_input(args.graphs_path, True)
    n_feat = np.array(get_all_input(args.n_feat_path, False))
    e_feat = np.array(get_all_input(args.e_feat_path, False))
    labels = np.array(get_all_input(args.label_path, False))

    hot_rate = sum(sum(labels == 0))/sum(sum(labels == 1))
    print(hot_rate)

    # train, test split
    n = len(dgls)  
    print('num of st-nets = {}'.format(n))
    split = int(n * .8)
    index = np.arange(n) 
    np.random.seed(50)
    np.random.shuffle(index)  
    train_index, test_index = index[:split], index[split:]  

    # prep labels
    train_labels = labels[train_index]
    test_labels = torch.FloatTensor(labels[test_index])
    test_labels = test_labels.cuda()

    # prep input data
    trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]
    testGs = [dgl.batch([dgl.DGLGraph(u[i]) for u in testGs]) for i in range(k)]  
    train_n_feat, test_n_feat = [n_feat[i] for i in train_index], [n_feat[i] for i in test_index]
    test_n_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in test_n_feat])) for i in range(k)] 


    train_e_feat, test_e_feat = [e_feat[i] for i in train_index], [e_feat[i] for i in test_index]
    test_e_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in test_e_feat])) for i in range(k)]  

    data = MyData(trainGs, train_n_feat, train_e_feat, train_labels)
    data_loader = DataLoader(data, batch_size=1000, shuffle=False, collate_fn=collate_fn_old) 
    warnings.filterwarnings("ignore")
    '''*************************************************************************************************************'''

    # define models
    model_GATRE = GATRE(args.gat_in, args.gat_hid, args.gat_out,args.drop)
    model_M = Mambas(args.mamba_d_model,args.mamba_d_state,args.mamba_d_conv,args.mamba_expand,args.period,args.mamba_input,args.mamba_output)

    model_GATRE.cuda()
    model_M.cuda()

    print(model_GATRE,model_M)
    print('lr:', args.lr)

    parameters = list(model_GATRE.parameters()) +list(model_M.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    dropout = nn.Dropout(args.drop)

    # MSE_loss = nn.MSELoss().cuda()
    BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hot_rate])).cuda()

    epoch_loss_all = []
    train_score = []
    test_score = []
    acc_score = []
    pre_score = []
    auc_score = []
    re_score = []
    epoch_times = []
    for epoch in tqdm(range(args.epoch)):
        start_time1 = time.perf_counter()
        model_GATRE.train()
        model_M.train()
        for step, (g, nf, ef, batch_labels) in enumerate(data_loader):
            batch_labels = batch_labels.cuda()
            sequence = torch.stack([model_GATRE(g[i], [nf[i].cuda(), ef[i].cuda()]) for i in range(k)], 1)
            out = model_M(sequence)

            loss = BCE_loss(out, batch_labels)

            #  back propagation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_all.append(loss.item())
        end_time1 = time.perf_counter()
        epoch_duration = end_time1 - start_time1
        if epoch % 10 == 0:
            print(f"epoch_duration:{epoch_duration}")
        epoch_times.append(epoch_duration)

        # eval

        model_GATRE.eval()
        model_M.eval()
        start_time2 = time.perf_counter()
        test_sequence = torch.stack([model_GATRE(testGs[i], [test_n_feat[i].cuda(), test_e_feat[i].cuda()])
                                     for i in range(k)], 1)  # 生成序列

        out = model_M(test_sequence)
        end_time2 = time.perf_counter()
        epoch_duration2 = end_time2 - start_time2
        if epoch % 10 == 0:
            print(f"test_epoch_time:{epoch_duration2}")

        temp_label = test_labels.cpu().view(1, -1).squeeze(0)
        temp_pre = np.int64(out.cpu().detach().numpy() > 0.5).reshape(1, -1).squeeze(0)

        test_acc = accuracy_score(temp_label, temp_pre)
        test_f1 = f1_score(temp_label, temp_pre)
        test_pre = precision_score(temp_label, temp_pre)
        test_auc = roc_auc_score(temp_label, temp_pre)
        test_recall = recall_score(temp_label, temp_pre)

        acc_score.append(test_acc)
        test_score.append(test_f1)
        pre_score.append(test_pre)
        auc_score.append(test_auc)
        re_score.append(test_recall)

        if epoch % 10 == 0:
            print('Epoch %d | acc_score: %.4f | pre_score:%.4f | re_score:%.4f | Train Loss: %.4f | Test F1: %.4f | Test AUC: %.4f '
                  % (epoch,test_acc,test_pre,test_recall, loss.item(), test_f1, test_auc))

            metrics_res = [epoch_loss_all, acc_score,pre_score, re_score, test_score, auc_score,epoch_times]
            # with open('value_result_cate_' + save_cate + '_result', 'wb') as f:  # save result,if no path,then create
            #     pickle.dump(metrics_res, f)
            #
            # if epoch % 200== 0:
            #     # save the params of net
            #     torch.save(model_GATRE.state_dict(), 'GATRE_params_' + save_cate + '.pkl')
            #     torch.save(model_M.state_dict(), 'mamba_params_' + save_cate + '.pkl')
    print('lr:{}, max(acc_score):{},max(pre_score):{}, max(re_score):{}, max(f1_score):{}, max(auc):{}'
          .format(args.lr, max(acc_score),max(pre_score), max(re_score), max(test_score), max(auc_score)))

    print('last_acc:{}\nlast_pre:{}\nlast_re:{}\nlast_f1:{}\nlast_auc:{}'.format(last_value(acc_score[-10:]),
                                                                    last_value(pre_score[-10:]),
                                                                    last_value(re_score[-10:]),
                                                                    last_value(test_score[-10:]),
                                                                     last_value(auc_score[-10:])))

# -*- coding: utf-8 -*-
import torch
import wandb
torch.set_num_threads(20)
torch.autograd.set_detect_anomaly(True)
import datetime
import os
from arguments import arg_parse

gpu_ids = arg_parse().gpu
gpu_ids = list(map(lambda x: str(x), gpu_ids))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

import torch.nn.functional as F
import os.path as osp
from aug import HealthDataSet as HealthDataset
from torch_geometric.loader import DataLoader
from utils import setup_seed, select_train_valid_test, get_negative_expectation, get_positive_expectation, \
    local_global_loss_, global_global_loss_
import os
import sys
import shutil
import numpy as np
from sklearn.metrics import roc_auc_score, plot_roc_curve, classification_report, average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from model import healthCRL
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-4])
sys.path.append(root_path)


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)
    epochs = args.epochs
    log_interval = 2
    batch_size = args.batch_size
    l2_coef = 0.001
    eps = 1e-8
    lr = args.lr
    DS = args.DS
    tasks_g = ['Mortality', 'Readmin', 'LOS3', 'LOS7']
    tasks_g_nums = len(tasks_g)
    tasks_g_index = []
    exponent_main_g = {}
    pred_index = {}
    key = 0
    if args.is_task_Mortality == 1:
        tasks_g_index.append(0)
        exponent_main_g[0] = args.exponent_main_g_1
        pred_index[0]=key
        key+=1
    if args.is_task_Readmin == 1:
        tasks_g_index.append(1)
        exponent_main_g[1] = args.exponent_main_g_2
        pred_index[1]=key
        key+=1
    if args.is_task_LOS3 == 1:
        tasks_g_index.append(2)
        exponent_main_g[2] = args.exponent_main_g_3
        pred_index[2]=key
        key+=1
    if args.is_task_LOS7 == 1:
        tasks_g_index.append(3)
        exponent_main_g[3] = args.exponent_main_g_4
        pred_index[3]=key
        key+=1

    for i in tasks_g_index:
        print('running task :{}'.format(tasks_g[i]))

    if args.DS == 'eICU':
        args.exponent_main_g_1 = 2.71
        args.exponent_main_g_2 = 3.27
        args.exponent_main_g_3 = 4.05
        args.exponent_main_g_4 = 2.93
        args.exponent_assist_gg = 6.0
        args.exponent_assist_gn = 8.0

    if args.DS == 'mimiciii':

        args.exponent_main_g_1 = 6.0
        args.exponent_main_g_2 = 2.0
        args.exponent_main_g_3 = 5.0
        args.exponent_main_g_4 = 3.0
        args.exponent_assist_gg = 9.0
        args.exponent_assist_gn = 9.0

    task_n = ['NodeClassification']
    file_root = '/home/project/GraphCLHealth/processed_data/' + DS + '/'
    print('file_root. {}'.format(file_root))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset_num_features = 99
    grap_calsses = 2
    node_classes = 0
    if DS == 'eICU':
        node_classes = 49  # 50
    elif DS == 'mimiciii':
        node_classes = 38  #
    elif DS == 'mimiciii_CCU':
        node_classes = 37  #
    elif DS == 'mimiciii_CSRU':
        node_classes = 37  #
    elif DS == 'mimiciii_NWARD':
        node_classes = 30  #

    node_labels = []
    for i in range(node_classes):
        node_labels.append(i)

    # whether need augmentation
    is_contrastive = 1
    if args.is_g_vs_g == 0 and args.is_g_vs_n == 0:
        is_contrastive = 0

    print('is_contrastive {}'.format(is_contrastive))


    model = healthCRL(args.hidden_dim,
                      args.num_gc_layers,
                      args.encoder_model,
                      dataset_num_features,
                      cl_num_g=grap_calsses,
                      cl_num_n=node_classes,
                      data_set=args.DS,
                      is_contrastive=is_contrastive,
                      tasks_g_index=tasks_g_index,
                      num_experts=args.num_experts,
                      is_task_node_cl=args.is_task_NodeClassification,
                      aug1=args.aug1,
                      aug2=args.aug2
                      )
    # print(model)
    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef, eps=eps)

    model.cuda()

    print('================')
    # print('num_features: {}'.format(dataset_num_features))
    # Print arguments
    for k, v in (vars(args).items()):
        print(k, '=', v)
    for i in range(len(tasks_g)):
        print("task {}: {}".format(i, tasks_g[i]))
    print('================')

    num_fold = 3
    # [[], [], [], []]
    list_acc_n = []
    list_auc_roc_n = []
    list_acc_g = [[], [], [], []]
    list_auc_roc_g = [[], [], [], []]
    list_auc_pr_g = [[], [], [], []]
    list_avg_prc_g = [[], [], [], []]
    # for fold_id in [2]:
    for fold_id in range(num_fold):
        wandb.init(
            entity="health",
            project="GraphCLHealth",
            name="MMMGCL",
            tags=["multi-task","fold_id,d%".format(fold_id)],
            notes="this is a training exp",
            config=args
        )

        fold_path_train_processed = file_root + 'fold_' + str(fold_id) + '/train/processed'
        fold_path_test_processed = file_root + 'fold_' + str(fold_id) + '/test/processed'

        if args.is_reinitialize_feature == 1:
            print('deleting exists processed....')
            if os.path.exists(fold_path_train_processed):
                shutil.rmtree(fold_path_train_processed)
            if os.path.exists(fold_path_test_processed):
                shutil.rmtree(fold_path_test_processed)
            print('deleted exists processed....')

        print('====================== num_fold {}  ============================='.format(fold_id))
        print("====================== start at: " + str(datetime.datetime.now()))
        train_fold = file_root + 'fold_' + str(fold_id) + '/train/'
        test_fold = file_root + 'fold_' + str(fold_id) + '/test/'
        train_dataset = HealthDataset(path, name=DS, use_node_attr=args.use_node_attr,
                                      cl_num_g=grap_calsses, cl_num_n=node_classes,
                                      file_root=train_fold).shuffle()
        test_dataset = HealthDataset(path, name=DS, use_node_attr=args.use_node_attr,
                                     cl_num_g=grap_calsses, cl_num_n=node_classes,
                                     file_root=test_fold).shuffle()
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        model.init_emb()

        best_epoch = 0
        best_acc_n = 0
        best_auc_roc_n = 0
        best_acc_g = [0.0 for i in range(tasks_g_nums)]
        best_auc_roc_g = [0.0 for i in range(tasks_g_nums)]
        best_auc_pr_g = [0.0 for i in range(tasks_g_nums)]
        best_avg_prc_g = [0.0 for i in range(tasks_g_nums)]
        best_current_threshold = 0
        cnt_wait = 0

        for epoch in range(1, epochs + 1):
            loss_all = 0
            model.train()
            int_tmp = 0
            current_threshold_epoch = 0
            for data in dataloader:
                optimizer.zero_grad()
                aug_data_list, pred_n, pred_g = model(data.cuda(), args.aug1,
                                                      args.aug2)  # x, edge_index, batch, num_graphs
                loss = 0.0
                # --- task loss graph level
                if len(tasks_g_index) > 0:
                    y_g = torch.LongTensor(data.y.view(-1, 4).cpu().numpy()).cuda()
                    for i in tasks_g_index:
                        y_label_tmp = y_g[:, i:i + 1].reshape(1, -1)[0]
                        loss_g_i = F.nll_loss(pred_g[pred_index[i]], y_label_tmp)
                        # loss += loss_g_i
                        loss += torch.pow(loss_g_i, 1 / exponent_main_g[i])
                # --- task loss node level
                if args.is_task_NodeClassification == 1:
                    y_n = torch.topk(data.node_labels, 1)[1].squeeze(1)
                    loss_task_n = F.cross_entropy(pred_n, y_n)
                    loss += torch.pow(loss_task_n, 1 / args.exponent_main_n)

                # --- contrastive loss
                for i in range(len(aug_data_list)):
                    g1, g2, n1, n2, batch_aug1, batch_aug2 = aug_data_list[i]
                    if args.is_g_vs_g == 1:
                        loss_global_global = model.loss_cal(g1, g2) / len(data)
                        loss += torch.pow(loss_global_global, 1 / args.exponent_assist_gg)
                    if args.is_g_vs_n == 1:
                        loss1 = local_global_loss_(n1, g2, batch_aug1, args.measure)
                        loss2 = local_global_loss_(n2, g1, batch_aug2, args.measure)
                        loss_local_global = (loss1 + loss2) / test_dataset.data.num_nodes
                        loss += torch.pow(loss_local_global, 1 / args.exponent_assist_gn)

                int_tmp += 1
                loss_all += loss
                loss.backward()
                optimizer.step()

            # ===========================================    test   ===========================================
            model.eval()
            graph_embedding = [[], [], [], []]

            current_threshold_epoch = 0
            correct_n = 0
            correct_g = [0 for i in range(tasks_g_nums)]
            y_true_g_list = [[], [], [], []]
            y_true_g_prob_list = [[], [], [], []]
            y_true_n_list = []
            y_true_n_prob_list = []
            for data in dataloader_test:
                with torch.no_grad():
                    # x, x_nodes, y_label = model.get_embeddings(data)
                    pred_n, pred_g, embedding_g = model.get_embeddings(data.cuda())
                    if len(tasks_g_index) > 0:
                        y_g = torch.LongTensor(data.y.view(-1, 4).cpu().numpy()).cuda()
                        for i in tasks_g_index:
                            graph_embedding[i].extend(embedding_g[pred_index[i]].cpu().numpy().tolist())
                            y_pred_g_i = pred_g[pred_index[i]].max(1)[1]  #
                            y_label_g_i = y_g[:, i:i + 1].reshape(1, -1)[0]
                            correct_g[i] += y_pred_g_i.eq(y_label_g_i.view(-1).cuda()).sum().item()
                            y_true_g_list[i].extend(y_g[:, i:i + 1].reshape(1, -1)[0].cpu().numpy())
                            y_true_g_prob_list[i].extend(pred_g[pred_index[i]][:, 1:2].cpu().numpy())
                    if args.is_task_NodeClassification == 1:
                        y_n = torch.topk(data.node_labels, 1)[1].squeeze(1)
                        y_pred_n = pred_n.max(1)[1]
                        correct_n += y_pred_n.eq(y_n.cuda()).sum().item()
                        y_true_n_list.extend(y_n.cpu().numpy())
                        y_true_n_prob_list.extend(pred_n.cpu().numpy())

            # node level task evaluation
            if args.is_task_NodeClassification == 1:
                acc_n = correct_n / test_dataset.data.num_nodes
                auc_roc_n = roc_auc_score(y_true_n_list, y_true_n_prob_list, multi_class='ovo', labels=node_labels)

            # graph level task evaluation
            if len(tasks_g_index) > 0:
                acc_g = [0 for i in range(tasks_g_nums)]
                auc_roc_g = [0 for i in range(tasks_g_nums)]
                auc_pr_g = [0 for i in range(tasks_g_nums)]
                avg_prc_g = [0 for i in range(tasks_g_nums)]
                for i in tasks_g_index:
                    acc_g[i] = correct_g[i] / test_dataset.len()
                    auc_roc_g[i] = roc_auc_score(y_true_g_list[i], y_true_g_prob_list[i])
                    precision_g, recall_g, _ = precision_recall_curve(y_true_g_list[i], y_true_g_prob_list[i])
                    avg_prc_g[i] = average_precision_score(y_true_g_list[i], y_true_g_prob_list[i], average='micro')
                    auc_pr_g[i] = auc(recall_g, precision_g)

                    print('num_fold {}, epoch {}, loss {:.5f}, task {}: acc {:.5f}, auc_roc {:.5f}, avg_prc {:.5f}, '
                          'auc_pr {:.5f} '
                          .format(fold_id, epoch, loss_all, i, acc_g[i], auc_roc_g[i], avg_prc_g[i], auc_pr_g[i]))
                    current_threshold_epoch = current_threshold_epoch + (acc_g[i] + auc_roc_g[i] + auc_pr_g[i])
            if args.is_task_NodeClassification == 1:
                print('num_fold {}, epoch {}, loss {:.5f}, task n: acc {:.5f}, auc_roc {:.5f}'
                      .format(fold_id, epoch, loss_all, acc_n, auc_roc_n))
                current_threshold_epoch = current_threshold_epoch + acc_n + auc_roc_n
            print('------' + str(datetime.datetime.now()) + '------- ')

            if epoch == epochs:
                print('writing embeddings files =============== start')
                for i in tasks_g_index:
                    file_emb = open(file_root + 'embeddings/graph_embeddings_fold_' + str(fold_id) + '_task_' + str(i) + '.txt', 'w')
                    file_y = open(file_root + 'embeddings/graph_y_fold_' + str(fold_id) + '_task_' + str(i) + '.txt', 'w')
                    # for j in range(test_dataset.len()):
                    for j in range(len(graph_embedding[i])):
                        s = to_str(graph_embedding[i][j]) + '\n'  # _node_types
                        s2 = to_str(y_true_g_list[i][j]) + '\n'  # _node_types
                        file_emb.write(s)
                        file_y.write(s2)
                    file_emb.close()
                    file_y.close()
                print('writing embeddings files =============== end')

            if current_threshold_epoch > best_current_threshold:
                best_current_threshold = current_threshold_epoch
                best_epoch = epoch
                for i in tasks_g_index:
                    best_acc_g[i] = acc_g[i]
                    best_auc_roc_g[i] = auc_roc_g[i]
                    best_auc_pr_g[i] = auc_pr_g[i]
                    best_avg_prc_g[i] = avg_prc_g[i]
                if args.is_task_NodeClassification == 1:
                    best_acc_n = acc_n
                    best_auc_roc_n = auc_roc_n

                best_pred_list = y_true_g_list[i]
                # torch.save(model.state_dict(), f'{dataset}-{gpu_ids_str}--{fold_id}--{best_epoch}.pkl')

            wandb.log({
                "loss": loss,
                "acc_g 0": acc_g[0],
                "acc_g 1": acc_g[1],
                "acc_g 2": acc_g[2],
                "acc_g 3": acc_g[3],
                "auc_roc_g 0": auc_roc_g[0],
                "auc_roc_g 1": auc_roc_g[1],
                "auc_roc_g 2": auc_roc_g[2],
                "auc_roc_g 3": auc_roc_g[3],
                "auc_pr_g 0": auc_pr_g[0],
                "auc_pr_g 1": auc_pr_g[1],
                "auc_pr_g 2": auc_pr_g[2],
                "auc_pr_g 3": auc_pr_g[3],
                "avg_prc_g 0": avg_prc_g[0],
                "avg_prc_g 1": avg_prc_g[1],
                "avg_prc_g 2": avg_prc_g[2],
                "avg_prc_g 3": avg_prc_g[3],
            })


        print('')
        print('best result for num_fold.{}, epoch.{}: '.format(fold_id, best_epoch))

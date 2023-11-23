import os
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, plot_roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, auc

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:, 0], x[:, 1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y, search):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    roc_auc = []
    roc_auc_val = []
    pr_auc = []
    pr_auc_val =[]

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.01)

        best_val = 0
        test_acc = None
        for it in range(1000):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
        roc_auc.append(roc_auc_score(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy()))
        precision, recall, _ = precision_recall_curve(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy())
        pr_auc.append(auc(recall, precision))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.01)

        best_val = 0
        test_acc = None
        for it in range(1000):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())
        roc_auc_val.append(roc_auc_score(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy()))
        precision, recall, _ = precision_recall_curve(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy())
        pr_auc_val.append(auc(recall, precision))


    return np.mean(accs_val), np.mean(accs), np.mean(roc_auc_val), np.mean(roc_auc), np.mean(pr_auc_val), np.mean(pr_auc)


def logistic_classify2(x, y, search):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    roc_auc = []
    roc_auc_val = []
    pr_auc = []
    pr_auc_val =[]

    # rus= RandomUnderSampler(random_state=0)
    rus = SMOTE(random_state=0)
    #x_re_sampled, y_re_sampled = rus.fit_resample(x,y)
    #print(sorted(Counter(y_re_sampled).items()))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]
        # 对训练样本进行重采样处理
        train_embs, train_lbls = rus.fit_resample(train_embs, train_lbls)


        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.01)

        best_val = 0
        test_acc = None
        for it in range(1000):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
        roc_auc.append(roc_auc_score(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy()))
        precision, recall, _ = precision_recall_curve(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy())
        pr_auc.append(auc(recall, precision))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]
        train_embs, train_lbls = rus.fit_resample(train_embs, train_lbls)

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.01)

        best_val = 0
        test_acc = None
        for it in range(1000):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())
        roc_auc_val.append(roc_auc_score(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy()))
        precision, recall, _ = precision_recall_curve(test_lbls.cpu().detach().numpy(), logits[:, 0:1].cpu().detach().numpy())
        pr_auc_val.append(auc(recall, precision))


    return np.mean(accs_val), np.mean(accs), np.mean(roc_auc_val), np.mean(roc_auc), np.mean(pr_auc_val), np.mean(pr_auc)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    roc_auc = []
    roc_auc_val = []
    pr_auc = []
    pr_auc_val =[]
    svc_model = 'HalvingGridSearchCV' # GridSearchCV, HalvingGridSearchCV
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            if svc_model == 'GridSearchCV':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                classifier = GridSearchCV(SVC(kernel='linear', probability=True), params, cv=5, scoring='accuracy',
                                          verbose=0)
            elif svc_model == 'HalvingGridSearchCV':
                hyperparameters = {
                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    'kernel': ('linear', 'rbf', 'sigmoid')
                }
                classifier = HalvingGridSearchCV(
                    estimator=SVC(probability=True),
                    param_grid=hyperparameters,
                    scoring='accuracy',
                    cv=5,
                    factor=2,
                    resource='n_samples',
                    min_resources=100,
                    max_resources=170)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        roc_auc.append(roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 0:1]))
        precision, recall,_ = precision_recall_curve(y_test, classifier.predict_proba(x_test)[:, 0:1])
        pr_auc.append(auc(recall,precision))


        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=True).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            if svc_model == 'GridSearchCV':
                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                # classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
                # 如果没有太多的超参数需要调优，并且pipeline运行时间不长，请使用GridSearchCV；
                classifier = GridSearchCV(SVC(kernel='linear', probability=True), params, cv=3, scoring='accuracy', verbose=0, n_jobs=20)
                # 对于较大的搜索空间和训练缓慢的模型，请使用HalvingGridSearchCV； 对于非常大的搜索空间和训练缓慢的模型，请使用 HalvingRandomSearchCV。
            elif svc_model == 'HalvingGridSearchCV':
                hyperparameters = {
                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    'kernel': ('linear', 'rbf', 'sigmoid')
                }
                classifier = HalvingGridSearchCV(
                    estimator=SVC(probability=True),
                    param_grid=hyperparameters,
                    scoring='accuracy',
                    cv=5,
                    factor=2,
                    resource='n_samples',
                    min_resources=100,
                    max_resources=170)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))
        roc_auc_val.append(roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 0:1]))
        precision, recall,_ = precision_recall_curve(y_test, classifier.predict_proba(x_test)[:, 0:1])
        pr_auc_val.append(auc(recall,precision))

    return np.mean(accuracies_val), np.mean(accuracies), np.mean(roc_auc_val), np.mean(roc_auc), np.mean(pr_auc_val),np.mean(pr_auc)


def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    roc_auc = []
    roc_auc_val = []

    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    ret = np.mean(accuracies)
    return np.mean(accuracies_val), ret, np.mean(roc_auc_val), np.mean(roc_auc)


def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    roc_auc = []
    roc_auc_val = []
    pr_auc = []
    pr_auc_val =[]

    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        roc_auc.append(roc_auc_score(y_test,classifier.predict_proba(x_test)[:, 0:1]))
        precision, recall, _ = precision_recall_curve(y_test, classifier.predict_proba(x_test)[:, 0:1])
        pr_auc.append(auc(recall, precision))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))
        roc_auc_val.append(roc_auc_score(y_test,classifier.predict_proba(x_test)[:, 0:1]))
        precision, recall, _ = precision_recall_curve(y_test, classifier.predict_proba(x_test)[:, 0:1])
        pr_auc_val.append(auc(recall, precision))


    return np.mean(accuracies_val), np.mean(accuracies), np.mean(roc_auc_val), np.mean(roc_auc), np.mean(pr_auc_val),np.mean(pr_auc)


def evaluate_embedding(embeddings, labels, evaluate_type, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0
    auc_roc = 0
    auc_roc_val = 0
    auc_pr = 0
    auc_pr_val = 0

    if evaluate_type == 'svc':
        _acc_val, _acc, _auc_roc_val, _auc_roc, _auc_pr_val, _auc_pr = svc_classify(x, y, search)
    if evaluate_type == 'logistic':
        _acc_val, _acc, _auc_roc_val, _auc_roc, _auc_pr_val, _auc_pr  = logistic_classify2(x, y, search)
    if evaluate_type == 'linearsvc':
        _acc_val, _acc, _auc_roc_val, _auc_roc = linearsvc_classify(x, y, search)
    if evaluate_type == 'randomforest':
        _acc_val, _acc, _auc_roc_val, _auc_roc = randomforest_classify(x, y, search)

    acc = _acc
    acc_val = _acc_val
    auc_roc = _auc_roc
    auc_roc_val = _auc_roc_val
    auc_pr = _auc_pr
    auc_pr_val = _auc_pr_val


    # if _acc_val > acc_val:
    #     acc_val = _acc_val
    #     acc = _acc
    #
    # if _auc_val > auc_val:
    #     auc_val = _auc_val
    #     auc = _auc

    """
    _acc_val, _acc, _auc_val, _auc = svc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    if _auc_val > auc_val:
        auc_val = _auc_val
        auc = _auc
    """

    """
    _acc_val, _acc = linearsvc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    """
    '''
    _acc_val, _acc = randomforest_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    # print(acc_val, acc, auc_val, auc)

    return acc_val, acc, auc_roc_val, auc_roc, auc_pr_val, auc_pr


'''
if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
'''

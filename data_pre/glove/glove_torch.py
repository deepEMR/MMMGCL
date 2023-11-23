# -*- coding: utf-8 -*-
import torch, pickle, os, argparse, json
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix

from tools import CorpusPreprocess, VectorEvaluation
from tools import ClinicalDataPreprocess

# get gpu
use_gpu = torch.cuda.is_available()


# calculation weight
def fw(X_c_s, x_max, alpha):
    return (X_c_s / x_max) ** alpha if X_c_s < x_max else 1


class Glove(nn.Module):
    def __init__(self, vocab, args):
        super(Glove, self).__init__()
        # center words weight and biase
        self.c_weight = nn.Embedding(len(vocab), args.embed_dim,
                                     _weight=torch.randn(len(vocab),
                                                         args.embed_dim,
                                                         dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.c_biase = nn.Embedding(len(vocab), 1, _weight=torch.randn(len(vocab),
                                                                       1, dtype=torch.float,
                                                                       requires_grad=True) / 100)

        # surround words weight and biase
        self.s_weight = nn.Embedding(len(vocab), args.embed_dim,
                                     _weight=torch.randn(len(vocab),
                                                         args.embed_dim, dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.s_biase = nn.Embedding(len(vocab), 1,
                                    _weight=torch.randn(len(vocab),
                                                        1, dtype=torch.float,
                                                        requires_grad=True) / 100)

    def forward(self, c, s):
        c_w = self.c_weight(c)
        c_b = self.c_biase(c)
        s_w = self.s_weight(s)
        s_b = self.s_biase(s)
        return torch.sum(c_w.mul(s_w), 1, keepdim=True) + c_b + s_b


# read data
class TrainData(Dataset):
    def __init__(self, coo_matrix, args):
        self.coo_matrix = [((i, j), coo_matrix.data[i][pos]) for i, row in enumerate(coo_matrix.rows) for pos, j in
                           enumerate(row)]
        self.x_max = args.x_max
        self.alpha = args.alpha

    def __len__(self):
        return len(self.coo_matrix)

    def __getitem__(self, idex):
        sample_data = self.coo_matrix[idex]
        sample = {"c": sample_data[0][0],
                  "s": sample_data[0][1],
                  "X_c_s": sample_data[1],
                  "W_c_s": fw(sample_data[1], self.x_max, self.alpha)}
        return sample


def loss_func(X_c_s_hat, X_c_s, W_c_s):
    X_c_s = X_c_s.view(-1, 1)
    W_c_s = X_c_s.view(-1, 1)
    loss = torch.sum(W_c_s.mul((X_c_s_hat - torch.log(X_c_s)) ** 2))
    return loss


# save vector
def save_word_vector(file_name, ix_to_word, glove):
    word_to_embed = {}
    with open(file_name, "w", encoding="utf-8") as f:
        if use_gpu:
            c_vector = glove.c_weight.weight.data.cpu().numpy()
            s_vector = glove.s_weight.weight.data.cpu().numpy()
            vector = c_vector + s_vector
        else:
            c_vector = glove.c_weight.weight.data.numpy()
            s_vector = glove.s_weight.weight.data.numpy()
            vector = c_vector + s_vector

        for i in tqdm(range(len(vector))):
            word = ix_to_word[i]
            s_vec = vector[i]
            s_vec = [str(s) for s in s_vec.tolist()]
            f_vec = vector[i]
            f_vec = [float(s) for s in f_vec.tolist()]
            write_line = str(i) + " " + " ".join(s_vec) + "\n"
            word_to_embed[i] = f_vec
            f.write(write_line)
        print("Glove vector save complete!")
    f.close()
    return word_to_embed


def save_ix_word_map(word_to_ix, ix_to_word, word_to_ix_path_txt, ix_to_word_path_txt):
    with open(word_to_ix_path_txt, "w", encoding="utf-8") as f:
        for i in word_to_ix:
            write_line = str(i) + "|@|" + str(word_to_ix[i]) + "\n"
            f.write(write_line)
        print('word_to_ix_path_txt save complete! ')
    f.close()

    with open(ix_to_word_path_txt, "w", encoding="utf-8") as f:
        for i in ix_to_word:
            write_line = str(i) + "|@|" + str(ix_to_word[i]) + "\n"
            f.write(write_line)
        print('ix_to_word_path_txt save complete! ')
    f.close()


def countCooccurrenceProduct(visit, coMap, word_to_ix):
    codeSet = set(visit)
    for code1 in codeSet:
        for code2 in codeSet:
            if code1 == code2: continue

            product = visit.count(code1) * visit.count(code2)
            key1 = (word_to_ix[code1], word_to_ix[code2])
            key2 = (word_to_ix[code2], word_to_ix[code1])

            if key1 in coMap:
                coMap[key1] += product
            else:
                coMap[key1] = product

            if key2 in coMap:
                coMap[key2] += product
            else:
                coMap[key2] = product


def train(args):
    print('------------------ processing dataset. {}'.format(args.dataset))
    inff = open(args.raw_data_path, 'r')
    patient_dict = {}
    # vocab
    token_text = []

    for line in inff.readlines():
        node_seq = line.split('|@|')
        node_id = node_seq[0]
        node_content = node_seq[1]
        node_type = node_seq[2]
        node_str2int = node_seq[3]
        graph_id = node_seq[4]

        # print(graph_id,node_content.split('|'))
        if int(graph_id) not in patient_dict:
            patient_dict[int(graph_id)] = []
        # print(node_content)
        patient_dict[int(graph_id)].extend(node_content.split('|'))
        # print(patient_dict[int(graph_id)])
    inff.close

    for i in patient_dict.keys():
        token_text.extend(patient_dict[i])
    len_token_text = len(token_text)

    vocab = set(token_text)
    vocab_size = len(vocab)
    # Create word to index and index to word mapping
    word_to_ix = {word: ind for ind, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    coMap = {}
    for i in patient_dict.keys():
        countCooccurrenceProduct(patient_dict[i], coMap, word_to_ix)

    # Construct co-occurence matrix
    co_occ_mat = np.zeros((vocab_size, vocab_size))
    for i in ix_to_word.keys():
        for j in ix_to_word.keys():
            if (i, j) in coMap:
                co_occ_mat[i, j] = coMap[(i, j)]

    # Non-zero co-occurrences
    co_occs = np.transpose(np.nonzero(co_occ_mat))
    # coo_matrix = co_occ_mat
    coo_matrix = lil_matrix(co_occ_mat)

    vocab = word_to_ix
    glove = Glove(vocab, args)

    print(glove)
    if os.path.isfile(args.embed_path_pkl):
        glove.load_state_dict(torch.load(args.embed_path_pkl))
        print('载入模型{}'.format(args.embed_path_pkl))
    if use_gpu:
        gpu = args.gpu
        torch.cuda.set_device(gpu)
        glove.cuda()
    optimizer = torch.optim.Adam(glove.parameters(), lr=args.learning_rate)

    train_data = TrainData(coo_matrix, args)
    data_loader = DataLoader(train_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    steps = 0
    loss_epochs = []
    for epoch in range(args.epoches):
        print("currently epoch is {}, all epoch is {}".format(epoch + 1, args.epoches))
        avg_epoch_loss = 0
        for i, batch_data in enumerate(data_loader):
            total_loss = 0
            c = batch_data['c']
            s = batch_data['s']
            X_c_s = batch_data['X_c_s']
            W_c_s = batch_data["W_c_s"]

            if use_gpu:
                c = c.cuda()
                s = s.cuda()
                X_c_s = X_c_s.cuda()
                W_c_s = W_c_s.cuda()

            W_c_s_hat = glove(c, s)
            loss = loss_func(W_c_s_hat, X_c_s, W_c_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_epoch_loss += loss / len(train_data)
            total_loss += loss.item()
            if steps % 1000 == 0:
                print("Steps {}, loss is {}".format(steps, loss.item()))
            steps += 1
        print("Epoches {}, complete!, avg loss {}.\n".format(epoch + 1, avg_epoch_loss))
        loss_epochs.append(avg_epoch_loss.cpu())
    word_to_embed = save_word_vector(args.embed_path_txt, ix_to_word, glove)
    save_graph_node_attri(args, word_to_embed, word_to_ix, args.embed_dim)
    save_ix_word_map(word_to_ix, ix_to_word, args.word_to_ix_path_txt, args.ix_to_word_path_txt)
    torch.save(glove.state_dict(), args.embed_path_pkl)

    # Visualize embeddings
    if args.embed_dim == 2:
        # Pick some random words
        word_inds = np.random.choice(np.arange(len(vocab)), size=20, replace=False)
        for word_ind in word_inds:
            w_embed = word_to_embed[word_ind]
            x, y = w_embed[0], w_embed[1]
            plt.scatter(x, y)
            plt.annotate(ix_to_word[word_ind], xy=(x, y), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom')
            # print(ix_to_word[word_ind])
        plt.show()

    plot_loss_fn(loss_epochs, "GloVe loss function")


def save_graph_node_attri(args, word_to_embed, word_to_ix, EMBEDDING_SIZE):
    # 获取没一个节点的embed
    infile = args.raw_data_path
    inff = open(infile, 'r')
    node_attribute = {}
    words_embeds = []
    for line in inff.readlines():
        node_seq = line.split('|@|')
        node_id = node_seq[0]
        node_content = node_seq[1]
        node_type = node_seq[2]
        node_str2int = node_seq[3]
        graph_id = node_seq[4]

        words_in_node = node_content.split('|')
        words_embed = []
        for word in words_in_node:
            w_embed = word_to_embed[word_to_ix[word]]
            words_embed.extend(w_embed)
        node_attribute[node_id] = words_embed
        words_embeds.append(words_embed)
    inff.close
    max_len = max([len(a)] for a in words_embeds)
    for key in node_attribute:
        node_attribute[key] = np.hstack((node_attribute[key], np.zeros(max_len[0] - len(node_attribute[key]))))

    outfile = args.out_put_path
    with open(outfile, 'w') as file:
        for key in node_attribute.keys():
            # print(to_str(node_attribute[key]))
            file.write(to_str(node_attribute[key]))
            file.write('\n')
        print('graph_node_attri save completed')
    file.close


# Plot loss fn
def plot_loss_fn(losses, title):
    for i in range(len(losses)):
        losses[i] = losses[i].detach().numpy()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.figure()
    plt.show()


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


if __name__ == "__main__":
    # file_path
    parser = argparse.ArgumentParser(description='generate word2vec by gensim')
    # mimiciii, eICU, mimiciii_CCU, mimiciii_CSRU,mimiciii_NWARD
    parser.add_argument('--dataset', type=str, default='mimiciii_CCU')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--raw_data_path', type=str, default='', help='the dir of raw data file,in this dir can contain more than one files')
    parser.add_argument('--out_put_path', type=str, default='', help='the path of train data file')
    parser.add_argument('--embed_path_txt', type=str, default="", help='the save path of word2vec with type txt')
    parser.add_argument('--embed_path_pkl', type=str, default="", help='the save path of word2vec with type pkl,which is array after pickle.load ')
    parser.add_argument('--word_to_ix_path_txt', type=str, default="", help='the save path of word_to_ix')
    parser.add_argument('--ix_to_word_path_txt', type=str, default="", help='the save path of ix_to_word')
    parser.add_argument('--vocab_path', type=str, default='', help='the save path of vocab')
    parser.add_argument('--embed_dim', type=int, default=16, help='the dim of word2vec')
    parser.add_argument('--x_max', type=int, default=100, help='')
    parser.add_argument('--alpha', type=float, default=0.75, help='')
    parser.add_argument('--epoches', type=int, default=20, help='epoches')
    parser.add_argument('--min_count', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--windows_size', type=int, default=5, help='windows_size')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='learning_rate')

    args = parser.parse_args()
    dataset = args.dataset
    args.raw_data_path = '/home/project/GraphCLHealth/processed_data/'+dataset+'/full/raw/' + dataset + '_node_seqs.txt'
    args.out_put_path = '/home/project/GraphCLHealth/processed_data/'+dataset+'/full/raw/' + dataset + '_node_attributes.txt'
    args.embed_path_txt = 'export/' + dataset + '/Vector.txt'
    args.embed_path_pkl = 'export/' + dataset + '/Vector.pkl'
    args.word_to_ix_path_txt = 'export/' + dataset + '/word_to_ix.txt'
    args.ix_to_word_path_txt = 'export/' + dataset + '/ix_to_word.txt'
    args.vocab_path='export/' + dataset + '/vocab.json'

    # Print arguments
    for k, v in (vars(args).items()):
        print(k, '=', v)
    train(args)

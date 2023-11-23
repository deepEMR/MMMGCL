import numpy
import random
import json, sys, os
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

def init_node_content_dict(data_set):
    raw_data_path = '/home/project/GraphCLHealth/processed_data/' + data_set + '/full/raw/' + data_set + '_node_seqs.txt'
    i = 0
    inff = open(raw_data_path, 'r')
    node_content_dict = {}

    for line in inff.readlines():
        node_seq = line.split('|@|')
        node_id = node_seq[0]
        node_content = node_seq[1]
        node_type = node_seq[2]
        node_str2int = node_seq[3]
        graph_id = node_seq[4]

        tmp = node_content.split('|')
        if node_content not in node_content_dict:
            node_content_dict[node_content] = []
            node_content_dict[node_content].append(node_type)
            node_content_dict[node_content].append(int(1))
            node_content_dict[node_content].append(int(len(tmp)))
            for nn in tmp:
                node_content_dict[node_content].append(nn)
        else:
            node_content_dict[node_content][1] += 1
        i += 1
    inff.close
    return node_content_dict


def init_node_list_dict(raw_data_path):
    raw_data_path = '/home/project/GraphCLHealth/processed_data/' + data_set + '/full/raw/' + data_set + '_node_seqs.txt'
    i = 0
    inff = open(raw_data_path, 'r')
    node_list = {}

    for line in inff.readlines():
        node_seq = line.split('|@|')
        node_id = node_seq[0]
        node_content = node_seq[1]
        node_type = node_seq[2]
        node_str2int = node_seq[3]
        graph_id = node_seq[4]

        node_list[node_id] = []
        node_list[node_id].append(node_content)
        node_list[node_id].append(node_type)
        node_list[node_id].append(node_str2int)
        node_list[node_id].append(graph_id)
    inff.close
    return node_list


def init_word_to_ix(data_set):
    raw_data_path = '/home/project/GraphCLHealth/data_pre/glove/export/' + data_set + '/word_to_ix.txt'
    inff = open(raw_data_path, 'r')
    word_to_ix = {}
    for line in inff.readlines():
        line_seq = line.split('|@|')
        word_to_ix[line_seq[0]] = int(line_seq[1].replace('\n', ''))
    inff.close
    return word_to_ix


def init_ix_to_word(data_set):
    raw_data_path = '/home/project/GraphCLHealth/data_pre/glove/export/' + data_set + '/ix_to_word.txt'

    inff = open(raw_data_path, 'r')
    ix_to_word = {}
    for line in inff.readlines():
        line_seq = line.split('|@|')
        ix_to_word[int(line_seq[0])] = line_seq[1]
    inff.close
    return ix_to_word


def init_get_vector(data_set):
    raw_data_path = '/home/project/GraphCLHealth/data_pre/glove/export/' + data_set + '/Vector.txt'
    inff = open(raw_data_path, 'r')
    vector = {}
    for line in inff.readlines():
        a = '[' + line.replace(' ', ',') + ']'
        a = json.loads(a)
        a = numpy.array(a)
        id = int(a[0:1])
        # vector[id]=a[1:]
        # print((a[1:].tolist())) # ndarray to list
        vector[id] = (a[1:]).astype(numpy.float32)
        # if((a[1:]==node_emb[0]).all()):
        #    print('sss')
    inff.close
    return vector


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


def get_parent_embedding(data_set):
    node_content_dict = init_node_content_dict(data_set)
    node_list_dict = init_node_list_dict(data_set)
    ix_to_word = init_ix_to_word(data_set)
    word_to_ix = init_word_to_ix(data_set)

    vector = init_get_vector(data_set)
    max_len = 6
    max_dim = 16  # glove 训练时的hidden_dim

    node_content_same_parent_dict = {}
    node_attribute = {}
    count = 0
    for key in node_list_dict.keys():
        if count % 1000 == 0:
            print(count)
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        # get embedding of each node
        node_embedding = []
        node_id_list = []
        tmp = node_list_dict[key][0].split('|')
        for i in range(len(tmp)):
            node_embedding.append(vector[word_to_ix[tmp[i]]])
            node_id_list.append(word_to_ix[tmp[i]])
        if len(node_embedding) < max_len:
            for i in range(max_len - len(node_embedding)):
                node_embedding.append(np.zeros(max_dim))
        node_attribute[key] = node_embedding

        str_list = node_list_dict[key][0].replace('\n', '').split('|')
        len_str = len(str_list)

        # 选择具有相同父节点的属性集合作为候选的list
        node_candidate = []
        for key_list, _ in node_content_dict.items():
            if node_content_dict[key_list][2] - len_str >= 0:
                flag = 1
                for i in range(len_str - 1):
                    if node_content_dict[key_list][3 + i] != str_list[i]:
                        flag = flag * 0
                if flag == 1:
                    node_candidate.append(key_list)

        select_id = random.randrange(0, len(node_candidate), 1)
        select_node_str = node_candidate[select_id]
        # 得到候选的embedding
        select_node_list = select_node_str.split('|')
        substitute_node_emb = []
        for att in select_node_list:
            substitute_node_emb.append(vector[word_to_ix[str(att)]])

        if len(node_embedding) < max_len:
            for i in range(max_len - len(node_embedding)):
                node_embedding.append(np.zeros(max_dim))

        if len(substitute_node_emb) < max_len:
            for i in range(max_len - len(substitute_node_emb)):
                substitute_node_emb.append(np.zeros(max_dim))
        node_content_same_parent_dict[key] = ([node_embedding, substitute_node_emb])
        # node_content_same_parent_dict[node_embedding] = substitute_node_emb

        count += 1

        # if count==100:
        #     break

    outfile = '/home/project/GraphCLHealth/processed_data/' + data_set + '/full/raw/' + data_set + '_random_parent_node_seqs.txt'
    with open(outfile, 'w') as file:
        for key in node_content_same_parent_dict.keys():
            # print(to_str(node_attribute[key]))
            file.write(to_str(node_content_same_parent_dict[key][0])+'|@|'+to_str(node_content_same_parent_dict[key][1]))
            file.write('\n')
        print('graph_node_attri save completed')
    file.close

    return node_content_same_parent_dict


data_set = 'mimiciii'
node_content_same_parent_dict = get_parent_embedding(data_set)

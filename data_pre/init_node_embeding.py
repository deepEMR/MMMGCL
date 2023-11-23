import sys, os
import numpy as np
import shutil
import numpy
import json


def init_node_content_dict(raw_data_path):
    raw_data_path
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


def init_word_to_ix(raw_data_path):
    inff = open(raw_data_path, 'r')
    word_to_ix = {}
    for line in inff.readlines():
        line_seq = line.split('|@|')
        word_to_ix[line_seq[0]] = int(line_seq[1].replace('\n', ''))
    inff.close
    return word_to_ix


def init_ix_to_word(raw_data_path):
    inff = open(raw_data_path, 'r')
    ix_to_word = {}
    for line in inff.readlines():
        line_seq = line.split('|@|')
        ix_to_word[int(line_seq[0])] = line_seq[1]
    inff.close
    return ix_to_word


def init_get_vector(raw_data_path):
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


def generate_attr(base_dir, vector, ix_to_word, word_to_ix, data_set):
    path_node_seqs = base_dir + '/' + data_set + '_node_seqs.txt'
    outfile = base_dir + '/' + data_set + '_node_attributes.txt'
    outfile_node_label = base_dir + '/' + data_set + '_node_attributes.txt'

    node_content_dict = init_node_content_dict(path_node_seqs)
    max_len = 6
    max_dim = 16  # glove 训练时的hidden_dim
    node_embedding_list = []
    node_attribute = {}
    for key in node_content_dict.keys():
        node_embedding = []
        tmp = node_content_dict[key][0].split('|')
        for i in range(len(tmp)):
            node_embedding.append(vector[word_to_ix[tmp[i]]])
        if len(node_embedding) < max_len:
            for i in range(max_len - len(node_embedding)):
                node_embedding.append(np.zeros(max_dim))

        node_attribute[key] = node_embedding

    with open(outfile, 'w') as file:
        for key in node_attribute.keys():
            # print(to_str(node_attribute[key]))
            file.write(to_str(node_attribute[key]))
            file.write('\n')
        print('graph_node_attri save completed')
    file.close


def main(argv):
    # 'eICU', 'mimiciii','mimiciii_CCU' 'mimiciii_CSRU' 'mimiciii_NWARD'
    data_set = 'mimiciii_NWARD'
    print('processing dataset: {}'.format(data_set))
    root_data_dir = '/home/project/GraphCLHealth/processed_data/' + data_set + '/'
    path_word_to_ix = '/home/project/GraphCLHealth/data_pre/glove/export/' + data_set + '/word_to_ix.txt'
    path_ix_to_word = '/home/project/GraphCLHealth/data_pre/glove/export/' + data_set + '/ix_to_word.txt'
    path_vector = '/home/project/GraphCLHealth/data_pre/glove/export/' + data_set + '/Vector.txt'

    print(root_data_dir)

    vector = init_get_vector(path_vector)
    ix_to_word = init_ix_to_word(path_ix_to_word)
    word_to_ix = init_word_to_ix(path_word_to_ix)

    num_folds = 3
    for i in range(num_folds):
        base_dir_train = root_data_dir + 'fold_' + str(i) + '/train/raw'
        base_dir_test = root_data_dir + 'fold_' + str(i) + '/test/raw'

        print('processing fold.{} >'.format(i))
        generate_attr(base_dir_train, vector, ix_to_word, word_to_ix, data_set)
        generate_attr(base_dir_test, vector, ix_to_word, word_to_ix, data_set)

        fold_path_train_processed = root_data_dir + 'fold_' + str(i) + '/train/processed'
        fold_path_test_processed = root_data_dir + 'fold_' + str(i) + '/test/processed'

        if os.path.exists(fold_path_train_processed):
            shutil.rmtree(fold_path_train_processed)
        if os.path.exists(fold_path_test_processed):
            shutil.rmtree(fold_path_test_processed)

    print('processing fold.full >')
    fold_path_full_processed = root_data_dir + '/full/processed'
    if os.path.exists(fold_path_full_processed):
        shutil.rmtree(fold_path_full_processed)
    # generate_attr(root_data_dir+'full/raw', vector, ix_to_word, word_to_ix,data_set)

    print('processed')


"""
    此方法用glove训练的 medical word embedding 来初始化 diagnosis codes 和 procedure codes
    对于lab，eICU中共计158个不同的lab name，采用word2vec进行embedding
    对于micro，有2个不同organism，有20个不同 culturesite
"""
if __name__ == '__main__':
    main(sys.argv)

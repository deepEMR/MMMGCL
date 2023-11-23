# -*- coding: utf-8 -*-
import _pickle as cPickle
import csv
import os
import sys
import datetime
import random
import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from icd_mapping import get_diagnosisstring_code_dict_eicu

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)


class EncounterInfo(object):

    def __init__(self, patient_id, encounter_id, encounter_timestamp, expired,
                 readmission, los_3day, los_7day, admission_type=99, marital_status=99):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.encounter_timestamp = encounter_timestamp
        self.expired = expired
        self.readmission = readmission
        self.admission_type = admission_type
        self.marital_status = marital_status
        self.los_3day = los_3day
        self.los_7day = los_7day
        self.gender = ''
        self.dx_ids = []
        self.dx_names = []
        self.dx_labels = []
        self.rx_ids = []
        self.rx_names = []
        self.lab_ids = []
        self.lab_names = []
        self.microbiology_ids = []
        self.microbiology_names = []
        self.physicals = []
        self.procedures_ids = []
        self.procedures_names = []


def minNums(startTime, endTime):
    '''计算两个时间点之间的分钟数'''
    # 处理格式,加上秒位
    startTime1 = startTime  # + ':00'
    endTime1 = endTime  # + ':00'
    startTime2 = datetime.datetime.strptime(startTime1, "%Y-%m-%d %H:%M:%S")
    endTime2 = datetime.datetime.strptime(endTime1, "%Y-%m-%d %H:%M:%S")
    seconds = (endTime2 - startTime2).seconds
    total_seconds = (endTime2 - startTime2).total_seconds()
    mins = total_seconds / 60
    return int(mins)


def process_patient(infile, encounter_dict, max_patient_num=1000, hour_threshold=12):
    inff = open(infile, 'r')
    count = 0
    patient_dict = {}
    admission_list = []

    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        patient_id = line['patienthealthsystemstayid']
        encounter_id = line['patientunitstayid']
        encounter_timestamp = -int(line['hospitaladmitoffset'])
        unit_type = line['unittype']

        age = line['age']
        if age == '> 89':
            age = 89
        if age == '':
            age = 0
        age = int(age)
        # hospitaladmitoffset：number of minutes from unit admit time that the patient was admitted to the hospital
        if age >= 18 and unit_type == 'MICU':
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            patient_dict[patient_id].append((encounter_timestamp, encounter_id))
    inff.close()

    # 随机的数量不能大于总的患者数量
    if max_patient_num > len(patient_dict):
        max_patient_num = len(patient_dict)
    # 随机选取 患者
    patient_random_keys = random.sample(patient_dict.keys(), max_patient_num)
    patient_random_del_keys = []

    for patient_id in patient_dict.keys():
        if patient_id not in patient_random_keys:
            patient_random_del_keys.append(patient_id)
    # 删除不在随机范围内的患者记录
    for patient_random_del_key in patient_random_del_keys:
        del patient_dict[patient_random_del_key]
    # admission_list，只存储随机到的患者的就诊记录
    for patient_id, encounter_ids in patient_dict.items():
        for encounter_id in encounter_ids:
            if encounter_id[1] not in admission_list:
                admission_list.append(encounter_id[1])

    # sort by time_enc_tuples
    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples)

    enc_readmission_dict = {}
    for patient_id, time_enc_tuples in patient_dict_sorted.items():
        for time_enc_tuple in time_enc_tuples[:-1]:
            enc_id = time_enc_tuple[1]
            enc_readmission_dict[enc_id] = True
        last_enc_id = time_enc_tuples[-1][1]
        enc_readmission_dict[last_enc_id] = False

    inff = open(infile, 'r')
    count = 0
    accept_patient = []
    for line in csv.DictReader(inff):
        if line['patientunitstayid'] in admission_list:
            if count % 1000 == 0:
                sys.stdout.write('%d\r' % count)
                sys.stdout.flush()

            patient_id = line['patienthealthsystemstayid']
            encounter_id = line['patientunitstayid']
            encounter_timestamp = -int(line['hospitaladmitoffset'])
            discharge_status = line['unitdischargestatus']
            duration_minute = float(line['unitdischargeoffset'])
            expired = True if discharge_status == 'Expired' else False
            readmission = enc_readmission_dict[encounter_id]
            losday = float(line['unitdischargeoffset']) / (24. * 60.)
            los_3day = True if losday > 3 else False
            los_7day = True if losday > 7 else False

            if duration_minute < 60. * hour_threshold:
                continue

            ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired,
                               readmission, los_3day, los_7day)
            if encounter_id in encounter_dict:
                print('Duplicate encounter ID!!')
                sys.exit(0)
            encounter_dict[encounter_id] = ei
            if patient_id not in accept_patient:
                accept_patient.append(patient_id)
            count += 1
    inff.close()
    print('Accepted patients: %d' % len(accept_patient))
    print('Accepted admissions: %d' % count)
    print('')
    return encounter_dict, admission_list


# 处理入院诊断
def process_admission_dx(infile, encounter_dict, admission_list):
    inff = open(infile, 'r')
    count = 0
    count_accept = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        if line['patientunitstayid'] in admission_list:
            encounter_id = line['patientunitstayid']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue

            dx_id = line['admitdxpath'].lower()
            encounter_dict[encounter_id].dx_ids.append(dx_id)
            count_accept += 1
        count += 1
    inff.close()
    print('')
    print('Admission Diagnosis without Encounter ID: %d' % missing_eid)
    print('Accepted Admission Diagnosis: %d' % count_accept)

    return encounter_dict


# 处理诊断信息
def process_diagnosis(infile, encounter_dict, admission_list):
    inff = open(infile, 'r')
    count = 0
    count_accept = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        if line['patientunitstayid'] in admission_list:
            encounter_id = line['patientunitstayid']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue

            dx_id = line['diagnosisstring'].lower()

            if len(set(encounter_dict[encounter_id].dx_ids)) <= 50:
                encounter_dict[encounter_id].dx_ids.append(dx_id)
                count_accept += 1
        count += 1
    inff.close()
    print('')
    print('Diagnosis without Encounter ID: %d' % missing_eid)
    print('Accepted Diagnosis: %d' % count_accept)

    return encounter_dict


def process_treatment(infile, encounter_dict, admission_list):
    inff = open(infile, 'r')
    count = 0
    count_accept = 0
    missing_eid = 0

    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        if line['patientunitstayid'] in admission_list:
            # 每处理10000行输出一次记录
            encounter_id = line['patientunitstayid']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            procedures_id = line['treatmentstring'].lower()
            if len(set(encounter_dict[encounter_id].procedures_ids)) <= 50:
                encounter_dict[encounter_id].procedures_ids.append(procedures_id)
                count_accept += 1
        count += 1
    inff.close()
    print('')
    print('Treatment without Encounter ID: %d' % missing_eid)
    print('Accepted treatments: %d' % count_accept)

    return encounter_dict


def process_lab(infile, encounter_dict, admission_list):
    inff = open(infile, 'r')
    count = 0
    count_accept = 0
    missing_eid = 0

    # chemistry class type
    # # 1 for chemistry, 2 for drug level, 3 for hemo, 4 for misc, 5 for non-mapped, 6 for sensitive, 7 for ABG lab
    lab_type_dic = {}
    lab_type_dic['1'] = 'chemistry'
    lab_type_dic['2'] = 'drug level'
    lab_type_dic['3'] = 'hemo'
    lab_type_dic['4'] = 'misc'
    lab_type_dic['5'] = 'non-mapped'
    lab_type_dic['6'] = 'sensitive'
    lab_type_dic['7'] = 'ABG lab'

    for line in csv.DictReader(inff):
        # 每处理10000行输出一次记录
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        if line['patientunitstayid'] in admission_list:
            encounter_id = line['patientunitstayid']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue

            labname = line['labname']
            labresult = line['labresult']
            labtypeid = line['labtypeid']

            if len(set(encounter_dict[encounter_id].lab_ids)) <= 50:
                encounter_dict[encounter_id].lab_ids.append(lab_type_dic[labtypeid] + '|' + labname)
                count_accept += 1
        count += 1
    inff.close()
    print('')
    print('Labs without Encounter ID: %d' % missing_eid)
    print('Accepted labs: %d' % count_accept)
    return encounter_dict


def generate_graph_files(output_path, encounter_dict, diagnosisstring_code_dict, level1_name):
    enc_dict = encounter_dict
    skip_duplicate = False
    min_num_codes = 1
    max_num_codes = 50
    num_duplicate = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    min_lab_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    max_lab_cut = 0

    visit_ids = []
    node_str2int = {}

    for _, enc in enc_dict.items():
        if skip_duplicate:
            if (len(enc.dx_ids) > len(set(enc.dx_ids))
                    or len(enc.procedures_ids) > len(set(enc.procedures_ids))
                    or len(enc.lab_ids) > len(set(enc.lab_ids))):
                num_duplicate += 1
                continue

        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue

        if len(set(enc.procedures_ids)) < min_num_codes:
            min_treatment_cut += 1
            continue

        if len(set(enc.lab_ids)) < min_num_codes:
            min_lab_cut += 1
            continue

        # 诊断的数量大于50的记录舍弃
        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue

        # 治疗记录的数量大于50的记录舍弃
        if len(set(enc.procedures_ids)) > max_num_codes:
            max_treatment_cut += 1
            continue

        # 治疗记录的数量大于50的记录舍弃
        if len(set(enc.lab_ids)) > max_num_codes:
            max_lab_cut += 1
            continue

        #  ----- labels：node_str2int： node_labels

        for dx_id in enc.dx_ids:
            if dx_id + '|dx' not in node_str2int:
                node_str2int[dx_id + '|dx'] = len(node_str2int)
        # print('dx 节点的 label编码从 0 到'+ str(max_dx_node_label))

        for procedures_id in enc.procedures_ids:
            if procedures_id + '|proc' not in node_str2int:
                node_str2int[procedures_id + '|proc'] = len(node_str2int)

        for labs_id in enc.lab_ids:
            if labs_id + '|lab' not in node_str2int:
                node_str2int[labs_id + '|lab'] = len(node_str2int)

        visit_ids.append(enc.encounter_id)

    node_labels_arg = []
    node_str2int_arg = {}
    for key in node_str2int.keys():
        tt = ''
        if key.find('|dx') > 0 and key[:key.index('|dx')] in diagnosisstring_code_dict.keys():
            tmp = key.split('|')
            tt = level1_name[diagnosisstring_code_dict[key[:key.index('|dx')]]] + '|' + 'dx'
        else:
            tmp = key.split('|')
            tt = tmp[0] + '|' + tmp[len(tmp) - 1]
        node_labels_arg.append(tt)

    node_vocab = set(node_labels_arg)
    vocab_size = len(node_vocab)
    node_to_ix = {word: ind for ind, word in enumerate(node_vocab)}
    ix_to_node = {i: word for i, word in enumerate(node_vocab)}

    print('nums of total node classes. {}'.format(len(node_to_ix)))

    for key in node_str2int.keys():
        tt = ''
        if key.find('|dx') > 0 and key[:key.index('|dx')] in diagnosisstring_code_dict.keys():
            tmp = key.split('|')
            tt = level1_name[diagnosisstring_code_dict[key[:key.index('|dx')]]] + '|' + 'dx'
        else:
            tmp = key.split('|')
            tt = tmp[0] + '|' + tmp[len(tmp) - 1]
        node_str2int_arg[key] = node_to_ix[tt]

    num_fold = 3
    for i in range(num_fold):
        fold_path = output_path + '/fold_' + str(i)
        fold_path_train = fold_path + '/train/raw'
        # fold_path_val = fold_path + '/val/raw'
        fold_path_test = fold_path + '/test/raw'

        if not os.path.exists(fold_path_train):
            os.makedirs(fold_path_train)
        # if not os.path.exists(fold_path_val):
        #     os.makedirs(fold_path_val)
        if not os.path.exists(fold_path_test):
            os.makedirs(fold_path_test)

        key_train, key_test = ms.train_test_split(visit_ids, test_size=0.3, random_state=i)
        save_processed_files_by_folds(key_train, enc_dict, fold_path_train, node_str2int_arg)
        save_processed_files_by_folds(key_test, enc_dict, fold_path_test, node_str2int_arg)

    fold_path_full = output_path + '/full/raw'
    if not os.path.exists(fold_path_full):
        os.makedirs(fold_path_full)
    save_processed_files_by_folds(visit_ids, enc_dict, fold_path_full, node_str2int_arg)

    print('accept visits.{}'.format(len(enc_dict)))


def save_processed_files_by_folds(key_list, enc_dict, output_path, node_str2int):
    graph_label = []
    node_attribute = []
    node_handled = {}

    graph_indicator = []
    adjency = []
    count = 0
    num_dx_ids = 0
    num_procedures = 0
    num_unique_dx_ids = 0
    num_unique_procedures = 0

    # 定义int 节点type类型
    note_type_dx = 1
    note_type_proc = 2
    note_type_lab = 3
    note_type_micro = 4

    graph_id = 1
    node_id = 1
    for _, enc in enc_dict.items():
        if enc.encounter_id in key_list:
            count += 1
            num_dx_ids += len(enc.dx_ids)
            num_procedures += len(enc.procedures_ids)
            num_unique_dx_ids += len(set(enc.dx_ids))
            num_unique_procedures += len(set(enc.procedures_ids))
            graph_label.append((graph_id, enc.encounter_id, enc.expired, enc.readmission, enc.los_3day, enc.los_7day))

            # ----------------------------------------------dx_ints ----------------------------------------------
            # note_type '1000'
            for i, dx_id in enumerate(set(enc.dx_ids)):
                if enc.encounter_id + ':dx:' + str(dx_id) not in node_handled:
                    node_handled[enc.encounter_id + ':dx:' + str(dx_id)] = node_id
                node_attribute.append((node_id, dx_id, 'dx', node_str2int[dx_id + '|dx'], graph_id,
                                       note_type_dx))
                graph_indicator.append((node_id, graph_id))
                node_id = node_id + 1

            # ----------------------------------------------procedures_ids ----------------------------------------------
            for i, procedures_id in enumerate(set(enc.procedures_ids)):
                if enc.encounter_id + ':proc:' + str(procedures_id) not in node_handled:
                    node_handled[enc.encounter_id + ':proc:' + str(procedures_id)] = node_id
                node_attribute.append(
                    (node_id, procedures_id, 'proc', node_str2int[procedures_id + '|proc'], graph_id, note_type_proc))
                graph_indicator.append((node_id, graph_id))
                node_id = node_id + 1

            # ----------------------------------------------lab_ids ----------------------------------------------
            for i, lab_id in enumerate(set(enc.lab_ids)):
                if enc.encounter_id + ':lab:' + str(lab_id) not in node_handled:
                    node_handled[enc.encounter_id + ':lab:' + str(lab_id)] = node_id
                node_attribute.append(
                    (node_id, lab_id, 'lab', node_str2int[lab_id + '|lab'], graph_id, note_type_lab))
                graph_indicator.append((node_id, graph_id))
                node_id = node_id + 1


            for i, dx_id in enumerate(set(enc.dx_ids)):
                for j, procedures_id in enumerate(set(enc.procedures_ids)):
                    for k, lab_id in enumerate(set(enc.lab_ids)):
                        adjency.append((node_handled[enc.encounter_id + ':dx:' + str(dx_id)],
                                        node_handled[enc.encounter_id + ':proc:' + str(procedures_id)]
                                        ))
                        adjency.append((node_handled[enc.encounter_id + ':proc:' + str(procedures_id)],
                                        node_handled[enc.encounter_id + ':dx:' + str(dx_id)]
                                        ))
                        adjency.append((node_handled[enc.encounter_id + ':proc:' + str(procedures_id)],
                                        node_handled[enc.encounter_id + ':lab:' + str(lab_id)]
                                        ))
                        adjency.append((node_handled[enc.encounter_id + ':lab:' + str(lab_id)],
                                        node_handled[enc.encounter_id + ':proc:' + str(procedures_id)]
                                        ))

            graph_id = graph_id + 1

            ###
    print('Saving {}/#Data_set#_A.txt'.format(output_path))
    file = open(output_path + '/eICU_A.txt', 'w')
    for i in range(len(adjency)):
        s = str(adjency[i][0]) + ',' + str(adjency[i][1]) + '\n'
        file.write(s)
    file.close()

    print('Saving {}/#Data_set#_graph_indicator.txt'.format(output_path))
    file = open(output_path + '/eICU_graph_indicator.txt', 'w')
    for i in range(len(graph_indicator)):
        s = str(graph_indicator[i][1]) + '\n'
        file.write(s)
    file.close()

    print('Saving {}/#Data_set#_graph_labels.txt'.format(output_path))
    file_lable1 = open(output_path + '/eICU_graph_labels.txt', 'w')
    for i in range(len(graph_label)):
        g_label = str(int(graph_label[i][2])) + ',' + \
                  str(int(graph_label[i][3])) + ',' + \
                  str(int(graph_label[i][4])) + ',' + \
                  str(int(graph_label[i][5])) + '\n'
        file_lable1.write(g_label)
    file_lable1.close()

    print('Saving {}/#Data_set#_node_attribute.txt'.format(output_path))
    file = open(output_path + '/eICU_node_types.txt', 'w')
    file3 = open(output_path + '/eICU_node_labels.txt', 'w')
    file2 = open(output_path + '/eICU_node_seqs.txt', 'w')

    for i in range(len(node_attribute)):
        s = str(to_str(node_attribute[i][5])) + '\n'  # _node_types
        s3 = str(to_str(node_attribute[i][3])) + '\n'  # _node_labels
        s2 = str(
            str(node_attribute[i][0]) + '|@|' + str(node_attribute[i][1]) + '|@|' + str(
                node_attribute[i][2]) + '|@|' +
            str(node_attribute[i][3])) + '|@|' + str(node_attribute[i][4]) + '\n'
        file.write(s)
        file3.write(s3)
        file2.write(s2)
    file.close()
    file3.close()
    file2.close()

    print('Saving {}/#Data_set#_node_str2int.txt'.format(output_path))
    file = open(output_path + '/node_str2int.txt', 'w')
    for key in node_str2int.keys():
        s = str(key) + str(node_str2int[key]) + '\n'
        file.write(s)
    file.close()


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


def delete_bare_code(encounter_dict, minimum_cnt=10):
    enc_dict = encounter_dict
    dx_list = []
    proc_list = []
    lab_list = []

    dx_node_deleted = 0
    proc_node_deleted = 0
    lab_node_deleted = 0
    for _, enc in enc_dict.items():
        for dx_id in enc.dx_ids:
            dx_list.append(dx_id)

        for proc_id in enc.procedures_ids:
            proc_list.append(proc_id)

        for lab_id in enc.lab_ids:
            lab_list.append(lab_id)

    dx_count = {}
    proc_count = {}
    lab_count = {}
    for dx_id in set(dx_list):
        dx_count[dx_id] = dx_list.count(dx_id)
    for proc_id in set(proc_list):
        proc_count[proc_id] = proc_list.count(proc_id)
    for lab_id in set(lab_list):
        lab_count[lab_id] = lab_list.count(lab_id)

    for _, enc in enc_dict.items():
        for dx_id in enc.dx_ids:
            if dx_count[dx_id] < minimum_cnt:
                enc.dx_ids.remove(dx_id)
                dx_node_deleted += 1

        for proc_id in enc.procedures_ids:
            if proc_count[proc_id] < minimum_cnt:
                enc.procedures_ids.remove(proc_id)
                proc_node_deleted += 1

        for lab_id in enc.lab_ids:
            if lab_count[lab_id] < minimum_cnt:
                enc.lab_ids.remove(lab_id)
                lab_node_deleted += 1

    print('')
    print('node removed more than {} times'.format(minimum_cnt))
    print('dx {}, proc {}, lab {}'.format(dx_node_deleted, proc_node_deleted, lab_node_deleted))

    return enc_dict


def main(argv):

    input_path = '/home/project/GraphCLHealth/data/eICU/'
    output_path = '/home/project/GraphCLHealth/processed_data/eICU'

    max_patient_num = 500000000
    print('max_patient_num:' + str(max_patient_num))
    minimum_cnt = 5
    flag_test_flag = 1  # whether to use the test files for debugging

    if flag_test_flag == 0:
        patient_file = input_path + '/patient10.csv'
        admission_dx_file = input_path + '/admissionDx10.csv'
        diagnosis_file = input_path + '/diagnosis10.csv'
        treatment_file = input_path + '/treatment10.csv'
        lab_file = input_path + '/lab10.csv'
        microlab_file = input_path + '/microlab10.csv'
        ccs_icd_file = input_path + 'ccs_multi_dx_tool_2015.csv'
        icd_10to9_file = input_path + 'icd10cmtoicd9gem.csv'
    else:
        patient_file = input_path + '/patient.csv'
        admission_dx_file = input_path + '/admissionDx.csv'
        diagnosis_file = input_path + '/diagnosis.csv'
        treatment_file = input_path + '/treatment.csv'
        lab_file = input_path + '/lab.csv'
        microlab_file = input_path + '/microlab.csv'
        ccs_icd_file = input_path + 'ccs_multi_dx_tool_2015.csv'
        icd_10to9_file = input_path + 'icd10cmtoicd9gem.csv'

    encounter_dict = {}
    print('Processing patient.csv')
    encounter_dict, admission_list = process_patient(
        patient_file, encounter_dict, max_patient_num, hour_threshold=12)
    print('Processing diagnosis.csv')
    encounter_dict = process_diagnosis(diagnosis_file, encounter_dict, admission_list)
    print('Processing treatment.csv')
    encounter_dict = process_treatment(treatment_file, encounter_dict, admission_list)
    print('Processing lab.csv')
    encounter_dict = process_lab(lab_file, encounter_dict, admission_list)

    print('Processing icd mapping dict')
    diagnosisstring_code_dict, level1_name = get_diagnosisstring_code_dict_eicu(ccs_icd_file, diagnosis_file,
                                                                                icd_10to9_file)

    generate_graph_files(output_path, encounter_dict, diagnosisstring_code_dict, level1_name)


if __name__ == '__main__':
    main(sys.argv)

import _pickle as cPickle
import csv
import os
import sys
import datetime
import random
import pandas as pd
import sklearn.model_selection as ms
from create_dataset import create_MIMIC_dataset

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
        self.dx_ids_lvl1 = []
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


# random_seed
def process_admission(admission_file, icustays_file, patients_file, encounter_dict, transfer_file, careunit, max_patient_num=500, hour_threshold=24):

    count = 0
    patient_dict = {}
    admission_list = []

    patient_base_dict = {}
    inff = open(patients_file, 'r')
    for line in csv.DictReader(inff):
        patient_id = line['SUBJECT_ID']
        gender = line['GENDER']
        birthday = line['DOB']
        if patient_id not in patient_base_dict:
            patient_base_dict[patient_id] = []
        patient_base_dict[patient_id].append((gender,birthday))
    inff.close()

    # --------------- transfer_file --------------------
    patient_trans_dict = {}
    inff = open(transfer_file, 'r')
    for line in csv.DictReader(inff):
        patient_id = line['SUBJECT_ID']
        encounter_id = line['HADM_ID']
        curr_careunit = line['CURR_CAREUNIT']
        intime = line['INTIME']
        birthday = patient_base_dict[patient_id][0][1]
        # print(birthday,intime)
        if intime!='':
            age = minNums(birthday, intime) / (24. * 60 * 365)
        else:
            age=0

        if curr_careunit == careunit and careunit != 'NWARD' and age > 18:
            if patient_id not in patient_trans_dict:
                patient_trans_dict[patient_id] = []
            patient_trans_dict[patient_id].append(encounter_id)

        if curr_careunit == careunit and careunit == 'NWARD' and age <= 18:
            if patient_id not in patient_trans_dict:
                patient_trans_dict[patient_id] = []
            patient_trans_dict[patient_id].append(encounter_id)

    inff.close()


    # --------------- admission_file --------------------
    inff = open(admission_file, 'r')
    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        # if count == max_admission_num:
        #     break

        patient_id = line['SUBJECT_ID']
        encounter_id = line['HADM_ID']
        admittime = line['ADMITTIME']
        dischtime = line['DISCHTIME']

        # encounter_timestamp = -int(line['hospitaladmitoffset'])
        encounter_timestamp = minNums(admittime, dischtime)
        # encounter_timestamp：number of minutes from unit admit time that the patient was admitted to the hospital

        if patient_id in patient_trans_dict.keys():
            if encounter_id in patient_trans_dict[patient_id]:
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                patient_dict[patient_id].append((admittime, encounter_timestamp, encounter_id))

        count += 1
    inff.close()

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
            if encounter_id[2] not in admission_list:
                admission_list.append(encounter_id[2])

    # sort
    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        # print(time_enc_tuples)
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples, reverse=False)

    enc_readmission_dict = {}
    for patient_id, time_enc_tuples in patient_dict_sorted.items():
        for time_enc_tuple in time_enc_tuples[:-1]:
            enc_id = time_enc_tuple[2]
            enc_readmission_dict[enc_id] = True
        last_enc_id = time_enc_tuples[-1][2]
        enc_readmission_dict[last_enc_id] = False

    inff = open(admission_file, 'r')
    count = 0
    for line in csv.DictReader(inff):
        if line['HADM_ID'] in admission_list:
            patient_id = line['SUBJECT_ID']
            encounter_id = line['HADM_ID']

            admittime = line['ADMITTIME']
            dischtime = line['DISCHTIME']

            encounter_timestamp = minNums(admittime, dischtime)

            hospital_expire_flag = line['HOSPITAL_EXPIRE_FLAG']
            duration_minute = encounter_timestamp
            losday =  duration_minute/(24. * 60)
            expired = True if hospital_expire_flag == '1' else False
            readmission = enc_readmission_dict[encounter_id]
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
            count = count + 1
    inff.close()

    print('Accepted Patients: {}'.format(max_patient_num))
    print('Accepted admissions: {}'.format(count))
    print('')
    return encounter_dict, admission_list


def process_patients(patients_file, encounter_dict):
    count = 0
    enc_dict = encounter_dict
    inff = open(patients_file, 'r')
    for line in csv.DictReader(inff):
        patient_id = line['SUBJECT_ID']
        gender = line['GENDER']
        for _, enc in enc_dict.items():
            if enc.patient_id == patient_id:
                enc.gender = gender
            count += 1
    inff.close()

    print('Accepted admissions: %d' % count)
    print('')
    return encounter_dict


def process_diagnosis_dx(infile, icdfile, encounter_dict, admission_list, dx_level, dx_level1_name):
    count = 0
    dx_map = {}
    icd_diagnoses = {}
    icd_dx_name_to_id = {}
    inff_d = open(icdfile, 'r')
    for line in csv.DictReader(inff_d):

        icd_code = line['ICD9_CODE']
        long_title = line['LONG_TITLE']
        if icd_code not in icd_diagnoses:
            icd_diagnoses[icd_code] = []
        icd_diagnoses[icd_code] = long_title
        if long_title not in icd_dx_name_to_id:
            icd_dx_name_to_id[long_title]=icd_code
    inff_d.close()

    inff = open(infile, 'r')
    count = 0
    count_accept = 0

    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        if line['HADM_ID'] in admission_list:
            encounter_id = line['HADM_ID']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue

            dx_id = line['ICD9_CODE']
            if len(set(encounter_dict[encounter_id].dx_ids)) <= 50:
                encounter_dict[encounter_id].dx_ids.append(dx_id)
                if dx_id in icd_diagnoses:
                    dx_name_string = icd_diagnoses[dx_id]
                    level1_name = dx_level[dx_id][1]
                    level2_name = dx_level[dx_id][3]
                    level3_name = dx_level[dx_id][5]
                    level4_name = dx_level[dx_id][7]
                    if len(level4_name.strip())>0:
                        dx_name_string = level4_name + '|' + dx_name_string
                    if len(level3_name.strip())>0:
                        dx_name_string = level3_name + '|' + dx_name_string
                    if len(level2_name.strip())>0:
                        dx_name_string = level2_name + '|' + dx_name_string
                    if len(level1_name.strip())>0:
                        dx_name_string = level1_name + '|' + dx_name_string
                    encounter_dict[encounter_id].dx_names.append(dx_name_string)
                    if dx_name_string not in dx_map:
                        dx_map[dx_name_string]=[dx_id, dx_level[dx_id][0], dx_level[dx_id][1]]
                count_accept += 1
            count += 1
    inff.close()
    print('Diagnosis without Encounter ID: %d' % missing_eid)
    print('Accepted Diagnosis: %d' % count_accept)
    print('')
    return encounter_dict, icd_dx_name_to_id, dx_map


def process_procedures(infile, icdfile, encounter_dict, admission_list, pr_level, pr_level1_name):
    count = 0
    proc_map = {}
    icd_procedures = {}
    icd_proc_name_to_id = {}
    inff_d = open(icdfile, 'r')
    for line in csv.DictReader(inff_d):
        icd_code = line['ICD9_CODE']
        long_title = line['LONG_TITLE']
        if icd_code not in icd_procedures:
            icd_procedures[icd_code] = []
        icd_procedures[icd_code] = long_title
        if long_title not in icd_proc_name_to_id:
            icd_proc_name_to_id[long_title] = icd_code
    inff_d.close()

    inff = open(infile, 'r')
    count = 0
    count_accept = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 1000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        if line['HADM_ID'] in admission_list:

            encounter_id = line['HADM_ID']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue

            procedure_id = line['ICD9_CODE']

            if procedure_id in icd_procedures.keys():
                if len(set(encounter_dict[encounter_id].procedures_ids)) <= 50:
                    encounter_dict[encounter_id].procedures_ids.append(procedure_id)
                    pr_name_string = icd_procedures[procedure_id]
                    level1_name = pr_level[procedure_id][1]
                    level2_name = pr_level[procedure_id][3]
                    level3_name = pr_level[procedure_id][5]
                    if len(level3_name.strip()) > 0:
                        pr_name_string = level3_name + '|' + pr_name_string
                    if len(level2_name.strip()) > 0:
                        pr_name_string = level2_name + '|' + pr_name_string
                    if len(level1_name.strip()) > 0:
                        pr_name_string = level1_name + '|' + pr_name_string
                    encounter_dict[encounter_id].procedures_names.append(pr_name_string)
                    if pr_name_string not in proc_map:
                        proc_map[pr_name_string] = [procedure_id, pr_level[procedure_id][0], pr_level[procedure_id][1]]
                    count_accept += 1
            count += 1
    inff.close()

    print('Treatment without Encounter ID: %d' % missing_eid)
    print('Accepted treatments: %d' % count_accept)
    print('')
    return encounter_dict, icd_proc_name_to_id, proc_map


def process_labevents(labevents_file, icdfile, encounter_dict, admission_list):
    count = 0
    icd_labItems = {}
    lab_category = []
    inff_d = open(icdfile, 'r')
    for line in csv.DictReader(inff_d):
        itemid = line['ITEMID']
        label = line['LABEL'].lower()
        fluid = line['FLUID'].lower()
        category = line['CATEGORY'].lower()
        loinc_code = line['LOINC_CODE']
        lab_category.append(category)
        icd_key = itemid
        if icd_key not in icd_labItems:
            icd_labItems[icd_key] = []
        icd_labItems[icd_key] = [label, category, fluid, loinc_code]
    inff_d.close()

    lab_category = set(lab_category)
    lab_type_dic = {word: ind for ind, word in enumerate(lab_category)}

    # {'BLOOD GAS', 'HEMATOLOGY', 'Hematology', 'Chemistry', 'Blood Gas', 'CHEMISTRY'}
    #
    # {'BLOOD GAS': 0,
    #  'HEMATOLOGY': 1,
    #  'Hematology': 2,
    #  'Chemistry': 3,
    #  'Blood Gas': 4,
    #  'CHEMISTRY': 5}

    count = 0
    missing_eid = 0
    count_lab = 0
    inff = open(labevents_file, 'r')
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        if line['HADM_ID'] in admission_list:
            encounter_id = line['HADM_ID']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue

            itemid = line['ITEMID']
            flag = line['FLAG']

            if len(set(encounter_dict[encounter_id].lab_ids)) <= 100:
                encounter_dict[encounter_id].lab_ids.append(itemid)
                if itemid in icd_labItems.keys():
                    encounter_dict[encounter_id].lab_names.append(str(icd_labItems[itemid][1])+'|'+ icd_labItems[itemid][2]+'|'+icd_labItems[itemid][0])
                count += 1
        count_lab += 1

    inff.close()
    print('Labitems without Encounter ID: %d' % missing_eid)
    print('Total lab events: %d' % count_lab)
    print('Accepted lab events: %d' % count)
    print('')

    return encounter_dict


def add_sparse_prior_guide_dp(encounter_dict):
    return encounter_dict


def generate_graph_files(output_path, encounter_dict, dx_level, pr_level, dx_level1_name, pr_level1_name, dx_map, proc_map, careunit):
    enc_dict = encounter_dict
    skip_duplicate = False
    min_num_codes = 1
    max_num_codes = 50

    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_procedures = 0
    num_unique_dx_ids = 0
    num_unique_procedures = 0

    min_dx_cut = 0
    min_treatment_cut = 0
    min_lab_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    max_lab_cut = 0

    graph_label = []
    node_attribute = []
    node_handled = {}
    graph_id = 1
    node_id = 1
    graph_indicator = []
    adjency = []

    visit_ids = []
    node_str2int = {}

    for _, enc in enc_dict.items():
        if skip_duplicate:
            # 舍弃存在重复诊断和治疗的就诊记录
            if (len(enc.dx_names) > len(set(enc.dx_names))
                    or len(enc.procedures_names) > len(set(enc.procedures_names))
                    or len(enc.lab_names) > len(set(enc.lab_names))):
                num_duplicate += 1
                continue

        if len(set(enc.dx_names)) < min_num_codes:
            min_dx_cut += 1
            continue

        if len(set(enc.procedures_names)) < min_num_codes:
            min_treatment_cut += 1
            continue

        if len(set(enc.lab_names)) < min_num_codes:
            min_lab_cut += 1
            continue

        # 诊断的数量大于50的记录舍弃
        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue

        # 治疗记录的数量大于50的记录舍弃
        if len(set(enc.procedures_names)) > max_num_codes:
            max_treatment_cut += 1
            continue

        # 检验结果数量大于50的记录舍弃
        if len(set(enc.lab_names)) > max_num_codes*2:
            max_lab_cut += 1
            continue

        #  ----- 生成节点的 labels：node_str2int： node_labels
        for dx_id in enc.dx_names:
            if dx_id + '|dx' not in node_str2int:
                node_str2int[dx_id + '|dx'] = len(node_str2int)
        # print('dx 节点的 label编码从 0 到'+ str(max_dx_node_label))
        for procedures_id in enc.procedures_names:
            if procedures_id + '|proc' not in node_str2int:
                node_str2int[procedures_id + '|proc'] = len(node_str2int)
        # print('proc 节点的 label编码从 '+ str(min_proc_node_label) +' 到'+ str(max_proc_node_label))
        for labevents_id in enc.lab_names:
            if labevents_id + '|lab' not in node_str2int:
                node_str2int[labevents_id + '|lab'] = len(node_str2int)

        visit_ids.append(enc.encounter_id)

    node_labels_arg = []
    node_str2int_arg = {}
    for key in node_str2int.keys():
        tt = ''
        if key.find('|dx') > 0 and key[:key.index('|dx')] in dx_map.keys():
            tmp = key.split('|')
            tt = dx_map[key[:key.index('|dx')]][2] + '|' + 'dx'
        elif key.find('|proc') > 0 and key[:key.index('|proc')] in proc_map.keys():
            tmp = key.split('|')
            tt = proc_map[key[:key.index('|proc')]][2] + '|' + 'proc'
        elif key.find('|lab') > 0:
            tmp = key.split('|')
            tt = tmp[0] + '|' + tmp[len(tmp) - 1]
        else:
            continue
        node_labels_arg.append(tt)

    node_vocab = set(node_labels_arg)
    vocab_size = len(node_vocab)
    node_to_ix = {word: ind for ind, word in enumerate(node_vocab)}
    ix_to_node = {i: word for i, word in enumerate(node_vocab)}

    print('nums of total node classes. {}'.format(len(node_to_ix)))

    for key in node_str2int.keys():
        tt = ''
        if key.find('|dx') > 0 and key[:key.index('|dx')] in dx_map.keys():
            tmp = key.split('|')
            tt = dx_map[key[:key.index('|dx')]][2] + '|' + 'dx'
        elif key.find('|proc') > 0 and key[:key.index('|proc')] in proc_map.keys():
            tmp = key.split('|')
            tt = proc_map[key[:key.index('|proc')]][2] + '|' + 'proc'
        else:
            tmp = key.split('|')
            tt = tmp[0] + '|' + tmp[len(tmp) - 1]

        node_str2int_arg[key] = node_to_ix[tt]

    num_fold = 3
    for i in range(num_fold):
        fold_path = output_path + '/fold_' + str(i)
        fold_path_train = fold_path + '/train/raw'
        fold_path_val = fold_path + '/val/raw'
        fold_path_test = fold_path + '/test/raw'

        if not os.path.exists(fold_path_train):
            os.makedirs(fold_path_train)

        if not os.path.exists(fold_path_test):
            os.makedirs(fold_path_test)

        key_train, key_test = ms.train_test_split(visit_ids, test_size=0.3, random_state=i)
        save_processed_files_by_folds(key_train, enc_dict, fold_path_train, node_str2int_arg,careunit)
        save_processed_files_by_folds(key_test, enc_dict, fold_path_test, node_str2int_arg,careunit)

    fold_path_full = output_path + '/full/raw'
    if not os.path.exists(fold_path_full):
        os.makedirs(fold_path_full)
    save_processed_files_by_folds(visit_ids, enc_dict, fold_path_full, node_str2int_arg,careunit)

    print('accept visits.{}'.format(len(enc_dict)))
    print('num_duplicate visits'.format(num_duplicate))
    print('visit_ids.{}'.format(len(visit_ids)))
    print('min_dx_cut.{}'.format(min_dx_cut))
    print('min_treatment_cut.{}'.format(min_treatment_cut))
    print('min_lab_cut.{}'.format(min_lab_cut))
    print('max_dx_cut.{}'.format(max_dx_cut))
    print('max_treatment_cut.{}'.format(max_treatment_cut))
    print('max_lab_cut.{}'.format(max_lab_cut))

def save_processed_files_by_folds(key_list, enc_dict, output_path, node_str2int,careunit):
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
            graph_label.append((graph_id, enc.encounter_id, enc.expired, enc.readmission, enc.los_3day, enc.los_7day, enc.dx_ids))

            # ----------------------------------------------dx_ints ----------------------------------------------
            # note_type '1000'
            for i, dx_id in enumerate(set(enc.dx_names)):
                if enc.encounter_id + ':dx:' + str(dx_id) not in node_handled:
                    node_handled[enc.encounter_id + ':dx:' + str(dx_id)] = node_id
                node_attribute.append((node_id, dx_id, 'dx', node_str2int[dx_id + '|dx'], graph_id,
                                       note_type_dx))  #
                graph_indicator.append((node_id, graph_id))
                node_id = node_id + 1

            # ----------------------------------------------procedures_ids ----------------------------------------------
            for i, procedures_id in enumerate(set(enc.procedures_names)):
                if enc.encounter_id + ':proc:' + str(procedures_id) not in node_handled:
                    node_handled[enc.encounter_id + ':proc:' + str(procedures_id)] = node_id
                node_attribute.append(
                    (node_id, procedures_id, 'proc', node_str2int[procedures_id + '|proc'], graph_id, note_type_proc))
                graph_indicator.append((node_id, graph_id))
                node_id = node_id + 1

            # ----------------------------------------------lab_ids ----------------------------------------------
            for i, lab_id in enumerate(set(enc.lab_names)):
                if enc.encounter_id + ':lab:' + str(lab_id) not in node_handled:
                    node_handled[enc.encounter_id + ':lab:' + str(lab_id)] = node_id
                node_attribute.append(
                    (node_id, lab_id, 'lab', node_str2int[lab_id + '|lab'], graph_id, note_type_lab))
                graph_indicator.append((node_id, graph_id))
                node_id = node_id + 1

            # --------------------------------------------------------------------------------------------
            for i, dx_id in enumerate(set(enc.dx_names)):
                for j, procedures_id in enumerate(set(enc.procedures_names)):
                    for k, lab_id in enumerate(set(enc.lab_names)):
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
    file = open(output_path + '/mimiciii_'+careunit+'_A.txt', 'w')
    for i in range(len(adjency)):
        s = str(adjency[i][0]) + ',' + str(adjency[i][1]) + '\n'
        file.write(s)
    file.close()

    print('Saving {}/#Data_set#_graph_indicator.txt'.format(output_path))
    file = open(output_path + '/mimiciii_'+careunit+'_graph_indicator.txt', 'w')
    for i in range(len(graph_indicator)):
        s = str(graph_indicator[i][1]) + '\n'
        file.write(s)
    file.close()



    print('Saving {}/#Data_set#_graph_labels.txt'.format(output_path))
    file_lable1 = open(output_path + '/mimiciii_'+careunit+'_graph_labels.txt', 'w')
    for i in range(len(graph_label)):
        g_label = str(int(graph_label[i][2])) + ',' + \
                  str(int(graph_label[i][3])) + ',' + \
                  str(int(graph_label[i][4])) + ',' + \
                  str(int(graph_label[i][5])) + '\n'
        file_lable1.write(g_label)
    file_lable1.close()

    print('Saving {}/#Data_set#_node_attribute.txt'.format(output_path))
    file = open(output_path + '/mimiciii_'+careunit+'_node_types.txt', 'w')
    file3 = open(output_path + '/mimiciii_'+careunit+'_node_labels.txt', 'w')
    file2 = open(output_path + '/mimiciii_'+careunit+'_node_seqs.txt', 'w')

    for i in range(len(node_attribute)):
        s = str((node_attribute[i][5])) + '\n'  # _node_types
        s3 = str((node_attribute[i][3])) + '\n'  # _node_labels
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


# borrowed from https://github.com/hoon9405/DescEmb
def get_icd_code_dict_mimiciii():

    ccs_dx = pd.read_csv('/home/project/GraphCLHealth/data/mimiciii/ccs_multi_dx_tool_2015.csv')
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    ccs_dx["'CCS LVL 1 LABEL'"] = ccs_dx["'CCS LVL 1 LABEL'"]
    ccs_dx["'CCS LVL 2'"] = ccs_dx["'CCS LVL 2'"].str[1:].str[:-1]
    ccs_dx["'CCS LVL 2 LABEL'"] = ccs_dx["'CCS LVL 2 LABEL'"]
    ccs_dx["'CCS LVL 3'"] = ccs_dx["'CCS LVL 3'"].str[1:].str[:-1]
    ccs_dx["'CCS LVL 3 LABEL'"] = ccs_dx["'CCS LVL 3 LABEL'"]
    ccs_dx["'CCS LVL 4'"] = ccs_dx["'CCS LVL 4'"].str[1:].str[:-1]
    ccs_dx["'CCS LVL 4 LABEL'"] = ccs_dx["'CCS LVL 4 LABEL'"]
    dx_level = {}
    for x, y1, y1_name, y2, y2_name, y3, y3_name, y4, y4_name \
            in zip(ccs_dx["'ICD-9-CM CODE'"],
                   ccs_dx["'CCS LVL 1'"], ccs_dx["'CCS LVL 1 LABEL'"],
                   ccs_dx["'CCS LVL 2'"], ccs_dx["'CCS LVL 2 LABEL'"],
                   ccs_dx["'CCS LVL 3'"], ccs_dx["'CCS LVL 3 LABEL'"],
                   ccs_dx["'CCS LVL 4'"], ccs_dx["'CCS LVL 4 LABEL'"]):
        dx_level[x] = [y1, y1_name, y2, y2_name, y3, y3_name, y4, y4_name]

    dx_level1_name = {}
    for x, y in zip(ccs_dx["'CCS LVL 1'"], ccs_dx["'CCS LVL 1 LABEL'"]):
        dx_level1_name[x] = y

    ccs_pr = pd.read_csv('/home/project/GraphCLHealth/data/mimiciii/ccs_multi_pr_tool_2015.csv')
    ccs_pr["'ICD-9-CM CODE'"] = ccs_pr["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_pr["'CCS LVL 1'"] = ccs_pr["'CCS LVL 1'"].str[1:].str[:-1]
    ccs_pr["'CCS LVL 1 LABEL'"] = ccs_pr["'CCS LVL 1 LABEL'"]
    ccs_pr["'CCS LVL 2'"] = ccs_pr["'CCS LVL 2'"].str[1:].str[:-1]
    ccs_pr["'CCS LVL 2 LABEL'"] = ccs_pr["'CCS LVL 2 LABEL'"]
    ccs_pr["'CCS LVL 3'"] = ccs_pr["'CCS LVL 3'"].str[1:].str[:-1]
    ccs_pr["'CCS LVL 3 LABEL'"] = ccs_pr["'CCS LVL 3 LABEL'"]
    pr_level = {}
    for x, y1, y1_name, y2, y2_name, y3, y3_name \
            in zip(ccs_pr["'ICD-9-CM CODE'"],
                   ccs_pr["'CCS LVL 1'"], ccs_pr["'CCS LVL 1 LABEL'"],
                   ccs_pr["'CCS LVL 2'"], ccs_pr["'CCS LVL 2 LABEL'"],
                   ccs_pr["'CCS LVL 3'"], ccs_pr["'CCS LVL 3 LABEL'"]):
        pr_level[x] = [y1, y1_name, y2, y2_name, y3, y3_name]

    pr_level1_name = {}
    for x, y in zip(ccs_pr["'CCS LVL 1'"], ccs_pr["'CCS LVL 1 LABEL'"]):
        pr_level1_name[x] = y

    return dx_level, pr_level, dx_level1_name, pr_level1_name


def delete_bare_code(encounter_dict, minimum_cnt=10):
    enc_dict = encounter_dict
    dx_list = []
    proc_list = []
    lab_list = []

    dx_node_deleted = 0
    proc_node_deleted = 0
    lab_node_deleted = 0
    for _, enc in enc_dict.items():
        for dx_id in enc.dx_names:
            dx_list.append(dx_id)

        for proc_id in enc.procedures_names:
            proc_list.append(proc_id)

        for lab_id in enc.lab_names:
            lab_list.append(lab_id)

    dx_count={}
    proc_count={}
    lab_count={}
    for dx_id in set(dx_list):
        dx_count[dx_id] = dx_list.count(dx_id)
    for proc_id in set(proc_list):
        proc_count[proc_id] = proc_list.count(proc_id)
    for lab_id in set(lab_list):
        lab_count[lab_id] = lab_list.count(lab_id)

    for _, enc in enc_dict.items():
        for dx_id in enc.dx_names:
            if dx_count[dx_id]<minimum_cnt:
                enc.dx_names.remove(dx_id)
                dx_node_deleted +=1

        for proc_id in enc.procedures_names:
            if proc_count[proc_id] < minimum_cnt:
                enc.procedures_names.remove(proc_id)
                proc_node_deleted +=1

        for lab_id in enc.lab_names:
            if lab_count[lab_id] < minimum_cnt:
                enc.lab_names.remove(lab_id)
                lab_node_deleted +=1

    print('')
    print('node removed more than {} times'.format(minimum_cnt))
    print('dx {}, proc {}, lab {}'.format(dx_node_deleted, proc_node_deleted, lab_node_deleted))

    return enc_dict

def main(argv):
    careunit = 'CCU'
    print('processing careunit:'+careunit)
    input_path = '/home/project/GraphCLHealth/data/mimiciii'
    output_path = '/home/project/GraphCLHealth/processed_data/mimiciii_'+careunit+'/'

    minimum_cnt = 5
    max_patient_num = 50000000
    print('max_patient_num:' + str(max_patient_num))


    flag_test_flag = 1 # whether to use the test files for debugging
    icd_diagnoses_file = input_path + '/D_ICD_DIAGNOSES.csv'
    icd_procedures_file = input_path + '/D_ICD_PROCEDURES.csv'
    icd_labItems_file = input_path + '/D_LABITEMS.csv'
    transfer_file = input_path + '/TRANSFERS.csv'


    if flag_test_flag == 0:
        admission_file = input_path + '/ADMISSIONS10.csv'
        diagnosis_file = input_path + '/DIAGNOSES_ICD10.csv'
        procedures_file = input_path + '/PROCEDURES_ICD10.csv'
        labevents_file = input_path + '/LABEVENTS10.csv'
        microbiology_file = input_path + '/microbiologyevents10.csv'
        patients_file = input_path + '/PATIENTS.csv'
        icustays_file = input_path + '/ICUSTAYS10.csv'
    else:
        admission_file = input_path + '/ADMISSIONS.csv'
        diagnosis_file = input_path + '/DIAGNOSES_ICD.csv'
        procedures_file = input_path + '/PROCEDURES_ICD.csv'
        labevents_file = input_path + '/LABEVENTS.csv'
        microbiology_file = input_path + '/microbiologyevents.csv'
        patients_file = input_path + '/PATIENTS.csv'
        icustays_file = input_path + '/ICUSTAYS.csv'
    # 调试用的文件

    encounter_dict = {}

    print('Processing icd mapping dict')
    dx_level, pr_level, dx_level1_name, pr_level1_name = get_icd_code_dict_mimiciii()

    print('Processing ADMISSIONS.csv')
    encounter_dict, admission_list = process_admission(admission_file, icustays_file, patients_file, encounter_dict, transfer_file, careunit, max_patient_num, hour_threshold=24)

    print('Processing DIAGNOSES_ICD.csv')
    encounter_dict, icd_dx_name_to_id, dx_map = process_diagnosis_dx(diagnosis_file, icd_diagnoses_file, encounter_dict, admission_list, dx_level, dx_level1_name)

    print('Processing PROCEDURES_ICD.csv')
    encounter_dict, icd_proc_name_to_id, proc_map = process_procedures(procedures_file, icd_procedures_file, encounter_dict, admission_list, pr_level, pr_level1_name)

    print('Processing LABEVENTS.csv')
    encounter_dict = process_labevents(labevents_file, icd_labItems_file, encounter_dict, admission_list)

    generate_graph_files(output_path, encounter_dict, dx_level, pr_level, dx_level1_name, pr_level1_name, dx_map, proc_map,careunit)


if __name__ == '__main__':
    main(sys.argv)

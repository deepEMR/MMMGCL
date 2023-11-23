import pandas as pd
from collections import Counter


# borrowed from https://github.com/hoon9405/DescEmb
def get_diagnosisstring_code_dict_eicu(ccs_icd_file, diagnosis_file, icd_10to9_file):
    ccs_dx = pd.read_csv(ccs_icd_file)
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    ccs_dx["'CCS LVL 1 LABEL'"] = ccs_dx["'CCS LVL 1 LABEL'"].str[1:].str[:-1]
    level1 = {}
    for x, y in zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]):
        level1[x] = y
    level1_name = {}
    for x, y in zip(ccs_dx["'CCS LVL 1'"], ccs_dx["'CCS LVL 1 LABEL'"]):
        level1_name[x] = y

    eicu_cohort = pd.read_csv(diagnosis_file)

    eicu_dx_df = eicu_cohort.dropna(subset=['diagnosisstring']).copy().reset_index(drop=True)
    eicu_diagnosis_list = []
    for x in eicu_dx_df['diagnosisstring']:
        eicu_diagnosis_list.append(str(x))
    eicu_dx_unique = list(set(eicu_diagnosis_list))
    eicu_dx = pd.read_csv(diagnosis_file)

    # eicu_dx all diagnosis status
    eicu_dx_list = list(eicu_dx['icd9code'].values)
    eicu_dx_list = [x for x in eicu_dx_list if x != 'nan' and type(x) != float]
    eicu_dx_list = [y.strip().replace('.', '') for x in eicu_dx_list for y in x.split(',')]
    eicu_ids = list(eicu_dx_df['patientunitstayid'].values)

    # drop the icd9code NaN for right now
    eicu_dx = eicu_dx.dropna(subset=['icd9code']).copy().reset_index(drop=True)

    # make diagnosisstring - ICD9 code dictionary
    diagnosisstring_code_dict = {}
    key_error_list = []

    for index, row in eicu_dx.iterrows():
        diagnosis_string = row['diagnosisstring']
        icd9code = row['icd9code']
        icd9code = icd9code.split(',')[0].replace('.', '')
        try:
            eicu_level1 = level1[icd9code]
            diagnosisstring_code_dict[diagnosis_string] = eicu_level1
        except KeyError:
            key_error_list.append(diagnosis_string)

            # Check key error list
    key_error_list = list(set(key_error_list))
    print('Number of diagnosis with only ICD 10 code: {}'.format(len(key_error_list)))

    # icd10 to icd9 mapping csv file
    icd10_icd9 = pd.read_csv(icd_10to9_file)

    # make icd10 - icd9 dictionary
    icd10_icd9_dict = {}
    for x, y in zip(icd10_icd9['icd10cm'], icd10_icd9['icd9cm']):
        icd10_icd9_dict[x] = y

    # map icd10 to icd9 code
    two_icd10_code_list = []
    icd10_key_error_list = []
    for i in range(len(key_error_list)):
        icd10code = eicu_dx[eicu_dx['diagnosisstring'] == key_error_list[i]]['icd9code'].values[0].split(',')
        if len(icd10code) >= 2:
            two_icd10_code_list.append(key_error_list[i])
            continue

        elif len(icd10code) == 1:
            icd10code = icd10code[0].replace('.', '')
            try:
                icd9code = icd10_icd9_dict[icd10code]
                diagnosisstring_code_dict[key_error_list[i]] = level1[icd9code]
            except KeyError:
                icd10_key_error_list.append(key_error_list[i])
    print('Number of more than one icd10 codes : {}'.format(len(two_icd10_code_list)))
    print('Number of icd10key_error_list : {}'.format(len(icd10_key_error_list)))

    # deal with more than one ICD10 code ??? why? more than one ICD10code should have more than one class?
    # class_list = ['6', '7', '6', '7', '2', '6', '6', '7', '6', '6', '6']
    # for i in range(11):
    #     diagnosisstring_code_dict[two_icd10_code_list[i]] = class_list[i]
    # fill in the blank!
    have_to_find = []
    already_in = []
    for i in range(len(eicu_dx_unique)):
        single_dx = eicu_dx_unique[i]
        try:
            oneoneone = diagnosisstring_code_dict[single_dx]
            already_in.append(single_dx)
        except KeyError:
            have_to_find.append(single_dx)
    print('Number of dx we have to find...{}'.format(len(have_to_find)))

    # one hierarchy above
    have_to_find2 = []
    for i in range(len(have_to_find)):
        s = "|".join(have_to_find[i].split('|')[:-1])
        try:
            depth1_code = diagnosisstring_code_dict[s]
            diagnosisstring_code_dict[have_to_find[i]] = depth1_code
        except KeyError:
            have_to_find2.append(have_to_find[i])
    print('Number of dx we have to find...{}'.format(len(have_to_find2)))

    # hierarchy below
    dict_keys = list(diagnosisstring_code_dict.keys())

    have_to_find3 = []
    for i in range(len(have_to_find2)):
        s = have_to_find2[i]
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])

        dx_list = list(set(dx_list))
        if len(dx_list) == 1:
            diagnosisstring_code_dict[s] = dx_list[0]
        else:
            have_to_find3.append(s)

    print('Number of dx we have to find...{}'.format(len(have_to_find3)))

    # hierarchy abovs
    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find4 = []

    for i in range(len(have_to_find3)):
        s = "|".join(have_to_find3[i].split('|')[:-1])
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])

        dx_list = list(set(dx_list))
        if len(dx_list) == 1:
            diagnosisstring_code_dict[have_to_find3[i]] = dx_list[0]
        else:
            have_to_find4.append(have_to_find3[i])

    print('Number of dx we have to find...{}'.format(len(have_to_find4)))

    for t in range(4):
        c = -t - 1
    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find_l = []
    for i in range(len(have_to_find4)):
        s = have_to_find4[i]
        s = "|".join(s.split("|")[:c])
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
        dx_list2 = list(set(dx_list))
        if len(dx_list2) > 1:
            cnt = Counter(dx_list)
            mode = cnt.most_common(1)
            diagnosisstring_code_dict[have_to_find4[i]] = mode[0][0]
        else:
            have_to_find_l.append(have_to_find4[i])
    del (have_to_find4)
    have_to_find4 = have_to_find_l.copy()
    print('Number of dx we have to find...{}'.format(len(have_to_find4)))

    dx_depth1 = []
    dx_depth1_unique = []
    solution = lambda data: [x for x in set(data) if data.count(x) != 1]
    return diagnosisstring_code_dict, level1_name


input_path = '/home/project/GraphCLHealth/data/eICU/'
microlab_file = input_path + '/microlab10.csv'
ccs_icd_file = input_path + 'ccs_multi_dx_tool_2015.csv'
icd_10to9_file = input_path + 'icd10cmtoicd9gem.csv'
diagnosis_file = input_path + '/diagnosis10.csv'

diagnosisstring_code_dict, level1_name = get_diagnosisstring_code_dict_eicu(ccs_icd_file, diagnosis_file, icd_10to9_file)

# test_icd_string = ['cardiovascular|arrhythmias|atrial fibrillation', 'cardiovascular|arrhythmias|atrial fibrillation']
# single_list = list(pd.Series(test_icd_string).map(diagnosisstring_code_dict))
# dx_depth1 = single_list
# dx_depth1_unique = list(set(single_list))

# print(test_icd_string)
# print(single_list, level1_name[single_list[0]])
# print(dx_depth1_unique)

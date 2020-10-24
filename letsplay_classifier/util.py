import os
import shutil
import json


def arg_max_list(list):
    """
    argmax as in numpy but on a regular list
    :param list: the list, where to find the max
    :return: the index where the max is found
    """
    pred_index, maxx = 0, -100
    for i, ol in enumerate(list):
        if ol > maxx:
            maxx = ol
            pred_index = i
    return pred_index

def retrieve_or_create_dict(json_file):
    """
    retrieve a dictionary from a json file or create a new empty one
    :param json_file: the json file containing the dictionrary
    :return:
    """
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            json_dict = json.load(f)
    else:
        json_dict = {}
    return json_dict

def move_files_to_right_place(data_dir, class_names):
    """
    move the files that we know are miscategorized in the correct directory
    :param data_dir: directory containing files
    :param class_names: the category names
    :return:
    """

    # list of dict, one for each category, each in form file --> true class
    tv_dict_list = [{} for c in range(len(class_names))]

    misclassified = retrieve_or_create_dict('misclassified.json')
    rejected = retrieve_or_create_dict('rejected.json')

    # build the true_values dictionary list, using misclassified
    for miskey in misclassified:
        label, predicted = map(int, miskey.split(':'))
        files_to_check = [x[0] for x in misclassified[miskey]]
        for file_to_check in files_to_check:
            tv_dict_list[label][file_to_check] = predicted


    # for each possible class (label), retrieved its dictionary { file_name --> true value }
    for label_idx, cur_dict in enumerate(tv_dict_list):
        for key_help in cur_dict:

            # the rejected map {file_name --> targeted value} contains entries that have been confirmed by users
            if key_help in rejected:
                target_idx = rejected[key_help]
                source_file = os.path.join(data_dir, class_names[label_idx], key_help)
                target_file = os.path.join(data_dir, class_names[target_idx], key_help)
                if not os.path.isfile(target_file) and os.path.isfile(source_file):
                    print(f'Moving {source_file} to  {target_file}')
                    shutil.move(source_file, target_file)

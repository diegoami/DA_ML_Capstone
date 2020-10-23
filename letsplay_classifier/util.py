import os
import shutil
import json

def move_files_to_right_place(data_dir, class_names):
    """
    move the files that we know are miscategorized in the correct directory
    :param data_dir: directory containing files
    :param class_names: the category names
    :return:
    """
    help_dict = [{} for c in range(len(class_names))]

    if os.path.isfile('misclassified.json'):
        with open('misclassified.json', 'r') as f:
            misclassified = json.load(f)
    else:
        misclassified = {}

    for miskey in misclassified:
        label, predicted = map(int,miskey.split(':'))
        files_to_check = [x[0] for x in misclassified[miskey]]
        for file_to_check in files_to_check:
            help_dict[label][file_to_check] = predicted



    if os.path.isfile('rejected.json'):
        with open('rejected.json', 'r') as f:
            rejected = json.load(f)
    else:
        rejected = {}

    for label_idx, cur_dict in enumerate(help_dict):
        for key_help in cur_dict:
            if key_help in rejected:
                target_idx = rejected[key_help]
                source_file = os.path.join(data_dir, class_names[label_idx], key_help)
                target_file = os.path.join(data_dir, class_names[target_idx], key_help)
                if not os.path.isfile(target_file) and os.path.isfile(source_file):
                    print(f'Moving {source_file} to  {target_file}')
                    shutil.move(source_file, target_file)

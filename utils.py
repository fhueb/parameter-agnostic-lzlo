import numpy as np
import torch
import os
import shutil
import csv
import re


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_result_dir(args):
    result_dir = args.result_dir
    result_dir = result_dir + args.algo + '/'
    mp = {
        'dataset': None,
        'model': None,
        'algo': '',
        'loss': None,
        'epochs': 'epoch',
        'batch_size': None,
        'momentum': 'mom',
        'gpu': None,
        'print_freq': None,
        'seed': '',
        'result_dir': None,
        'resume': None,
        'data': None,
        'dist_url': None,
        'dist_backend': None,
        'emsize': None,
        'nhid': None,
        'nlayers': None,
        'bptt': None,
        'dropout': None,
        'dropouth': None,
        'dropouti': None,
        'dropoute': None,
        'wdrop': None,
        'nonmono': None,
        'log_interval': None,
        'alpha': None,
        'beta': None,
        'wd': None,
        'tied': None,
        'cuda': None
    }
    for arg in vars(args):
        if arg in mp and mp[arg] is None:
            continue
        value = getattr(args, arg)
        if type(value) == bool:
            value = 'T' if value else 'F'
        if type(value) == list:
            value = str(value).replace(' ', '')
        name = mp.get(arg, arg)
        result_dir += name + str(value) + '_'
    return result_dir


def create_result_dir(args):
    result_dir = get_result_dir(args)
    id = 0
    while True:
        result_dir_id = result_dir + '_%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    return result_dir_id


class Logger(object):
    def __init__(self, dir):
        self.fp = open(dir, 'w')

    def __del__(self):
        self.fp.close()

    def print(self, *args, **kwargs):
        print(*args, file=self.fp, **kwargs)
        print(*args, **kwargs)


class TableLogger(object):
    def __init__(self, path, header, keep_training):
        import csv
        mode = 'a' if keep_training else 'w'
        self.fp = open(path, mode)
        self.logger = csv.writer(self.fp, delimiter='\t')
        if not keep_training:
            self.logger.writerow(header)
        self.header = header

    def __del__(self):
        self.fp.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.fp.flush()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def infer_type(value, col_name):
    if col_name == 'epoch':
        return int(value)
    else:
        return float(value)


def read_tsv_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
        header = next(reader)

        # Initialize empty lists for each header
        for col in header:
            result_dict[col] = []

        # Read rows and append values to corresponding lists
        for row in reader:
            for i, value in enumerate(row):
                col_name = header[i]
                value = infer_type(value, col_name)
                result_dict[col_name].append(value)

    return result_dict


def extract_params_from_folder(folder_name):
    params = {}
    for param in ['lr', 'gamma', 'mom', 'nu', 'save', 'lr_decay', 'mom_decay', 'epoch']:
        if param in ['save', 'epoch']:
            match = re.search(f"{param}([0-9]+)", folder_name)
        else:
            match = re.search(f"{param}([0-9.]+)", folder_name)
        if match:
            if param == 'save':
                params[param] = match.group(1)
            elif param == 'epoch':
                params[param] = int(match.group(1))
            else:
                params[param] = float(match.group(1))

    # Extract the seed value
    seed_match = re.search(r"epoch.*_([0-9]+)_save", folder_name)
    if seed_match:
        params['seed'] = int(seed_match.group(1))

    return params


# Reads the results from the logged data into python.
def get_results(algs=None, root_path='./result_50', included_params=None, filter_params=None, filter_values=None):

    algs = ['adagrad', 'nsgdm', 'sgd_clip', 'mix_clip'] if algs is None else algs
    included_params = ['lr', 'gamma', 'mom', 'nu'] if included_params is None else included_params
    filter_params = ['seed'] if filter_params is None else filter_params
    filter_values = [2020] if filter_values is None else filter_values
    final_dict = {}

    for alg_name in algs:
        alg_path = os.path.join(root_path, alg_name)
        if os.path.exists(alg_path):
            for folder_name in os.listdir(alg_path):
                folder_path = os.path.join(alg_path, folder_name)
                if os.path.isdir(folder_path):
                    params = extract_params_from_folder(folder_name)

                    # Rename sgd_clip with learning rate decay.
                    new_name = alg_name
                    if alg_name == 'sgd_clip':
                        if params['lr_decay'] > 0.1:
                            new_name = 'decaying_sgd_clip'

                    actual_values = [params[filter_param] for filter_param in filter_params]
                    if actual_values == filter_values:
                        train_log = read_tsv_to_dict(os.path.join(folder_path, 'train.log'))
                        test_log = read_tsv_to_dict(os.path.join(folder_path, 'test.log'))
                        key_list = [new_name]
                        key_list.extend([params.get(param, None) for param in included_params])
                        key_tuple = tuple(key_list)
                        final_dict[key_tuple] = {'train': train_log, 'test': test_log}
                else:
                    print(f"Path {folder_name} is not a directory.")
        else:
            print(f"Path {alg_path} does not exist.")

    return final_dict


# Finds the best performing model over all epochs.
def find_min_values(log_dict, keys=None):
    keys = ['loss', 'ppl'] if keys is None else keys
    min_values = {}
    for key in keys:
        min_values[key] = min(log_dict.get(key, [float('inf')]))
    return min_values


# Maps each combination of hyperparameters to the 4 min values
def create_best_model_dict(results_dict):
    best_model_dict = {}
    for key_tuple, data in results_dict.items():
        train_min_values = find_min_values(data['train'])
        test_min_values = find_min_values(data['test'])
        best_model_dict[key_tuple] = {'train': train_min_values['loss'],
                                      'test': test_min_values['loss']}
    return best_model_dict


# Copies the whole content of from_dir to to_dir
def copy_contents(src_dir, dest_dir):
    for item in os.listdir(src_dir):
        src_item_path = os.path.join(src_dir, item)
        dest_item_path = os.path.join(dest_dir, item)

        if os.path.isdir(src_item_path):
            shutil.copytree(src_item_path, dest_item_path)
        else:
            shutil.copy2(src_item_path, dest_item_path)

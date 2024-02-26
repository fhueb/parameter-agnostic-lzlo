import os
import math
from utils import extract_params_from_folder
from utils_lstm import model_load
from algorithm import SGDClip, MomClip, MixClip, SGD, NormalizedSGD, Adagrad
import torch


def find_matching_folder(results_path, alg_name, target_params):
    # Iterate through alg directories
    for candidate_alg in os.listdir(results_path):
        if candidate_alg == alg_name:
            alg_path = os.path.join(results_path, candidate_alg)
            # Iterate through model directories
            for candidate_folder in os.listdir(alg_path):
                candidate_params = extract_params_from_folder(candidate_folder)

                # Check if all target_params match the candidate_params
                if all(candidate_params.get(k) == v for k, v in target_params.items()):
                    return os.path.join(alg_path, candidate_folder)

    return None  # Return None if no matching folder is found


def calc_iter(eta, etat, lr_decay):
    # etat = eta * iter^(-lr_decay)
    # => iter = (etat / eta)^(-1/lr_decay)
    factor = etat/eta
    t = round(math.pow(factor, -1/lr_decay))
    return t + 1


# Since the AlgorithmBase are custom, they are not saved correctly and need to be set again.
# All the relevant states are set correctly however.
def load_prev_optimizer(args, optimizer):
    # Builds the corresponding optimizer
    alg = args.algo
    if alg == 'sgd':
        optimizer.algo = SGD
    if alg == 'sgd_clip':
        optimizer.algo = SGDClip
    if alg == 'mom_clip':
        optimizer.algo = MomClip
    if alg == 'mix_clip':
        optimizer.algo = MixClip
    if alg == 'nsgdm':
        optimizer.algo = NormalizedSGD
    if alg == 'adagrad':
        optimizer.algo = Adagrad

    # Calculates iterations
    if args.lr_decay > 0:
        loaded_lr = optimizer.param_groups[0]['lr']
        iter = calc_iter(args.lr, loaded_lr, args.lr_decay)
        print(f'args.lr: {args.lr}, optimizer_lr: {loaded_lr}, calced iter: {iter}.')
    else:
        iter = 1

    # iter is considered seperately due to my hacky solution. FeelsBadMan
    return iter


# Moves all the tensors of the loaded model to the correct gpu
def move_to_gpu(args, model, criterion, optimizer):
    model = model.cuda(args.gpu)
    criterion = criterion.cuda(args.cuda)
    # Move optimizer.state
    for state in optimizer.state.values():
        for k, v in state.items():
            # print(f'{k}: {v}')
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(args.gpu)

    # Move any tensors in optimizer.param_groups if needed
    for param_group in optimizer.param_groups:
        group_state = param_group.get('group_state', {})
        for k, v in group_state.items():
            # print(f'{k}: {v}')
            if isinstance(v, torch.Tensor):
                group_state[k] = v.cuda(args.gpu)


def load_prev_model(args, model_folder):
    # Loading the previous model. Assumes that args.save is already updated to the model that should be loaded.
    model, criterion, optimizer = model_load(args.save)

    # Moving it to our current gpu
    if args.cuda:
        move_to_gpu(args, model, criterion, optimizer)

    # Fixes custom AlgorithmBase and calculates the correct iter.
    iter = load_prev_optimizer(args, optimizer)

    return model, criterion, optimizer, iter
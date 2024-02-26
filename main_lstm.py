from utils_lstm import *
from rerun import load_prev_model, find_matching_folder
from utils import TableLogger, create_result_dir, extract_params_from_folder, copy_contents


if __name__ == '__main__':
    # Run commands
    # python main_lstm.py --data [data_folder] --result_dir result/ --epochs 300 --algo nsgdm --lr 25.0 --lr_decay 0.75
    # --mom_decay 0.5 --seed 1970
    args = parse_arguments()

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Keep Training Logic
    keep_training = args.start_epoch > 0

    # Creates the folder for saving if we train a new model, and loads the previous folder if we keep training
    if keep_training:
        # Checks whether there exists a saved model that matches the specified params and if, returns its folder
        target_params = {'lr': args.lr,
                         'gamma': args.gamma,
                         'lr_decay': args.lr_decay,
                         'mom_decay': args.mom_decay,
                         'epoch': args.start_epoch,
                         'seed': args.seed}
        model_dir = find_matching_folder(args.result_dir, args.algo, target_params=target_params)
        if model_dir is None:
            raise ImportError("No model with fitting parameters was found.")
        # Also loads the specific save number of this model and overwrites the current one.
        params = extract_params_from_folder(model_dir)
        args.save = f"{params['save']}.pt"

    # Creates the result dir for the new params, independently whether we extend training or not.
    # Also changes the args.save from model_name.pt to result_dir/model_name.pt.
    result_dir = create_result_dir(args)
    args.save = os.path.join(result_dir, args.save)

    # Copies the old results to the new folder
    if keep_training:
        copy_contents(model_dir, result_dir)

    train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'ppl'], keep_training)
    test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'ppl'], keep_training)

    # Hacky-ish Solution to introduce decaying stepsizes.
    # Schedulers sadly only work for the lr, not for momentum.
    lr_lambda = lambda t: pow(t, -args.lr_decay) * args.lr
    mom_lambda = lambda t: 1 - pow(t, -args.mom_decay)

    train_data, val_data, test_data, eval_batch_size, test_batch_size, corpus = load_data(args)
    if keep_training:
        model, criterion, optimizer, iter = load_prev_model(args, result_dir)
    else:
        model, criterion, optimizer = build_model(args, corpus)
        iter = 1

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(args.start_epoch, args.epochs):
            # For timing purposes
            epoch_start_time = time.time()

            # Training
            iter = train(train_logger, iter, args, train_data, lr_lambda, mom_lambda, epoch, model, criterion, optimizer)
            print(iter)
            # Validation
            val_loss = evaluate(val_data, test_logger, eval_batch_size, args, epoch, model, criterion)
            # Nice plotting
            end_of_epoch_print(val_loss, epoch_start_time, epoch)

            # Always save the last model to keep training if we want.
            model_save(args.save, model, criterion, optimizer)
            print('Saving latest model.')

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model, criterion, optimizer = model_load(args.save)

    # Run on test data.
    test_loss = evaluate(test_data, None, test_batch_size, args, epoch, model, criterion)
    print('=' * 89)
    print('| End of training | test loss {:7.4f} | test ppl {:9.3f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)


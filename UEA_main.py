import os
import argparse
import logging
import warnings

import pandas as pd

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, dataset_class, Data_Verifier, train, show_losses, show_accuracies, \
    train_val, evaluate
from Models_.model import model_factory, count_parameters
from Models_.optimizers import get_optimizer
from Models_.loss import get_loss_module
from Models_.utils import load_model

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/Multivariate_ts', choices={'Dataset/UEA/Multivariate_ts'},
                    help='Data path')

parser.add_argument('--problem', default='BasicMotions', choices={'UEA'},
                    help='Data path')

parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['C-T'], choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)")
# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')

parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--lr_min', type=float, default=4e-4, help='learning rate min')

parser.add_argument('--l2_reg', type=float, default=1e-3, help='l2_reg')

parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--init', type=bool, default=True, help='model init')

parser.add_argument('--dropout', type=float, default=0.3, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')

parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()

if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    device = Initialization(config)

    columns = ["dataset", "acc", "f1", "recall", "precision", "test_data_shape", "train_data_shape", "val_data_shape"]

    results_df = pd.DataFrame(columns=columns)

    for problem in os.listdir(config.data_path):

        config.data_dir = os.path.join(config.data_path, problem)
        config.problem = problem

        logger.info(f'------------------------------------------------------------------------------------')
        logger.info(f'dataset: {problem}')
        logger.info("Loading Data ...")

        train_dataset = Data_Loader(config, 'train')
        val_dataset = Data_Loader(config, 'test')
        test_dataset = Data_Loader(config, 'test')
        logger.info('train_dataset: ' + str(train_dataset.feature.shape))
        #
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        # --------------------------------------------------------------------------------------------------------------
        # # -------------------------------------------- Build Model -----------------------------------------------------
        dic_position_results = [config.data_dir.split('/')[-1]]

        logger.info("Creating model ...")
        config.Data_shape = train_dataset.feature.shape
        config.num_labels = int(max(train_dataset.labels.shape)) + 1
        model = model_factory(config)

        # logger.info("Model:\n{}".format(model))
        # logger.info("Total number of parameters: {}".format(count_parameters(model)))

        # -------------------------------------------- Model Initialization ------------------------------------
        optim_class = get_optimizer("RAdam")

        config.optimizer = optim_class(model.parameters(), lr=config.lr, weight_decay=0)
        config.loss_module = get_loss_module()

        save_model_path = os.path.join(config.save_dir, problem)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        last_save_path = os.path.join(save_model_path, problem + '_model_{}.pth'.format('last'))
        best_save_path = os.path.join(save_model_path, problem + '_model_{}.pth'.format('best'))

        config.last_save_path = last_save_path
        config.best_save_path = best_save_path

        tensorboard_writer = SummaryWriter(os.path.join(config.tensorboard_dir, config.problem))

        tensorboard_writer.add_graph(model, input_to_model=torch.randn(*config.Data_shape))

        config.tensorboard_writer = tensorboard_writer

        model.to(device)

        # ---------------------------------------------- Training The Model ------------------------------------
        logger.info('Starting training...')
        torch.cuda.empty_cache()

        train_losses, train_acces, eval_losses, eval_acces = train(model, train_loader, val_loader,
                                                                   device, config)


        best_model, optimizer, start_epoch = load_model(model, best_save_path, config.optimizer)
        best_model.to(device)

        test_metric = evaluate(config, test_loader, model, device)

        f1_score = test_metric['f1_score']
        precision = test_metric['precision']
        recall = test_metric['recall']
        acc = test_metric['acc']

        result = {
            "dataset": problem,
            "acc": acc,
            "f1": f1_score,
            "recall": recall,
            "precision": precision,
            "test_data_shape": test_dataset.feature.shape,
            "train_data_shape": train_dataset.feature.shape,
            "val_data_shape": val_dataset.feature.shape,
        }

        problem_df = pd.DataFrame([result])

        problem_df.to_csv(os.path.join(config.pred_dir + '/' + problem + '.csv'))

        results_df = pd.concat([results_df, problem_df], ignore_index=True)

    results_df.to_csv(os.path.join(config.output_dir, 'Results.csv'))

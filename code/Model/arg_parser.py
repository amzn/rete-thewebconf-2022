import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run DyRec.")

    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for using multi GPUs.')
    parser.add_argument('--device', type=int, default=1,
                        help='device id')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='elec',
                        help='Choose a dataset from {elec, music, book, beauty, book_large}')
    parser.add_argument('--data_dir', nargs='?', default='../PreProcess/',
                        help='Input data path.')

    parser.add_argument('--time_step', type=int, default=11, help='number of time steps')
    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with BERT.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--rec_batch_size', type=int, default=64,
                        help='recommendation batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--entity_dim', type=int, default=128,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=128,
                        help='Relation Embedding size.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--rec_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating recommendation l2 loss.')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='margin value in ranking loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=30,
                        help='Number of epoch for early stopping')

    parser.add_argument('--rec_print_every', type=int, default=1,
                        help='Iter interval of printing recommendation loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--K', type=int, default=20,
                        help='Calculate metric@K when evaluating.')

    parser.add_argument('--sampler_config', type=str, default='config/gat_5_ppr_ogb.yml',
                        help = 'Config files for subgraph sampler.')
    args = parser.parse_args()


    save_dir = '../trained_model/{}/entitydim{}_lr{}_pretrain{}/'.format(
        args.data_name, args.entity_dim, args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args
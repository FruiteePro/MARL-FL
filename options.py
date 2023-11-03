import argparse
import os
from Dataset.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--train_mark', type=str, default='10')
    parser.add_argument('--model', type=str, default='MARL_train', help='model name')
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_mnist', type=str, default=os.path.join(path_dir, 'data/MNIST/'))
    parser.add_argument('--datased_ID', type=str, default='mnist')
    parser.add_argument('--model_ID', type=str, default='102801')
    parser.add_argument('--model_round_ID', type=str, default='86')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_servers', type=int, default=3)
    parser.add_argument('--num_dataset', type=int, default=1)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=400)
    parser.add_argument('--num_marl_train_episodes', type=int, default=200)
    parser.add_argument('--num_marl_episode_length', type=int, default=500)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--minimal_size', type=int, default=1000)
    parser.add_argument('--batch_size_rl', type=int, default=500)
    parser.add_argument('--update_interval', type=int, default=200)
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--xi', type=float, default=64)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.3, help='balance parameter between different models')
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--target_acc', type=float, default=0.5)
    parser.add_argument('--lr_actor', type=float, default=1e-2, help='learning rate for actor network')
    parser.add_argument('--lr_critic', type=float, default=1e-2, help='learning rate for critic network')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_id', type=int, default='1')
    parser.add_argument('--non_iid_alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.97)

    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    args.device = args.device + ':' + str(args.device_id)

    return args

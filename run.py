from server import Server
from client import Client
import utils
from marl import MADDPG, TwoLayerNet
from Dataset.long_tailed_cifar10 import train_long_tail, get_100_samples, get_imb_samples
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
from Dataset.sample_dirichlet import clients_indices, clients_indices_noniid
from Dataset.Gradient_matching_loss import match_loss
from sklearn.decomposition import PCA
import logging
import copy
import random
from tqdm import tqdm
import csv
import os
from threading import Thread
import torch
import torch.nn as nn
import torch.optim as optim
from Model.Resnet8 import ResNet_cifar
from options import args_parser
import numpy as np
from torchvision.transforms import transforms
from torchvision import datasets
import wandb



# 处理 cifar10 数据集
def get_cifar10_data(args, seed_num):
    logging.info('Loading cifar10 data...')
    random_state = np.random.RandomState(seed_num)
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)

    list_label2indices = classify_label(data_local_training, args.num_classes)
    
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                       args.imb_factor, args.imb_type)

    list_client2indices = clients_indices_noniid(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, seed_num)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                                args.num_classes)
    indices2data = Indices2Dataset(data_local_training)

    return list_client2indices, indices2data, data_global_test

# 处理 cifar100 数据集
def get_cifar100_data(args, seed_num):
    # logging.info('Loading cifar100 data...')
    random_state = np.random.RandomState(seed_num)
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])

    data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_all)

    list_label2indices = classify_label(data_local_training, args.num_classes)
    
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                       args.imb_factor, args.imb_type)

    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, seed_num)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                                args.num_classes)
    indices2data = Indices2Dataset(data_local_training)

    return list_client2indices, indices2data, data_global_test

# 处理 mnist 数据集
def get_mnist_data(args, seed_num):
    # logging.info('Loading mnist data...')
    random_state = np.random.RandomState(seed_num)
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_local_training = datasets.MNIST(args.path_mnist, train=True, download=True, transform=transform_all)
    data_global_test = datasets.MNIST(args.path_mnist, train=False, transform=transform_all)

    list_label2indices = classify_label(data_local_training, args.num_classes)
    
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                       args.imb_factor, args.imb_type)

    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, seed_num)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                                args.num_classes)
    indices2data = Indices2Dataset(data_local_training)

    return list_client2indices, indices2data, data_global_test

# 获得设备硬件信息
def get_clients_info(num_clients):
    # logging.info('Loading client info...')
    info = []
    B = []
    for i in np.arange(1, 5, 0.02):
        if (i / 0.02) % 10 == 3:
            continue
        B.append(i);
    with open('data.csv', 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for i in range(num_clients):
            for key in rows[i]:
                if key == "fList":
                    rows[i][key] = eval(rows[i][key])
                else:
                    rows[i][key] = float(rows[i][key])
            rows[i]["BList"] = B
            
            info.append(rows[i])
    return info


def action_to_deviceSelection(num_clients, action:list):
    deviceSelection = []
    train_client_list = []
    for client_id in range(num_clients):
        train_model_id = [] 
        for i in range(len(action)):
            if action[i][client_id] == 1:
                train_model_id.append(i)
        deviceSelection.append(train_model_id)
        if len(train_model_id) > 0:
            train_client_list.append(client_id)
    return train_client_list, deviceSelection

def client_to_states(num_servers, client_list):
    states = []
    for server_id in range(num_servers):
        state = []
        for client in client_list:
            state_i = client.get_state(server_id).copy()
            state.extend(state_i)
        states.append(state)
    return states

def client_to_states_param(num_servers, client_list:list, server_list:list):
    states = []
    for server_id in range(num_servers):
        state = []
        server_state = server_list[server_id].get_state_params()
        state.append(server_state)
        for client in client_list:
            state_i = client.get_state_params(server_id)
            state.append(state_i)
        matrix = torch.stack(state)
        states.append(matrix)
    return states

def pca_first(states, num_client):
    new_states = []
    trans_matrix = []
    for state in states:
        pca = PCA(n_components=num_client)
        state = state.cpu().numpy()
        state = pca.fit_transform(state)
        new_states.append(state.flatten().tolist())
        trans_matrix.append(torch.from_numpy(pca.components_).t())
    return new_states, trans_matrix

def pca_compute(states, trans_matrix, device):
    new_states = []
    for state, matrix in zip(states, trans_matrix):
        matrix = matrix.to(device)
        state = torch.matmul(state, matrix)
        new_states.append(state.view(-1).tolist())
    return new_states


def get_global_model_params(server_list):
    global_model_params = []
    for server in server_list:
        global_model_params.append(server.download_params().copy())
    return global_model_params

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getE_reward(sumE):
    return (sigmoid(1 / (sumE + 0.1)) * 0.5) - 0.5

# run
def run():
    # 输入参数
    args = args_parser()
    # 设置 np.random 随机
    random_state = np.random.RandomState(args.seed)
    # 设置 random 随机
    random.seed(args.seed)
    log_pth = './log/'
    model_pth = './model_saved/'
    os.makedirs(log_pth, exist_ok=True)
    os.makedirs(model_pth, exist_ok=True)
    log_name = log_pth + args.train_mark + '.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_name, level=logging.DEBUG, format=log_format)
    # 获得数据集 分别存储每个模型的数据
    # 标签到设备的索引列表
    list_list_client2indices = []
    # 数据包列表
    list_indices2data = []
    # 测试数据列表
    list_data_global_test = []

    wandb.init(
        project="New_Lib" + args.train_mark,
        config={
            "dataset": args.dataset_ID,
            "epochs": args.num_rounds,
        }
    )

    # 创建数据
    for i in range(args.num_servers):
        # 获取数据
        if args.dataset_ID == 'cifar10':
            list_client2indices, indices2data, data_global_test = get_cifar10_data(args, args.seed + i)
        elif args.dataset_ID == 'cifar100':
            list_client2indices, indices2data, data_global_test = get_cifar100_data(args, args.seed + i)
        elif args.dataset_ID == 'mnist':
            list_client2indices, indices2data, data_global_test = get_mnist_data(args, args.seed + i)
        # 添加到列表
        list_list_client2indices.append(list_client2indices)
        list_indices2data.append(indices2data)
        list_data_global_test.append(data_global_test)

    # 获取客户端信息
    client_info = get_clients_info(args.num_clients)

    # 存放客户和服务器对象
    client_list = []
    server_list = []

    # 联邦学习部分初始化

    # 创建 server 对象
    # logging.info('Creating server...')
    for i in range(args.num_servers):
        server_temp = Server(i)
        server_list.append(server_temp)
        server_list[i].config(args)
        server_list[i].set_data(list_data_global_test[i])

    # 创建 client 对象
    # logging.info('Creating client...')
    for client_id in range(args.num_clients):
        client_list.append(Client(client_id))
        client_list[client_id].config(args)
        # 初始化 client 数据
        for data_id in range(args.num_servers):
            # list_indices2data[data_id].load(list_list_client2indices[data_id][client_id])
            # data_client = list_indices2data[data_id]
            client_list[client_id].set_data_index(list_list_client2indices[data_id][client_id])
            client_list[client_id].set_dataloader(list_indices2data[data_id])
            # client_list[client_id].set_data(list_indices2data[data_id])
        # 初始化 client 模型
        client_list[client_id].set_models(args.num_servers)
        # 初始化 clinet 硬件信息
        client_list[client_id].setup(client_info[client_id])

    # 多智能体强化学习部分初始化
    
    state_dims = []
    action_dims = []

    total_clients = [i for i in range(args.num_clients)]

    # logging.info('Creating multi-agent reinforcement learning...')
    for i in range(args.num_servers):
        state_dims.append((args.num_clients * (args.num_clients + 1)))
        action_dims.append(args.num_clients)
    critic_input_dim = sum(state_dims) + sum(action_dims)

    # 创建多智能体强化学习对象
    maddpg = MADDPG(args.num_servers, args.device, args.lr_actor, args.lr_critic, args.hidden_dim, state_dims,
                action_dims, critic_input_dim, args.gamma, args.tau, args.epsilon)
    replay_buffer = utils.ReplayBuffer(args.buffer_size)
    total_step = 0

    if args.model_ID != '0' :
        marl_model_id_idx = model_pth + args.model_ID + '_' + args.model_round_ID + '_round_model_ddpg_'
        maddpg.load_model(marl_model_id_idx)
        args.minimal_size = 0

    # states = client_to_states(args.num_servers, client_list)

    reward_list = [[] for i in range(args.num_servers)]

    epoch_reward_list = [[] for i in range(args.num_servers)]

    def reset():
        # logging.info("reset clients and servers...")
        for client in client_list:
            client.reset_state()
        for server in server_list:
            server.reset_state()


    # 强化学习训练过程
    # logging.info('Start training...')
    for r in tqdm(range(1, args.num_marl_train_episodes+1), desc='marl-training'):
        if_unfinish = True
        reset()
        # 全部设备先训练一次
        train_client_list = [i for i in range(args.num_clients)]
        deviceSelection = [[i for i in range(args.num_servers)] for j in range(args.num_clients)]
        global_model_params = get_global_model_params(server_list)
        for client_id in train_client_list:
                client_list[client_id].update_models(global_model_params, deviceSelection)
                client_list[client_id].set_time_limit(5)

        # for client_id in train_client_list:
        #     client_list[client_id].train_models()
        group_size = 50  # 每组的大小
        for i in range(0, len(train_client_list), group_size):
            group = train_client_list[i:i+group_size]  # 获取当前组的client列表
            threads = [Thread(target=client_list[client_id].train_models) for client_id in group]
            [t.start() for t in threads]
            [t.join() for t in threads]

        reports = [client_list[client_id].get_result() for client_id in train_client_list]
        for server in server_list:
            server.load_reports(reports)
            server.aggregation()
            server.global_eval()
        states = client_to_states_param(args.num_servers, client_list, server_list)
        states, trans_matrix = pca_first(states, args.num_clients)

        for k in tqdm(range(1, args.num_marl_episode_length+1), desc='episode-training'):
            if not if_unfinish:
                break
            if (total_step > args.minimal_size):
                actions = maddpg.take_action2(states, explore=True)
                actions = utils.onetop_logits(actions)
            else:
                actions = []
                for i in range(args.num_servers):
                    selected = []
                    online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
                    for i in range(args.num_clients):
                        if i in online_clients:
                            selected.append(1)
                        else:
                            selected.append(0)
                    actions.append(selected)
            
            train_client_list, deviceSelection = action_to_deviceSelection(args.num_clients, actions)

            global_model_params = get_global_model_params(server_list)

            for client_id in train_client_list:
                client_list[client_id].update_models(global_model_params, deviceSelection)
                client_list[client_id].set_time_limit(5)
            
            threads = [Thread(target=client_list[client_id].train_models) for client_id in train_client_list]
            [t.start() for t in threads]
            [t.join() for t in threads]
            # for client_id in train_client_list:
            #     client_list[client_id].train_models()

            reports = [client_list[client_id].get_result() for client_id in train_client_list]
            reward = []
            done = []
            acc_last = []
            sum_List = []
            for server in server_list:
                server.load_reports(reports)
                server.aggregation()
                server.global_eval()
                sumE = sum([client_list[client_id].get_last_E() for client_id in train_client_list]) / len(train_client_list)
                sum_List.append(sumE)
                # reward.append(pow(args.xi, server.fedavg_acc[-1] - args.target_acc) - 1)
                acc_last.append(server.fedavg_acc[-1])
                done.append(server.fedavg_acc[-1] > args.target_acc)
            
            sumE = sum(sum_List) / len(sum_List)
            for i, acc in enumerate(acc_last):
                reward_val = pow(args.xi, acc - args.target_acc) - 1 + getE_reward(sumE)
                # reward_val = pow(args.xi, acc - args.target_acc) - 1
                reward.append(reward_val)
                wandb.log({"acc_" + (str)(r) + "_" + (str)(i): acc})

            # logging.info("episode: {}, acc_last: {}".format(k, acc_last))
            next_states = client_to_states_param(args.num_servers, client_list, server_list)
            # 处理一下
            next_states = pca_compute(next_states, trans_matrix, args.device)
            replay_buffer.add(states, actions, reward, next_states, done)
            states = next_states
            total_step += 1

            # if all item in done is True, if_unfinish = False
            if_unfinish = not all(done)
            # print(replay_buffer.size())

            for i in range(args.num_servers):
                reward_list[i].append(reward[i])

            #  强化学习训练 
            if replay_buffer.size() >= args.minimal_size and total_step % args.update_interval == 0:
                sample = replay_buffer.sample(args.batch_size_rl)
                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(args.device)
                        for aa in rearranged
                    ]
                sample = [stack_array(x) for x in sample]
                for a_i in range(len(server_list)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
                # logging.info("step: {}, sum_reward: {}".format(total_step, sum(reward)))
                # logging.info("reward_list: {}".format(reward_list))
                # print("reward_list: {}".format(reward_list))
            
        for i in range(args.num_servers):
            epoch_reward_list[i].append(sum(reward_list[i]))
            wandb.log({"epoch_reward_" + (str)(i): epoch_reward_list[i][-1]})
            reward_list[i] = []
        
        # logging.info("epoch_reward_list: {}".format(epoch_reward_list))
        model_save = model_pth + args.train_mark + '_' + str(r) +'_round_model'
        maddpg.save_model(model_save)

    # 保存 强化学习 模型
    model_save = model_pth + args.train_mark
    maddpg.save_model(model_save)
    # name = args.train_mark + 'rewark.csv'
    # with open(name, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(epoch_reward_list)
    wandb.finish()

def fedavg():
    # 输入参数
    args = args_parser()
    # 设置 np.random 随机
    random_state = np.random.RandomState(args.seed)
    # 设置 random 随机
    random.seed(args.seed)
    log_pth = './log/'
    model_pth = './model_saved/'
    os.makedirs(log_pth, exist_ok=True)
    os.makedirs(model_pth, exist_ok=True)
    log_name = log_pth + args.train_mark + '.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_name, level=logging.DEBUG, format=log_format)
    # logging.info('Running FedAvg...')
    # 获得数据集 分别存储每个模型的数据
    # 标签到设备的索引列表
    list_list_client2indices = []
    # 数据包列表
    list_indices2data = []
    # 测试数据列表
    list_data_global_test = []

    # 创建数据
    for i in range(args.num_servers):
        # 获取数据
        if args.dataset_ID == 'cifar10':
            list_client2indices, indices2data, data_global_test = get_cifar10_data(args, args.seed + i)
        elif args.dataset_ID == 'cifar100':
            list_client2indices, indices2data, data_global_test = get_cifar100_data(args, args.seed + i)
        elif args.dataset_ID == 'mnist':
            list_client2indices, indices2data, data_global_test = get_mnist_data(args, args.seed + i)
        # 添加到列表
        list_list_client2indices.append(list_client2indices)
        list_indices2data.append(indices2data)
        list_data_global_test.append(data_global_test)

    # 获取客户端信息
    client_info = get_clients_info(args.num_clients)

    # 存放客户和服务器对象
    client_list = []
    server_list = []

    total_clients = [i for i in range(args.num_clients)]

    # 联邦学习部分初始化

    # 创建 server 对象
    for i in range(args.num_servers):
        server_temp = Server(i)
        server_list.append(server_temp)
        server_list[i].config(args)
        server_list[i].set_data(list_data_global_test[i])

    # 创建 client 对象
    for client_id in range(args.num_clients):
        client_list.append(Client(client_id))
        client_list[client_id].config(args)
        # 初始化 client 数据
        for data_id in range(args.num_servers):
            # list_indices2data[data_id].load(list_list_client2indices[data_id][client_id])
            # data_client = list_indices2data[data_id]
            # client_list[client_id].set_data(data_client)
            client_list[client_id].set_data_index(list_list_client2indices[data_id][client_id])
            client_list[client_id].set_dataloader(list_indices2data[data_id])
        # 初始化 client 模型
        client_list[client_id].set_models(args.num_servers)
        # 初始化 clinet 硬件信息
        client_list[client_id].setup(client_info[client_id])

    done = []
    acc_list = [[] for i in range(args.num_servers)]
    Esum_list = []
    for server in server_list:
        done.append(False)

    # 联邦学习
    for r in tqdm(range(1, args.num_rounds+1), desc='fedavg-training'):
        # 设备选择
        actions = []
        for i in range(args.num_servers):
            selected = []
            if not done[i]:
                online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
                for i in range(args.num_clients):
                    if i in online_clients:
                        selected.append(1)
                    else:
                        selected.append(0)
            else:
                selected = [0 for i in range(args.num_clients)]
            actions.append(selected)

        train_client_list, deviceSelection = action_to_deviceSelection(args.num_clients, actions)

        global_model_params = get_global_model_params(server_list)

        for client_id in train_client_list:
            client_list[client_id].update_models(global_model_params, deviceSelection)
            client_list[client_id].set_time_limit(5)
        
        threads = [Thread(target=client_list[client_id].train_models) for client_id in train_client_list]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # for client_id in train_client_list:
        #     client_list[client_id].train_models()

        reports = [client_list[client_id].get_result() for client_id in train_client_list]
        acc_last = []
        for i, server in enumerate(server_list):
            if not done[i]:
                server.load_reports(reports)
                server.aggregation()
                server.global_eval()
            # reward.append(pow(args.xi, server.fedavg_acc[-1] - args.target_acc) - 1)
                acc_last.append(server.fedavg_acc[-1])
                done[i] = server.fedavg_acc[-1] > args.target_acc

        sumE = sum([client_list[client_id].get_last_E() for client_id in train_client_list])
        Esum_list.append(sumE)

        for i, acc in enumerate(acc_last):
            acc_list[i].append(acc)

        logging.info("episode: {}, acc_last: {}".format(r, acc_last))
        logging.info("Esum: {}".format(sumE))

        # if all item in done is True, if_unfinish = False
        is_finish = all(done)

        if is_finish:
            break

    logging.info("acc_list: {}".format(acc_list))
    logging.info("Esum_list: {}".format(Esum_list))

def FedMARL():
    # 输入参数
    args = args_parser()
    # 设置 np.random 随机
    random_state = np.random.RandomState(args.seed)
    # 设置 random 随机
    random.seed(args.seed)
    log_pth = './log/'
    model_pth = './model_saved/'
    os.makedirs(log_pth, exist_ok=True)
    os.makedirs(model_pth, exist_ok=True)
    log_name = log_pth + args.train_mark + '.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_name, level=logging.DEBUG, format=log_format)
    logging.info('Running FedMARL...')
    # 获得数据集 分别存储每个模型的数据
    # 标签到设备的索引列表
    list_list_client2indices = []
    # 数据包列表
    list_indices2data = []
    # 测试数据列表
    list_data_global_test = []

    # 创建数据
    for i in range(args.num_servers):
        # 获取数据
        if args.dataset_ID == 'cifar10':
            list_client2indices, indices2data, data_global_test = get_cifar10_data(args, args.seed + i)
        elif args.dataset_ID == 'cifar100':
            list_client2indices, indices2data, data_global_test = get_cifar100_data(args, args.seed + i)
        elif args.dataset_ID == 'mnist':
            list_client2indices, indices2data, data_global_test = get_mnist_data(args, args.seed + i)
        # 添加到列表
        list_list_client2indices.append(list_client2indices)
        list_indices2data.append(indices2data)
        list_data_global_test.append(data_global_test)

    # 获取客户端信息
    client_info = get_clients_info(args.num_clients)

    # 存放客户和服务器对象
    client_list = []
    server_list = []

    # 联邦学习部分初始化

    # 创建 server 对象
    for i in range(args.num_servers):
        server_temp = Server(i)
        server_list.append(server_temp)
        server_list[i].config(args)
        server_list[i].set_data(list_data_global_test[i])

    # 创建 client 对象
    for client_id in range(args.num_clients):
        client_list.append(Client(client_id))
        client_list[client_id].config(args)
        # 初始化 client 数据
        for data_id in range(args.num_servers):
            # list_indices2data[data_id].load(list_list_client2indices[data_id][client_id])
            # data_client = list_indices2data[data_id]
            # client_list[client_id].set_data(data_client)
            client_list[client_id].set_data_index(list_list_client2indices[data_id][client_id])
            client_list[client_id].set_dataloader(list_indices2data[data_id])
        # 初始化 client 模型
        client_list[client_id].set_models(args.num_servers)
        # 初始化 clinet 硬件信息
        client_list[client_id].setup(client_info[client_id])

    state_dims = []
    action_dims = []

    for i in range(args.num_servers):
        state_dims.append((args.num_clients * (args.num_clients + 1)))
        action_dims.append(args.num_clients)
    critic_input_dim = sum(state_dims) + sum(action_dims)

    # 创建多智能体强化学习对象
    maddpg = MADDPG(args.num_servers, args.device, args.lr_actor, args.lr_critic, args.hidden_dim, state_dims,
                action_dims, critic_input_dim, args.gamma, args.tau, args.epsilon, args.num_online_clients)
    
    marl_model_id_idx = model_pth + args.model_ID + '_' + args.model_round_ID + '_round_model_ddpg_'
    maddpg.load_model(marl_model_id_idx)


    done = []
    acc_list = [[] for i in range(args.num_servers)]
    Esum_list = []
    for server in server_list:
        done.append(False)

    def reset():
        logging.info("reset clients and servers...")
        for client in client_list:
            client.reset_state()
        for server in server_list:
            server.reset_state()

    reset()
    # 全部设备先训练一次
    train_client_list = [i for i in range(args.num_clients)]
    deviceSelection = [[i for i in range(args.num_servers)] for j in range(args.num_clients)]
    global_model_params = get_global_model_params(server_list)
    for client_id in train_client_list:
        client_list[client_id].update_models(global_model_params, deviceSelection)
        client_list[client_id].set_time_limit(5)

    # for client_id in train_client_list:
    #     client_list[client_id].train_models()
    threads = [Thread(target=client_list[client_id].train_models) for client_id in train_client_list]
    [t.start() for t in threads]
    [t.join() for t in threads]

    reports = [client_list[client_id].get_result() for client_id in train_client_list]
    for server in server_list:
        server.load_reports(reports)
        server.aggregation()
        server.global_eval()
    states = client_to_states_param(args.num_servers, client_list, server_list)
    states, trans_matrix = pca_first(states, args.num_clients)

    logging.info("start FedMARL...")
    # 联邦学习
    for r in tqdm(range(1, args.num_rounds+1), desc='fedmarl-running'):
        # 设备选择
        actions = maddpg.take_action(states, done, explore=False)
        actions = utils.ktop_from_logits(actions, args.num_online_clients)

        train_client_list, deviceSelection = action_to_deviceSelection(args.num_clients, actions)

        global_model_params = get_global_model_params(server_list)

        for client_id in train_client_list:
            client_list[client_id].update_models(global_model_params, deviceSelection)
            client_list[client_id].set_time_limit(5)
        
        threads = [Thread(target=client_list[client_id].train_models) for client_id in train_client_list]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # for client_id in train_client_list:
        #     client_list[client_id].train_models()

        reports = [client_list[client_id].get_result() for client_id in train_client_list]
        acc_last = []
        for i, server in enumerate(server_list):
            if not done[i]:
                server.load_reports(reports)
                server.aggregation()
                server.global_eval()
            # reward.append(pow(args.xi, server.fedavg_acc[-1] - args.target_acc) - 1)
                acc_last.append(server.fedavg_acc[-1])
                done[i] = server.fedavg_acc[-1] > args.target_acc

        sumE = sum([client_list[client_id].get_last_E() for client_id in train_client_list])
        Esum_list.append(sumE)

        for i, acc in enumerate(acc_last):
            acc_list[i].append(acc)
        
        next_states = client_to_states_param(args.num_servers, client_list, server_list)
        states = pca_compute(next_states, trans_matrix, args.device)

        logging.info("episode: {}, acc_last: {}".format(r, acc_last))

        # if all item in done is True, if_unfinish = False
        is_finish = all(done)

        if is_finish:
            break

    logging.info("acc_list: {}".format(acc_list))
    logging.info("Esum_list: {}".format(Esum_list))

    

if __name__ == '__main__':
    args = args_parser()
    if (args.model == 'FedAvg'):
        fedavg()
    elif (args.model == 'FedMARL'):
        FedMARL()
    else:
        run()














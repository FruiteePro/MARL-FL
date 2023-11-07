import client
import logging
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch import no_grad, eq
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar
from Model.MnistCNN import MnistCNN


class Server(object):
    def __init__(self, model_id):
        self.model_id = model_id
    
    # 配置服务器
    def config(self, config):
        # logging.info('Config {} server...'.format(self.config.server))
        self.config = config
        
        self.device = self.config.device
        self.fedavg_acc = []
        # global 模型
        self.dataset_ID = self.config.dataset_ID
        if self.dataset_ID == 'mnist':
            self.syn_model = MnistCNN().to(self.device)
        else:
            self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=self.config.num_classes).to(self.device)
        self.clients = []
        # 验证的bathc_size
        self.batch_size_test = self.config.batch_size_test

    # 为服务器设置验证集数据
    def set_data(self, data):
        self.testdata = copy.deepcopy(data)

    # 记录服务器所选客户端
    def client_selection(self, clients, client_ids):
        self.clients.clear()
        # select client from clients if in client_ids
        for client in clients:
            if client.client_id in client_ids:
                self.clients.append(client.client_id)

    # 下载服务器模型参数
    def download_params(self):
        return copy.deepcopy(self.syn_model.state_dict())
    
    # 上传客户端模型参数
    def load_reports(self, reports):
        self.reports = []
        for report in reports: 
            if self.model_id in report.model_list:
                for model in report.model_params:
                    if model["model_id"] == self.model_id:
                        self.reports.append(model)
                        break
                
    # 聚合全局模型参数
    def aggregation(self):
        weights = [model["local_weights"] for model in self.reports]
        samples = [model["num_samples"] for model in self.reports]
        total_samples = sum(samples)
        global_weights = copy.deepcopy(weights[0])
        
        for name_param in weights[0]:
            list_values_param = []
            for weight, num_sample in zip(weights, samples):
                list_values_param.append(weight[name_param] * num_sample)
            value_global_param = sum(list_values_param) / total_samples
            global_weights[name_param] = value_global_param

        self.syn_model.load_state_dict(copy.deepcopy(global_weights))

    # 评估全局模型
    def global_eval(self):
        self.syn_model.eval()
        with no_grad():
            correct = 0
            total = len(self.testdata)
            test_loader = DataLoader(self.testdata, batch_size=self.batch_size_test)
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                _, output = self.syn_model(data)
                # print(output.shape)
                # _, predicted = max(output, -1)
                predicted = torch.argmax(output, dim=-1)
                correct += sum(eq(predicted.cpu(), target.cpu())).item()
            accuracy = correct / total
            self.fedavg_acc.append(accuracy)

    def reset_state(self):
        self.clients.clear()
        self.fedavg_acc.clear()
        if self.dataset_ID == 'mnist':
            neo_model = MnistCNN().to(self.device)
        else:
            neo_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=self.config.num_classes).to(self.device)
        self.syn_model.load_state_dict(neo_model.state_dict())

    def get_state_params(self):
        model = copy.deepcopy(self.syn_model)
        all_params = model.state_dict()
        params = torch.cat([value.view(-1) for value in all_params.values()])
        return params









        


    

        

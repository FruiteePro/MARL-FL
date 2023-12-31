import logging
import copy
import torch
import queue
import numpy as np
from sko.PSO import PSO
from options import args_parser
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor, transforms
from Model.Resnet8 import ResNet_cifar
from Model.MnistCNN import MnistCNN
from utils import shared_lock


def enqueue(queue_obj, item):
    if queue_obj.full():
        _ = queue_obj.get()
    queue_obj.put(item)

def rayleigh_fading(std_dev, path_loss, num_samples=1):
    """
    生成服从Rayleigh分布的随机数样本，并加上路径损失
    
    参数:
        num_samples (int): 生成的样本数量
        mean (float): Rayleigh分布的均值
        std_dev (float): Rayleigh分布的标准差
        path_loss (float): 平均路径损失
        
    返回:
        samples (numpy.ndarray): 生成的随机数样本数组
    """
    scale = std_dev / np.sqrt(2)  # 尺度参数
    fading_samples = np.random.rayleigh(scale, size=num_samples)
    path_loss_samples = fading_samples + path_loss
    return path_loss_samples

class Client(object):
    def __init__(self, client_id):
        self.client_id = client_id

    # 下载数据
    def download(self, argv):
        try:
            return argv.copy()
        except:
            return argv
    
    # 上传数据
    def upload(self, argv):
        try:
            return argv.copy()
        except:
            return argv
        
    # 配置客户端 调度相关参数
    def setup(self, info):
        self.workload = [len(idxs) for idxs in self.data_index]
        # self.transload = []
        self.W = 0
        self.L = 0
        self.C = info["C"]
        self.N = info["N"]
        self.h = rayleigh_fading(1, 0.001, 1)
        self.fList = info["fList"]
        self.BList = info["BList"]
        self.fsize = len(self.fList)
        self.Bsize = len(self.BList)
        self.cycPreBit = 737

    # 设置客户端的工作量
    # def set_workload(self, workload, transload, model_id):
    #     self.workload = workload
    #     self.L = transload

    # 设置客户端的本轮的时间限制
    def set_time_limit(self, time_limit):
        self.time_l = time_limit

    # 本轮客户端的调度
    def schedule(self):
        self.W = float(sum([self.workload[i] for i in self.train_model_ids]))
        self.L = float(sum([self.transload[i] for i in self.train_model_ids]))
        self.W = self.W * 32 * 32 * 3 * self.cycPreBit
        self.h = rayleigh_fading(1, 0.001, 1)
        bf, bB, t1, t2, E = min(self.fList), 0, 0, 0, 0
        if self.W == 0:
            return bf, bB, t1, t2, E
        def mapping(input1, input2):
            x1, x2 = input1, input2
            p1 = int(x1 / 10 * (self.fsize - 1));
            p2 = int(x2 / 10 * (self.Bsize - 1));
            if p1 >= self.fsize:
                p1 = self.fsize - 1
            if p2 >= self.Bsize:
                p2 = self.Bsize - 1
            nf = self.fList[p1]
            nB = self.BList[p2]
            return nf, nB
        
        def e_func(x):
            x1, x2 = x
            nf, nB = mapping(x1, x2)
            nf *= np.power(10, 9)
            nB *= np.power(10, 5)
            t1 = self.W / nf
            d1 = - 2 * self.C * np.power(self.W, 3) / np.power(t1, 3)
            t2 = self.time_l - t1
            LBd = self.L / nB
            if (LBd / t2) > 63:
                temp = 63
            else:
                temp = LBd / t2
            d2 = (self.N / self.h) * (np.power(2, temp) - LBd * np.log(2) * np.power(2, temp) / t2 - 1)
            return np.abs(d1 - d2) 
        
        def getEsum(bf, bB, t1, t2):
            if t2 < 1e-4:
                t2 = 1e-4
            LBd = self.L / bB
            if (LBd / t2) > 60:
                temp = 60
            else:
                temp = LBd / t2
            e = self.C * np.power(self.W, 3) / np.square(t1) + self.N * t2 / self.h * (np.power(2, temp) - 1)
            return e
        
        const_ueq = [
            lambda x: - (self.W / (self.fList[int(x[0] / 10 * (self.fsize - 1))] * np.power(10, 9))),
            lambda x: - (self.time_l - self.W / (self.fList[int(x[0] / 10 * (self.fsize - 1))] * np.power(10, 9))),
        ]

        pso = PSO(func=e_func, dim=2, pop=20, max_iter=50, lb=[0, 0], ub=[10, 10], constraint_ueq=const_ueq)
        pso.run()
        # print('best_x is ', pso.gbest_x)
        x1, x2 = pso.gbest_x
        bf, bB = mapping(x1, x2)
        nbf = bf * np.power(10, 9)
        nbB = bB * np.power(10, 5)
        t1 = self.W / nbf
        t2 = self.time_l - t1
        E = getEsum(nbf, nbB, t1, t2)
        return bf, bB, t1, t2, E
    
    def maxfschedule(self):
        self.W = float(sum([self.workload[i] for i in self.train_model_ids]))
        self.L = float(sum([self.transload[i] for i in self.train_model_ids]))
        self.W = self.W * 32 * 32 * 3 * self.cycPreBit
        self.h = rayleigh_fading(1, 0.001, 1)
        bf = max(self.fList) * np.power(10, 9)
        bB = max(self.BList) * np.power(10, 5)
        t1 = self.W / bf  
        t2 = self.time_l - t1
        if t2 < 1e-4:
            t2 = 1e-4
        LBd = self.L / bB
        if (LBd / t2) > 60:
            temp = 60
        else:
            temp = LBd / t2
        E = self.C * self.W * np.power(bf, 2) + self.N * t2 / self.h * (np.power(2, temp) - 1)
        return bf, bB, t1, t2, E
    
    def search(self):
        self.W = float(sum([self.workload[i] for i in self.train_model_ids]))
        self.L = float(sum([self.transload[i] for i in self.train_model_ids]))
        self.W = self.W * 32 * 32 * 3 * self.cycPreBit
        self.h = rayleigh_fading(1, 0.001, 1)
        def get_E(input):
            f, B = input
            f = f * np.power(10, 9)
            B = B * np.power(10, 5)
            t1 = self.W / f 
            t2 = self.time_l - t1
            if t2 < 1e-4:
                t2 = 1e-4
            LBd = self.L / B
            if (LBd / t2) > 60:
                temp = 60
            else:
                temp = LBd / t2
            e = self.C * self.W * np.power(f, 2) + self.N * t2 / self.h * (np.power(2, temp) - 1)
            return e
        retf = 0;
        retB = 0;
        minE = 10000000
        for f in self.fList:
            for B in self.BList:
                input = [f, B]
                Esum = get_E(input)
                if Esum < minE:
                    minE = Esum
                    retf = f
                    retB = B
        t1 = self.W / retf
        t2 = self.time_l - t1
        return retf, retB, t1, t2, minE



    def config(self, config):
        config = self.download(config)

        self.epoch = config.num_epochs_local_training
        self.batch_size = config.batch_size_local_training
        self.lr = config.lr_local_training
        self.num_models = config.num_servers
        self.schedule_type = config.schedule_type

        self.device = config.device

        self.models = []
        self.data = []
        self.dataloader = []
        self.data_index = []
        self.train_model_ids = []
    
        self.num_classes = config.num_classes

        self.E_que = queue.Queue(maxsize=10)
        self.load_que = []

        self.dataset_ID = config.dataset_ID

        for i in range(10):
            enqueue(self.E_que, 0)
        
        for i in range(self.num_models):
            self.load_que.append(queue.Queue(maxsize=10))
            for j in range(10):
                enqueue(self.load_que[i], 0)
            
        

        # self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
        #                                 save_activations=False, group_norm_num_groups=None,
        #                                 freeze_bn=False, freeze_bn_affine=False, num_classes=config.num_classes).to(self.device)
        # self.local_model.eval()
        # self.optimizer = optim.SGD(self.local_model.parameters(), lr=config.lr_local_training)
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    # 下载所有模型全局模型参数
    def set_models(self, model_num):
        self.transload = []
        for model_id in range(model_num):
            if self.dataset_ID == "cifar10":
                model = ResNet_cifar(resnet_size=8, scaling=4,
                                            save_activations=False, group_norm_num_groups=None,
                                            freeze_bn=False, freeze_bn_affine=False, num_classes=self.num_classes).to(self.device)
            elif self.dataset_ID == "mnist":
                model = MnistCNN().to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss().to(self.device)
            self.models.append({"model_id": model_id,
                                "model": model, 
                                "optimizer": optimizer, 
                                "criterion": criterion})
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.transload.append(total_params)

    # 更新本轮收到的模型的参数
    def update_models(self, global_model_params:list, train_model_ids:list):
        self.train_model_ids = train_model_ids[self.client_id].copy()
        global_model_params = self.download(global_model_params)
        for mid in train_model_ids[self.client_id]:
            self.models[mid]["model"].load_state_dict(global_model_params[mid])
        # for model_param in global_model_params:
        #     for model in self.models:
        #         if model_param["model_id"] == model["model_id"]:
        #             model["model"].load_state_dict(model_param["model"])
        # global_model = self.download(global_model)
        # self.local_model.load_state_dict(global_model)

    # 初始化客户端数据
    def set_data(self, data):
        self.data.append(copy.deepcopy(data))

    def set_dataloader(self, dataloder):
        self.dataloader.append(dataloder)

    def set_data_index(self, data_index):
        self.data_index.append(copy.deepcopy(data_index))

    # 得到本轮客户端训练结果
    def get_result(self):
        return self.upload(self.report)
    
    def get_last_E(self):
        return self.last_E
    
    # 客户端训练
    def train_models(self):
        logging.info('Training on client #{}'.format(self.client_id))
        # print("Training on client #{}".format(self.client_id))
        self.report = Report(self)  
        if self.schedule_type == 'maxf':
            bf, bB, t1, t2, E = self.maxfschedule()
        elif self.schedule_type == 'search':
            bf, bB, t1, t2, E = self.search()
        else:
            bf, bB, t1, t2, E = self.schedule()
        self.last_E = E
        # enqueue(self.E_que, E)
        for mid in self.train_model_ids:
            # enqueue(self.load_que[mid], (bf - min(self.fList)) / (max(self.fList) - min(self.fList)) * 100)
            model = self.models[mid]     
            model_id = model["model_id"]
            model_net = model["model"]
            optimizer = model["optimizer"]
            criterion = model["criterion"]
            if self.dataset_ID == "cifar10":
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip()])
            elif self.dataset_ID == "mnist":
                transform_train = transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    # 随机水平翻转
                    transforms.RandomHorizontalFlip()])
            model_net.to(self.device)
            criterion.to(self.device)
            model_net.train()
            for epoch in range(self.epoch):
                
                shared_lock.acquire()
                self.dataloader[mid].load(self.data_index[mid])
                data_l = copy.deepcopy(self.dataloader[mid])
                shared_lock.release()

                data_loader = DataLoader(dataset=data_l, 
                                        batch_size=self.batch_size,  
                                        shuffle=True,
                                        num_workers=4)

                for data_batch in data_loader:
                    data, target = data_batch
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    data = transform_train(data)
                    _, output = model_net(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            local_weights = copy.deepcopy(model_net.state_dict())
            self.report.model_list.append(model_id)
            self.report.model_params.append({
                "model_id": model_id,
                "num_samples": len(self.data_index[model_id]),
                "local_weights": local_weights
            })

    def reset_state(self):
        for model_id in range(self.num_models):
            if self.dataset_ID == "cifar10":
                neo_model = ResNet_cifar(resnet_size=8, scaling=4,
                                            save_activations=False, group_norm_num_groups=None,
                                            freeze_bn=False, freeze_bn_affine=False, num_classes=self.num_classes).to(self.device)
            elif self.dataset_ID == "mnist":
                neo_model = MnistCNN().to(self.device)
            self.models[model_id]["model"].load_state_dict(neo_model.state_dict())
         

    def get_state(self, server_id):
        states = []
        states.extend(list(self.E_que.queue))
        states.extend(list(self.load_que[server_id].queue))
        states.append(self.workload[server_id])
        return states
    
    def get_state_params(self, server_id):
        all_params = copy.deepcopy(self.models[server_id]["model"].state_dict())
        params = torch.cat([value.view(-1) for value in all_params.values()])
        return params
        




# 用来存放训练结果
class Report(object):
    """Federated learning client report."""

    def __init__(self, client):
        self.client_id = client.client_id
        self.model_list = []
        self.model_params = []





        


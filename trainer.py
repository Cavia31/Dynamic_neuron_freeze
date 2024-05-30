import freezable_modules.modules as fnn
from utils import accuracy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randint
from csv import DictWriter
from time import time



class Trainer():

    def __init__(self, train_config, model:fnn.FreezableModule, result_file:DictWriter, train_set:DataLoader, test_set:DataLoader, valid_set=None, device=None):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.valid_set = valid_set
        for attr_name in train_config:
            # set attributes such as epochs, optimizer, device, method
            self.__setattr__(attr_name, train_config[attr_name])

        if device:
            self.device = device

        # Put the model in the correct device
        self.model.to(self.device)

        # Initialize the optimizer
        self.lr_scheduler = None
        if self.optimizer == "sgd":
            self.optimizer = optim.SGD(model.parameters(), **self.optim_args)
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **self.scheduler_args)
        elif self.optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), **self.optim_args)

        # Initialize the loss function
        if self.loss_fn == "cross_entropy":
            self.loss_fn = F.cross_entropy
        elif self.loss_fn == "mse":
            self.loss_fn = F.mse_loss

        if self.method == "semi_random":
            self.freeze = self.model.n_random_freezing_matrices(self.method_config["ratio"])
        elif self.method == "proportional":
            self.freeze = self.model.n_proportional_matrices(self.method_config["ratio"])
            self.i = 0
        elif self.method == "velocity":
            self.model.neq_mode(False, torch.tensor(self.method_config["momentum"], device=device))

        self.result_file = result_file
        print(self.device)

    def train_step(self,epoch):
        self.model.train()
        epoch_loss = 0.
        n = torch.zeros(1, device=self.device)
        with tqdm(
            total=len(self.train_set),
            desc="Train Epoch #{}".format(epoch + 1)
        ) as t:
            for idx, (data,target) in enumerate(self.train_set):
                self.model.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                out = self.model(data)
                loss = self.loss_fn(out, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss
                n = idx
                t.set_postfix({
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                t.update()
                
            n += 1
            epoch_loss = epoch_loss.detach()/n
            test_loss, acc1, acc5 = self.test()
            if self.lr_scheduler:
                    self.lr_scheduler.step()
        return epoch_loss.detach(), test_loss, acc1, acc5
    
    def reset_optim(self):
        if isinstance(self.optimizer, optim.SGD):
            # Reset the momentum without reseting the lr_scheduler
            state = self.optimizer.state_dict()
            state['state'] = {}
            self.optimizer.load_state_dict(state)
    
    def update_backward_method(self, epoch):

        if self.method == "full_random":
            if epoch % self.method_config['epoch_change'] != 0:
                return self.n_unfrozen
            else:
                print("Changing backward update scheme")
            mat,n_unfrozen = self.model.random_freezing_matrix(self.method_config['ratio'])
            self.model.set_freezing_matrix(mat)
            self.reset_optim()
            return n_unfrozen
        elif self.method == "semi_random":
            if epoch % self.method_config['epoch_change'] != 0:
                return self.n_unfrozen
            else:
                print("Changing backward update scheme")
            i = randint(0, len(self.freeze)-1)
            self.model.set_freezing_matrix(self.freeze[i][0])
            self.reset_optim()
            return self.freeze[i][1]
        elif self.method == "proportional":
            if epoch % self.method_config['epoch_change'] != 0:
                return self.n_unfrozen
            else:
                print("Changing backward update scheme")
            self.model.set_freezing_matrix(self.freeze[self.i][0])
            uf = self.freeze[self.i][1]
            self.i = (self.i + 1)%len(self.freeze)
            self.reset_optim()
            return uf
        elif self.method == "velocity":
            if epoch % self.method_config['epoch_change'] != 0:
                return self.n_unfrozen
            else:
                print("Changing backward update scheme")
            self.valid()
            n_unfrozen = self.model.update_neq(self.method_config['eps'])
            self.reset_optim()
            return n_unfrozen

    def train(self):
        self.model.train()
        for e in range(self.epochs):
            self.n_unfrozen = self.update_backward_method(e)
            beg = time()
            loss, test_loss, acc1, acc5 = self.train_step(e)
            end = time()
            res_dict = {
                'epoch': e+1,
                'train_loss': loss.item(),
                'test_loss': test_loss.item(),
                'acc1': acc1.item(),
                'acc5': acc5.item(),
                'unfrozen_neurons': self.n_unfrozen,
                'total_params': self.model.get_total_parameters(),
                'unfrozen_params': self.model.get_unfrozen_parameters(),
                'epoch_time': end-beg
            }
            self.result_file.writerow(res_dict)
            print('epoch {}, train loss {}, test loss {}, test acc1 {}, test acc5 {}'.format(e+1, loss.item(), test_loss.item(), acc1.item(), acc5.item()))

    def test(self):
        self.model.eval()
        test_loss = 0
        acc1,acc5 = 0,0
        with tqdm(
            total=len(self.test_set),
            desc="Test"
        ) as t:
            with torch.no_grad():
                for data, target in self.test_set:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    haty = self.model(data)

                    test_loss += self.loss_fn(haty, target)
                    dacc1,dacc5 = accuracy(haty, target, (1,5))
                    acc1 += dacc1
                    acc5 += dacc5

                    t.update()
                test_loss /= len(self.test_set)
                acc1 /= len(self.test_set)
                acc5 /= len(self.test_set)
        return test_loss.detach(), acc1, acc5

    def valid(self):
        self.model.neq_mode(True)
        test_loss = 0
        acc1,acc5 = 0,0
        with tqdm(
            total=len(self.valid_set),
            desc="Validation"
        ) as t:
            with torch.no_grad():
                for data, target in self.valid_set:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    haty = self.model(data)

                    test_loss += self.loss_fn(haty, target)
                    dacc1,dacc5 = accuracy(haty, target, (1,5))
                    acc1 += dacc1
                    acc5 += dacc5

                    t.update()
                test_loss /= len(self.valid_set)
                acc1 /= len(self.valid_set)
                acc5 /= len(self.valid_set)
        self.model.neq_mode(False)
        return test_loss.detach(), acc1, acc5
import torch
import torch.nn as nn
import pandas as pd

from graph import Graph
from rule import Rule
from device import device

class KnowledgeGraph:
    def __init__(self, data_path:str, add_reverse=True):
        data = pd.read_csv(data_path, sep='\t', header=None)

        total_entity = data[0].append(data[2])
        self.n_entity = len(total_entity.unique())
        self.n_predicate = len(data[1].unique())
        self.predicate_offset = 0
        if (add_reverse):
            self.predicate_offset = self.n_predicate
        self.graph = Graph(self.n_entity, self.n_predicate + self.predicate_offset)
        self.entity2id = {}
        self.id2entity = {}
        self.predicate2id = {}
        self.id2predicate = {}
        self.triplets = []

        length = len(data[0])
        cur_ent = 0
        cur_pre = 0
        for i in range(length):
            for j in [0, 2]:
                if (data[j][i] not in self.entity2id):
                    self.entity2id[data[j][i]] = cur_ent
                    self.id2entity[cur_ent] = data[j][i]
                    cur_ent = cur_ent + 1
                
            if (data[1][i] not in self.predicate2id):
                self.predicate2id[data[1][i]] = cur_pre
                if (self.predicate_offset != 0):
                    self.predicate2id['rev-' + str(data[1][i])] = cur_pre + self.predicate_offset
                    self.id2predicate[cur_pre + self.predicate_offset] = 'rev-' + str(data[1][i])
                self.id2predicate[cur_pre] = data[1][i]
                cur_pre = cur_pre + 1
            
            self.graph.add_edge(self.entity2id[data[0][i]], self.entity2id[data[2][i]], self.predicate2id[data[1][i]])
            if (self.predicate_offset != 0):
                self.graph.add_edge(self.entity2id[data[2][i]], self.entity2id[data[0][i]], self.predicate2id[data[1][i]] + self.predicate_offset)
            self.triplets.append([self.entity2id[data[0][i]], self.predicate2id[data[1][i]], self.entity2id[data[2][i]], 1.0])
            self.triplets.append([self.entity2id[data[2][i]], self.predicate2id[data[1][i]] + self.predicate_offset, self.entity2id[data[0][i]], 1.0])
    
    def n_predicate_total(self):
        return self.n_predicate + self.predicate_offset
    
    def n_predicate_original(self):
        return self.n_predicate

    def construct_model(self, rule: list):
        model = Rule(self.n_predicate_total(), len(rule) - 1)
        for i in range(model.n_step):
            model.R[i].R.requires_grad_(False)
        for i in range(model.n_step):
            for j in range(model.n_predicate):
                model.R[i].R[j] = float('-inf')
        for i in range(0, len(rule) - 1):
            model.R[i].R[self.predicate2id[rule[i + 1]]] = 1.0
        return model.to(device)

    def check_rule(self, rule: list):
        """
        rule = [head_name, body0_name, body1_name, ...]
        """
        total = 0
        cnt = 0

        model = self.construct_model(rule)

        mask0 = torch.ones((1, self.graph.n_vertex)).to(device)
        a = (model(self.graph) + 0.3).type(torch.long).type(torch.float32).T
        for v_from in range(self.graph.n_vertex):
            mask1 = (self.graph.A[self.predicate2id[rule[0]]].T[v_from]+0.3).type(torch.long).type(torch.float32)
            ans = a[v_from]
            total += int(mask0.matmul(ans.T))
            cnt += int(mask1.matmul(ans.T))

        num = int(torch.sum(self.graph.A[self.predicate2id[rule[0]]]))


        return num, total, cnt
    
    def print_rule(self, rule: list):
        num, total, cnt = self.check_rule(rule)
        print("#True: "+ str(num))
        print("#Predicted True: " + str(total))
        print("#True And Predicted True: " + str(cnt))
        print()
        if (total != 0):
            print("#True Among Predicted True: " + str(cnt / total))
            print()
        if (num != 0):
            print("#Predicted True Among True: " + str(cnt / num))
            print()
    
    def print_model(self, rule: Rule, head_name=None):
        temp = torch.zeros(rule.n_step)
        for i in range(rule.n_step):
            temp[i] = torch.argmax(rule.R[i].R, dim=0)
        print("Rule: ")
        s = '['
        l = []
        if (head_name != None):
            l.append(head_name)
        for i in range(temp.shape[0]):
            if (temp[i].tolist() >= self.n_predicate):
                s = s + str(self.id2predicate[temp[i].tolist()]) + ', '
                l.append(self.id2predicate[temp[i].tolist()])
            else:
                s = s + str(self.id2predicate[temp[i].tolist()]) + ', '
                l.append(self.id2predicate[temp[i].tolist()])
        s += ']'
        print(s)

        print("Score: ")
        x = torch.zeros(temp.shape)
        for i in range(rule.n_step):
            x[i] = torch.max(rule.R[i]._regularized_R())
        print(x.tolist())
        if (head_name != None):
            self.print_rule(l)

        print()
        print("Parameter Matrix: ")
        param = torch.zeros((rule.n_step, rule.n_predicate))
        for i in range(rule.n_step):
            param[i] = torch.softmax(rule.R[i].R, dim=0)
        print(param.T)
        print()
    
    def output_model(self, rule: Rule, head_name=None):
        R = rule.R
        temp = torch.zeros(rule.n_step)
        for i in range(rule.n_step):
            temp[i] = torch.argmax(rule.R[i].R, dim=0)
        l = []
        if (head_name != None):
            l.append(head_name)
        for i in range(temp.shape[0]):
            if (temp[i].tolist() >= self.n_predicate):
                l.append(self.id2predicate[temp[i].tolist()])
            else:
                l.append(self.id2predicate[temp[i].tolist()])
        return l
from pickletools import optimize
from tempfile import tempdir
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import pandas
import os
from tqdm import tqdm
import random

from graph import Graph
from knowledgegraph import KnowledgeGraph
from rule import Rule
from device import device

def loss_one_batch(model: Rule, KG: KnowledgeGraph, pos_triplets, neg_triplets, power_ratio):
    l0 = torch.zeros(())
    ans0 = model(KG.graph, 2.0 * power_ratio)
    ans1 = ans0
    l1 = torch.sum(ans1 * (neg_triplets + pos_triplets))
    l0 = torch.sum(ans0 * pos_triplets)
    l2 = l1.detach()
    loss = -l0 / l1 - l0 / (torch.sum(pos_triplets) + 1)
    # loss = -l0 / l1
    # loss = -(l0 / len(batch_triplets)) / (l1 / (KG.graph.n_vertex * KG.graph.n_vertex))
    loss.requires_grad_(True)
    return loss

def train(rule_body_len, target_predicate: int, KG: KnowledgeGraph, BATCH_SIZE: int, pos_triplets, neg_triplets, EPOCH: int, lr=0.3, model=None, decay_st=50, randinit=False):
    if (model == None): 
        model = Rule(KG.graph.n_edge_type, rule_body_len, randinit).to(device)
    rec = 0.0

    target_triplets = list()
    offset = KG.predicate_offset
    for triplet in KG.triplets:
        if (triplet[1] == target_predicate):
            target_triplets.append(triplet)
    if (target_predicate >= KG.n_predicate_original()):
        offset = -offset
    
    power_ratio = 1.0
    for epoch in range(EPOCH):
        lrdiv = torch.max(model._regularized_R(3))
        if not lr / lrdiv >= 0:
            print(model._regularized_R(3))
        cnt = 0
        # for triplet in batch:
        #     KG.graph.mask_edge(triplet[0], triplet[2], triplet[1])
        #     KG.graph.mask_edge(triplet[2], triplet[0], offset + triplet[1])
        cnt += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr / lrdiv)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr / lrdiv)
        optimizer.zero_grad()
        loss = loss_one_batch(model, KG, pos_triplets, neg_triplets, power_ratio=power_ratio)
        loss.backward()
        rec = loss.item()
        optimizer.step()
        # KG.graph.restore_masked_edge()
    if (epoch > decay_st):
        power_ratio = 1.5
        
    # print(rec)
    return model

def train_step_by_step(rule_body_len, target_predicate: int, KG: KnowledgeGraph, BATCH_SIZE: int, EPOCH: int, lr=0.3, model=None, power_ratio = 1.0):
    if (model == None): 
        model = Rule(KG.graph.n_edge_type, rule_body_len)

    target_triplets = list()
    for triplet in KG.triplets:
        if (triplet[1] == target_predicate):
            target_triplets.append(triplet)
    offset = KG.predicate_offset
    if (target_predicate >= KG.n_predicate_original()):
        offset = -offset
    
    with tqdm(range(EPOCH), ncols=80) as _tqdm:
        temp = 1.0
        for epoch in _tqdm:
            lrdiv = temp
            temp = 1.0
            for step in range(model.n_step):
                temp *= torch.max(model.R[step]._regularized_R())
                random.shuffle(target_triplets)
                batches = [target_triplets[i:i + BATCH_SIZE] for i in range(0, len(target_triplets), BATCH_SIZE)]
                cnt = 0
                for batch in batches:
                    # for triplet in batch:
                    #     KG.graph.mask_edge(triplet[0], triplet[2], triplet[1])
                    #     KG.graph.mask_edge(triplet[2], triplet[0], offset + triplet[1])
                    cnt += 1
                    optimizer = torch.optim.Adam(model.R[step].parameters(), lr=lr / lrdiv)
                    optimizer.zero_grad()
                    loss = loss_one_batch(model, KG, batch, power_ratio=power_ratio)
                    loss.backward()
                    optimizer.step()
                    _tqdm.set_postfix_str(str(cnt) + '/' + str(len(batches)) + 'loss = ' + str(loss.item()))
                    # KG.graph.restore_masked_edge()
            
    return model

def train_pow_decay(rule_body_len, target_predicate: int, KG: KnowledgeGraph, BATCH_SIZE: int, EPOCH: int, lr=0.3, model=None, decay_st=50, randinit=False):
    if (model == None): 
        model = Rule(KG.graph.n_edge_type, rule_body_len, randinit).to(device)
    target_triplets = list()
    offset = KG.predicate_offset
    for triplet in KG.triplets:
        if (triplet[1] == target_predicate):
            target_triplets.append(triplet)
    if (target_predicate >= KG.n_predicate_original()):
        offset = -offset
    
    power_ratio = 1.0
    temp = 1.0
    for epoch in range(EPOCH):
        lrdiv = temp
        temp = 1.0
        for step in range(model.n_step):
            temp *= torch.max(model.R[step]._regularized_R(power_ratio))
            random.shuffle(target_triplets)
            batches = [target_triplets[i:i + BATCH_SIZE] for i in range(0, len(target_triplets), BATCH_SIZE)]
            cnt = 0
            for batch in batches:
                # for triplet in batch:
                #     KG.graph.mask_edge(triplet[0], triplet[2], triplet[1])
                #     KG.graph.mask_edge(triplet[2], triplet[0], offset + triplet[1])
                cnt += 1
                optimizer = torch.optim.Adam(model.R[step].parameters(), lr=lr / lrdiv)
                optimizer.zero_grad()
                loss = loss_one_batch(model, KG, batch, power_ratio=power_ratio)
                loss.backward()
                optimizer.step()
                # KG.graph.restore_masked_edge()
        if (epoch > decay_st):
            power_ratio = 1.5
            
    return model
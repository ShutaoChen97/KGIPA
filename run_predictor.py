# -*- coding: utf-8 -*-
import os
import sys
import csv
import torch
import pickle
import argparse
import numpy as np

from conf import *
from FeatureExtract import *
from model import KEIPA

device = torch.device("cpu")

def SeqMask(seqlen_list, seqlen_lim, device):
    seq_mask = torch.zeros((len(seqlen_list), seqlen_lim), dtype=torch.float32)
    for i, length in enumerate(seqlen_list):
        seq_mask[i, :length] = 1.0
    seq_mask = seq_mask.to(device)
    return seq_mask
       
def KEIPA_Predict(x_seq_pep, x_ss_pep, x_physical_pep, x_dense_pep, x_pretrain_pep, 
                  x_seq_pro, x_ss_pro, x_physical_pro, x_dense_pro, x_pretrain_pro, 
                  x_edge_pep, x_edge_pro, peplen, prolen):
    # Loading the trained KEIPA model and its parameters
    model = KEIPA().to(device)
    model_name = os.path.join(MODEL_FOLD, 'KEIPA_Saved_Model.pth')
    model.load_state_dict(torch.load(model_name, map_location='cpu', weights_only=True))
    
    '''mask generate'''
    x_seqmask_pep = SeqMask(peplen, 50, device)
    x_seqmask_pro = SeqMask(prolen, 800, device)
    
    '''round up to the nearest integer'''
    x_edge_pep = torch.round(x_edge_pep)
    x_edge_pro = torch.round(x_edge_pro)
    
    model.eval()
    ouputs_type, ouputs_all = model(x_seq_pep, x_ss_pep, x_physical_pep, x_dense_pep, x_pretrain_pep, 
                                    x_seq_pro, x_ss_pro, x_physical_pro, x_dense_pro, x_pretrain_pro, 
                                    x_edge_pep, x_edge_pro, x_seqmask_pep, x_seqmask_pro)
    
    ouputs_type = ouputs_type.detach().numpy()
    ouputs_all = ouputs_all.detach().numpy()
    ouputs_type = ouputs_type[0, :peplen[0], :prolen[0], :]  # (peplen, prolen, 7)
    ouputs_all = ouputs_all[0, :peplen[0], :prolen[0]]  # (peplen, prolen)
    
    # Prediction of peptide and protein binding residues distinguishing non-covalent bond types
    ouputs_type_pep, ouputs_type_pro = [], []
    for type in range(ouputs_type.shape[-1]):
        pep_pred = np.max(ouputs_type[:, :, type], 1)
        pro_pred = np.max(ouputs_type[:, :, type], 0)
        ouputs_type_pep.append(np.array(pep_pred))
        ouputs_type_pro.append(np.array(pro_pred))
    ouputs_type_pep = np.array(ouputs_type_pep)
    ouputs_type_pro = np.array(ouputs_type_pro)
    
    # Prediction of peptide and protein binding residues without distinguishing between non-covalent bond types
    ouputs_all_pep = np.max(ouputs_all, 1)  # (peplen,)
    ouputs_all_pro = np.max(ouputs_all, 0)  # (prolen,)
    
    return ouputs_type, ouputs_all, ouputs_type_pep, ouputs_type_pro, ouputs_all_pep, ouputs_all_pro

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-uip', type=str)
    args = parser.parse_args()
    uip = args.uip
    
    pep_uip = os.path.join(uip, 'Peptide_Seq.fasta')
    pro_uip = os.path.join(uip, 'Protein_Seq.fasta')

    pep_seq_list, pro_seq_list = [], []
    for i in open(pep_uip):
        if i[0] != '>':
            pep_seq_list.append(i.strip())
    for i in open(pro_uip):
        if i[0] != '>':
            pro_seq_list.append(i.strip())
            
    # Parallel extraction of protein and peptide features
    (protein_features, peptide_features) = parallel_feature_extract(pro_seq_list, pep_seq_list, uip)

    # Split features
    x_seq_pro, x_physical_pro, x_dense_pro, x_ss_pro, x_pretrain_pro, x_edge_pro = protein_features
    x_seq_pep, x_physical_pep, x_dense_pep, x_ss_pep, x_pretrain_pep, x_edge_pep = peptide_features
    
    pep_len = [len(pep_seq_list[0])]
    pro_len = [len(pro_seq_list[0])]
    
    # KEIPA prediction
    ouputs_type, ouputs_all, ouputs_type_pep, ouputs_type_pro, ouputs_all_pep, ouputs_all_pro = KEIPA_Predict(\
        x_seq_pep, x_ss_pep, x_physical_pep, x_dense_pep, x_pretrain_pep, 
        x_seq_pro, x_ss_pro, x_physical_pro, x_dense_pro, x_pretrain_pro, 
        x_edge_pep, x_edge_pro, pep_len, pro_len)
    
    # Save results to pkl file
    result = [pep_seq_list[0], pro_seq_list[0], ouputs_type, ouputs_all, 
              ouputs_type_pep, ouputs_type_pro, ouputs_all_pep, ouputs_all_pro]
    result_filepath = open(os.path.join(uip, 'result.pkl'), 'wb')
    pickle.dump(result, result_filepath, protocol=2)
    result_filepath.close()


# -*- coding: utf-8 -*-
"""
Created on 2022-11-09 20:58

@author: Shutao Chen
"""
# %%
import time
import argparse
import numpy as np
from Bio import SeqIO
import sys, os, subprocess
from multiprocessing import Pool

from conf import *
sys.path.append('Utility')

complet_n = 0

def run_simple_search(fd):
    protein_name = fd.split('.')[0]
    global complet_n
    complet_n += 1
    print('Processing:%s---%d' % (protein_name, complet_n*1))
    outfmt_type = 5
    num_iter = 10
    evalue_threshold = 0.001
    fasta_file = Profile_HOME + protein_name + '.fasta'
    pssm_file = Profile_HOME + protein_name + '.pssm'
    if os.path.isfile(pssm_file):
        pass
    else:
        cmd = ' '.join([BLAST,
                        '-query ' + fasta_file,
                        '-db ' + BLAST_DB,
                        '-evalue ' + str(evalue_threshold),
                        '-num_iterations ' + str(num_iter),
                        '-outfmt ' + str(outfmt_type),
                        '-out_ascii_pssm ' + pssm_file,  # Write the pssm file
                        '-num_threads ' + '1']
                       )
        # subprocess.call(cmd, shell=True)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

def generateMSA(file_path):
    seq_DIR = ['Protein_Seq.fasta']
    pssm_dir = Profile_HOME

    pool = Pool(1)
    results = pool.map(run_simple_search, seq_DIR)
    pool.close()
    pool.join()

def Read_SeqID(FilePath):
    f0 = open(FilePath, 'r')
    lines = f0.readlines()
    count = 0
    info1 = []
    info2 = []
    for line in lines:
        if count % 2 == 0:
            info1.append(line.strip('\n').strip('>'))
        else:
            info2.append(line.strip('\n'))
        count += 1
    f0.close()
    return info1, info2

def get_protein_blosum(proteinseq):
    blosum62 = {}
    blosum_reader = open(BLOSUM62_FOLD, 'r')
    count = 0
    for line in blosum_reader:
        count = count + 1
        if count <= 7:
            continue
        line = line.strip('\r').split()
        blosum62[line[0]] = [float(x) for x in line[1:21]]
        
    protein_lst = []
    for aa in proteinseq:
        aa = aa.upper()
        # 这个地方要检查一下sequence中的氨基酸是不是都为标准氨基酸！
        if aa not in blosum62.keys():
            aa = 'X'
        protein_lst.append(blosum62[aa])
    return np.asarray(protein_lst)

def read_pssm(pssm_file):
    with open(pssm_file, 'r') as f:
        lines = f.readlines()
        lines = lines[3:-6]
        pro_seq = []
        mat = []
        for line in lines:
            tmp = line.strip('\n').split()
            pro_seq.append(tmp[1])
            tmp = tmp[2:22]
            mat.append(tmp)
        mat = np.array(mat)
        mat = mat.astype(float)
    return pro_seq, mat

def pssm_feature_extract(prolist, uip):
    global BLAST
    global BLAST_DB
    BLAST = BLAST_FOLD
    BLAST_DB = BLAST_DB_FOLD

    file_path = uip + 'Protein_Seq.fasta'
    
    global Profile_HOME
    Profile_HOME = uip
    generateMSA(file_path)
    
    proabb, prosequnique = Read_SeqID(uip + 'Protein_Seq.fasta')
    
    protein_dense_feature_dict = {}
    for tmp in range(len(prosequnique)):
        proseqtmp = prosequnique[tmp]
        if os.path.exists(os.path.join(uip, 'Protein_Seq.pssm')):
            pssmfeature = read_pssm(os.path.join(uip, 'Protein_Seq.pssm'))[1]
        else:
            pssmfeature = get_protein_blosum(proseqtmp)
        protein_dense_feature_dict[prosequnique[tmp]] = pssmfeature
    return protein_dense_feature_dict

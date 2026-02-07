import os
import re
import time
import torch
import subprocess
import numpy as np
from conf import *
from torch import tensor
from distribution_2_map import dist
from PSSMExtract import pssm_feature_extract
from transformers import T5EncoderModel, T5Tokenizer

def set_permissions(file_path):
    """Set file permissions to 777."""
    permissions = 0o777
    os.chmod(file_path, permissions)

def run_script(command):
    """Run a shell script and wait for it to complete."""
    process = subprocess.Popen(command, shell=True)
    process.wait()
    return process.returncode

def batch_feature(seqlist, featuredict, seq_max_length, mode):
    featurelist = []
    for tmp in range(len(seqlist)):
        feature = featuredict[seqlist[tmp]]
        if len(feature) < seq_max_length:
            diff = seq_max_length - len(feature)
            diffarr = np.zeros((diff, feature.shape[1]))
            featureuse = np.vstack((feature, diffarr))
        else:
            featureuse = feature[0:seq_max_length, ]
        featurelist.append(featureuse)
    if mode == 'float':
        featuretensor = torch.as_tensor(np.array(featurelist)).float()
    else:
        featuretensor = torch.as_tensor(np.array(featurelist)).long()
    return featuretensor

# Sequence Integer Coding 
def sequence_feature_extract(seqlist, seq_max_length):
    AminoAcidDic = dict(A=1, C=2, E=4, D=5, G=6, F=7, I=8, H=9, K=10, M=11, L=12, 
                        N=14, Q=15, P=16, S=17, R=18, T=20, W=21, V=22, Y=23, X=24)
    sequence_feature_dict = {}
    for tmp1 in range(len(seqlist)):
        seq_tmp_feature = []
        for tmp2 in range(seq_max_length):
            if tmp2 < len(seqlist[tmp1]):
                try:
                    value = AminoAcidDic[seqlist[tmp1][tmp2]]
                except:
                    value = 24
            else:
                value = 0
            seq_tmp_feature.append(value)
        sequence_feature_dict[seqlist[tmp1]] = np.array(seq_tmp_feature)
    out = batch_feature(seqlist, sequence_feature_dict, seq_max_length, 'long')
    return out

# Sequence Physicochemical Properties
def physical_feature_extract(seqlist, seq_max_length):
    AminoAcidDic = dict(A=1, R=6, N=4, D=5, C=3, Q=4, E=5, G=2, H=6, I=1, L=1, 
                        K=6, M=1, F=1, P=1, S=4, T=4, W=2, Y=4, V=1, X=7)
    physical_feature_dict = {}
    for tmp1 in range(len(seqlist)):
        seq_tmp_feature = []
        for tmp2 in range(seq_max_length):
            if tmp2 < len(seqlist[tmp1]):
                try:
                    value = AminoAcidDic[seqlist[tmp1][tmp2]]
                except:
                    value = 7
            else:
                value = 0
            seq_tmp_feature.append(value)
        physical_feature_dict[seqlist[tmp1]] = np.array(seq_tmp_feature)  
    out = batch_feature(seqlist, physical_feature_dict, seq_max_length, 'long')
    return out

# Sequence Disorder Trend Extraction
def iupred2a_feature_extract(uip, seqtype):
    ''' seqtype = 'Protein' or 'Peptide' '''
    # Setting input and output file paths
    seq_dir = os.path.join(uip, seqtype + '_Seq.fasta')
    out_long_dir = os.path.join(uip, seqtype + '_IUPred2A_long.txt')
    out_short_dir = os.path.join(uip, seqtype + '_IUPred2A_short.txt')
    out_glob_dir = os.path.join(uip, seqtype + '_IUPred2A_glob.txt')
    # Build command
    base_command = os.path.join(IUPRED2A_FOLD, 'iupred2a.sh')
    command_1 = f'{base_command} {seq_dir} long {out_long_dir}'
    command_2 = f'{base_command} {seq_dir} short {out_short_dir}'
    command_3 = f'{base_command} {seq_dir} glob {out_glob_dir}'
    # launch subprocesses for parallel execution
    processes = [
        subprocess.Popen(command_1, shell=True),
        subprocess.Popen(command_2, shell=True),
        subprocess.Popen(command_3, shell=True),
    ]
    # Wait for all child processes to complete
    for process in processes:
        process.wait()
    time.sleep(10)
    
    # Setting file permissions
    for output_file in [out_long_dir, out_short_dir, out_glob_dir]:
        set_permissions(output_file)
    return None

# Disordered Trend Result Extraction
def read_iupred2a_result(filepath, types):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if types == 'long' or types == 'short':
            lines = lines[7:]
        else:
            tmpidx = 0
            for line in lines:
                if line.strip('\n') == '# POS\tRES\tIUPRED2':
                    tmpidx += 1
                    break
                else:
                    tmpidx += 1
                    continue
            lines = lines[tmpidx:]
        pro_seq = []
        mat = []
        for line in lines:
            tmp = line.strip('\n').split()
            pro_seq.append(tmp[1])
            tmp = tmp[2]
            mat.append(tmp)
        mat = np.array(mat)
        mat = mat.astype(float)
    return pro_seq, mat

# Sequence Dense Feature Extraction (Disordered features + PSSM)
def dense_feature_extract(seqlist, uip, seqtype, seq_max_length):
    # Disordered features
    iupred2a_feature_extract(uip, seqtype)
    time.sleep(30)
    
    dense_feature_iupred2a_dict = {}
    types = ['long', 'short', 'glob']
    for tmp1 in range(len(seqlist)):
        seq_tmp_feature = []
        for type_tmp in types:
            filepath = os.path.join(uip, seqtype + '_IUPred2A_' + type_tmp + '.txt')
            seq_tmp_feature.append(read_iupred2a_result(filepath, type_tmp)[1])
        dense_feature_iupred2a_dict[seqlist[tmp1]] = np.array(seq_tmp_feature)
    
    # PSSM Feature
    if seqtype == 'Protein':
        dense_feature_pssm_dict = pssm_feature_extract(seqlist, uip)
    
    # Dense feature merging and output
    dense_feature_dict = {}
    for tmp in range(len(dense_feature_iupred2a_dict)):
        seq_tmp = list(dense_feature_iupred2a_dict)[tmp]
        iupred2a_tmp = dense_feature_iupred2a_dict[seq_tmp].T
        if seqtype == 'Protein':
            pssm_tmp = dense_feature_pssm_dict[seq_tmp]
            dense_feature_tmp = np.hstack((iupred2a_tmp, pssm_tmp))
        else:
            dense_feature_tmp = iupred2a_tmp
        
        if len(dense_feature_tmp) < seq_max_length:
            diff = seq_max_length - len(dense_feature_tmp)
            diffarr = np.zeros((diff, dense_feature_tmp.shape[1]))
            use = np.vstack((dense_feature_tmp, diffarr))
        else:
            use = dense_feature_tmp[0:seq_max_length, ]
        if use.shape[0] == seq_max_length:
            dense_feature_dict[seq_tmp] = np.array(use)
        else:
            print('error!')
            break
    
    out = batch_feature(seqlist, dense_feature_dict, seq_max_length, 'float')
    return out

def read_seq_and_index(filepath):
    f0 = open(filepath, 'r')
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

# Sequence Secondary Structure Feature Extraction
def secondary_structure_feature_extract(seqlist, uip, seqtype, seq_max_length):
    seq_dir = os.path.join(uip, seqtype + '_Seq.fasta')
    out_dir = os.path.join(uip, seqtype + '.out')
    
    # Build command
    base_command = os.path.join(SCRATCH_FOLD, 'bin', 'SCRATCH.sh')
    ss_command = f'{base_command} {seq_dir} {out_dir}'
    process = subprocess.Popen(ss_command, shell=True)
    process.wait()
    
    # Secondary structure analysis
    AminoAcidDic = dict(A=1, C=2, E=4, D=5, G=6, F=7, I=8, H=9, K=10, M=11, L=12, 
                        N=14, Q=15, P=16, S=17, R=18, T=20, W=21, V=22, Y=23, X=24)

    amino = list(AminoAcidDic)
    AminoAcidDic_SSPro = {}
    SSProEle = ['H', 'C', 'E']
    for tmp1 in range(len(AminoAcidDic)):
        aminouse = amino[tmp1]
        embeddinguse = AminoAcidDic[aminouse]
        for tmp2 in range(len(SSProEle)):
            SSProEleUse = SSProEle[tmp2]
            keyuse = aminouse + SSProEleUse
            if keyuse not in AminoAcidDic_SSPro:
                AminoAcidDic_SSPro[keyuse] = embeddinguse * 3 - (2 - tmp2)

    while os.path.isfile(os.path.join(uip, seqtype+'.out.ss')) == False:
        time.sleep(100)
        
    set_permissions(os.path.join(uip, seqtype+'.out.ss'))
    id2, seq = read_seq_and_index(os.path.join(seq_dir))
    id1, sspro = read_seq_and_index(os.path.join(uip, seqtype+'.out.ss'))

    ss_feature_dict = {}
    for tmp1 in range(len(seq)):
        sequse = seq[tmp1]
        ssprouse = sspro[tmp1]
        seq_tmp_feature = []
        for tmp2 in range(len(sequse)):
            seq_tmp_feature.append(AminoAcidDic_SSPro[sequse[tmp2]+ssprouse[tmp2]])
        seq_tmp_feature = np.array(seq_tmp_feature)
        
        if len(seq_tmp_feature) < seq_max_length:
            diff = seq_max_length - len(seq_tmp_feature)
            diffarr = np.zeros((diff, ), dtype=float)
            seq_tmp_feature = np.hstack((seq_tmp_feature, diffarr))
        elif len(seq_tmp_feature) > seq_max_length:
            seq_tmp_feature = seq_tmp_feature[0:seq_max_length]
        ss_feature_dict[seq[tmp1]] = np.array(seq_tmp_feature)
       
    out = batch_feature(seqlist, ss_feature_dict, seq_max_length, 'long')
    return out

# Sequence Pre-training Feature Extraction
def pretrain_feature_extract(seqlist, uip, seqtype, seq_max_length):    
    _, sequnique = read_seq_and_index(os.path.join(uip, seqtype + '_Seq.fasta'))
    
    seqexa = []
    for tmp1 in range(len(sequnique)):
        seqtmp = sequnique[tmp1]
        sequse = ''
        count = 0
        for tmp2 in seqtmp:
            sequse += tmp2
            count += 1
            if count == len(seqtmp):
                continue
            sequse += ' '
        seqexa.append(sequse)

    model = T5EncoderModel.from_pretrained(T5MODEL_FOLD)
    tokenizer = T5Tokenizer.from_pretrained(T5MODEL_FOLD)
    
    pretrain_feature_dict = {}
    for tmp in range(len(seqexa)):
        print('Current: ' + str(tmp) + '\n')
        seqlen = len(sequnique[tmp])
        sequence_Example = seqexa[tmp]
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = tokenizer(sequence_Example, return_tensors='pt')
        output = model(**encoded_input)
        result = output[0][0][1:seqlen+1]
        seq_tmp_feature = result.detach().numpy()
        if len(seq_tmp_feature) < seq_max_length:
            diff = seq_max_length - len(seq_tmp_feature)
            diffarr = np.zeros((diff, seq_tmp_feature.shape[1]))
            use = np.vstack((seq_tmp_feature, diffarr))
        else:
            use = seq_tmp_feature[0:seq_max_length, ]
        if use.shape[0] == seq_max_length:
            pretrain_feature_dict[sequnique[tmp]] = np.array(use)
        else:
            print('error!')
            break
    
    seq_tmp = list(pretrain_feature_dict)[0]
    pretrain_tmp = pretrain_feature_dict[seq_tmp][True, :]
    pretrain_feature_dict[seq_tmp] = pretrain_tmp[0]
    out = batch_feature(seqlist, pretrain_feature_dict, seq_max_length, 'float')
    return out.to(torch.float32)


# Sequence 3D Structure Feature Extraction
def edge_feature_extract(seqlist, uip, seqtype, seq_max_length):
    # trRosetta does not support the analysis of peptides with sequence lengths of less than 10 and therefore uses the fully populated form
    if (seqtype == 'Peptide') and (len(seqlist[0]) < 10):
        edge_feature = np.zeros((seq_max_length, seq_max_length), dtype=int)
        edge_feature[:len(seqlist[0]), :len(seqlist[0])] = 1
    else:
        # Define file paths
        seq_dir = os.path.join(uip, seqtype + '_Seq.fasta')
        a3m_dir = os.path.join(uip, seqtype + '_A3M.a3m')
        npz_dir = os.path.join(uip, seqtype + '_NPZ.npz')
        
        # A3M: Run the MSA generation script
        msa_command = f"{os.path.join(TRROSETTAX_FOLD, 'generate_msa.sh')} {seq_dir} {a3m_dir}"
        if run_script(msa_command) == 0:
            set_permissions(a3m_dir)
        else:
            print("Failed to generate A3M.")

        # NPZ: Run the prediction script
        npz_command = f"bash {os.path.join(TRROSETTAX_FOLD, 'predict.sh')} {a3m_dir} {npz_dir}"
        if run_script(npz_command) == 0:
            set_permissions(npz_dir)
        else:
            print("Failed to generate NPZ.")
        
        # Analysis distance and contact
        npz = np.load(npz_dir)
        img, cont = dist(npz)
        result = np.array(cont)
        
        if len(seqlist[0]) >= seq_max_length:
            edge_feature = result[:seq_max_length, :seq_max_length]
        else:
            edge_feature = np.zeros((seq_max_length, seq_max_length))
            edge_feature[:len(seqlist[0]), :len(seqlist[0])] = result        
    
    edge_feature_list = [edge_feature]
    out = torch.as_tensor(np.array(edge_feature_list)).float()
    return out

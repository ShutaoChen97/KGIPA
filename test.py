# %%
import numpy as np

# %%
# 无序提取
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

# %%
filepath = '/home/www/KEIPA-G/example/Protein_IUPred2A_short.txt'
types = 'short'

# %%
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



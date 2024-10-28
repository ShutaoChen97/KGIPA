# -*- coding: utf-8 -*-
from utils import *
import concurrent.futures

# 总的蛋白质特征提取脚本
def protein_feature_extract(prolist, uip):
    # 预先设定蛋白质的最长序列长度为800
    protein_max_length = 800
    
    # 定义各个特征提取的函数任务
    def extract_features():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_seq_pro = executor.submit(sequence_feature_extract, prolist, protein_max_length)
            future_physical_pro = executor.submit(physical_feature_extract, prolist, protein_max_length)
            future_dense_pro = executor.submit(dense_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_ss_pro = executor.submit(secondary_structure_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_pretrain_pro = executor.submit(pretrain_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_edge_pro = executor.submit(edge_feature_extract, prolist, uip, 'Protein', protein_max_length)
            
            # 等待所有特征提取完成并获取结果
            x_seq_pro = future_seq_pro.result()
            x_physical_pro = future_physical_pro.result()
            x_dense_pro = future_dense_pro.result()
            x_ss_pro = future_ss_pro.result()
            x_pretrain_pro = future_pretrain_pro.result()
            x_edge_pro = future_edge_pro.result()
            
        return x_seq_pro, x_physical_pro, x_dense_pro, x_ss_pro, x_pretrain_pro, x_edge_pro
    
    return extract_features()

# 总的多肽特征提取脚本
def peptide_feature_extract(peplist, uip):
    # 预先设定蛋白质的最长序列长度为800
    peptide_max_length = 50
    
    # 定义各个特征提取的函数任务
    def extract_features():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_seq_pep = executor.submit(sequence_feature_extract, peplist, peptide_max_length)
            future_physical_pep = executor.submit(physical_feature_extract, peplist, peptide_max_length)
            future_dense_pep = executor.submit(dense_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_ss_pep = executor.submit(secondary_structure_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_pretrain_pep = executor.submit(pretrain_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_edge_pep = executor.submit(edge_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            
            # 等待所有特征提取完成并获取结果
            x_seq_pep = future_seq_pep.result()
            x_physical_pep = future_physical_pep.result()
            x_dense_pep = future_dense_pep.result()
            x_ss_pep = future_ss_pep.result()
            x_pretrain_pep = future_pretrain_pep.result()
            x_edge_pep = future_edge_pep.result()
        
        return x_seq_pep, x_physical_pep, x_dense_pep, x_ss_pep, x_pretrain_pep, x_edge_pep
    
    return extract_features()

# 并行提取蛋白质和多肽的特征
def parallel_feature_extract(pro_seq_list, pep_seq_list, uip):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 并行调用蛋白质和多肽特征提取
        future_protein = executor.submit(protein_feature_extract, pro_seq_list, uip)
        future_peptide = executor.submit(peptide_feature_extract, pep_seq_list, uip)
        
        # 获取结果
        protein_features = future_protein.result()
        peptide_features = future_peptide.result()

    return protein_features, peptide_features
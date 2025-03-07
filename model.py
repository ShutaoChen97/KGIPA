# %%
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
class FCNet(nn.Module):
    """
    Simple Class for Non-linear Fully Connect Network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
            
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)

# %%
class InContextRepresentation(nn.Module):
    """
    Realisation of Intra-linguistic Contextual Representations
    """
    def __init__(self):
        super(InContextRepresentation, self).__init__()
        # Embedding layers for peptide and protein
        self.embed_seq = nn.Embedding(25, 128)
        self.embed_ss = nn.Embedding(73, 128)
        self.embed_two = nn.Embedding(8, 128)
        
        # Dense layers for peptide and protein
        self.dense_pep = nn.Linear(3, 128)
        self.dense_pro = nn.Linear(23, 128)
        self.dense_pretrain_pep = nn.Linear(1024, 128)
        self.dense_pretrain_pro = nn.Linear(1024, 128)
        
        # GCN layers
        self.gcn_pep_1 = GCNConv(640, 640)
        self.gcn_pep_2 = GCNConv(640, 640)
        self.gcn_pro_1 = GCNConv(640, 640)
        self.gcn_pro_2 = GCNConv(640, 640)
        
        # Fully connected layers for peptide and protein transformation
        decoder_hidden_i, decoder_hidden_o = 640, 128
        self.pep_trans = self._create_fc_net_layers(7, decoder_hidden_i, decoder_hidden_o)
        self.pro_trans = self._create_fc_net_layers(7, decoder_hidden_i, decoder_hidden_o)
        
    def forward(self, x_pep, x_ss_pep, x_2_pep, x_dense_pep, x_pretrain_pep,
                x_pro, x_ss_pro, x_2_pro, x_dense_pro, x_pretrain_pro,
                x_edge_pep, x_edge_pro, x_seqmask_pep, x_seqmask_pro):
        batch_size = x_pep.size(0)
        
        # Peptide embeddings
        pep_seq_emb = self.embed_seq(x_pep)
        pep_ss_emb = self.embed_ss(x_ss_pep)
        pep_2_emb = self.embed_two(x_2_pep)
        pep_dens1 = self.dense_pep(x_dense_pep)
        pep_dens2 = self.dense_pretrain_pep(x_pretrain_pep)
        
        # Protein embeddings
        pro_seq_emb = self.embed_seq(x_pro)
        pro_ss_emb = self.embed_ss(x_ss_pro)
        pro_2_emb = self.embed_two(x_2_pro)
        pro_dens1 = self.dense_pro(x_dense_pro)
        pro_dens2 = self.dense_pretrain_pro(x_pretrain_pro)
        
        # Concatenate embeddings
        encode_peptide = torch.cat([pep_seq_emb, pep_ss_emb, pep_2_emb, pep_dens1, pep_dens2], dim=-1)
        encode_protein = torch.cat([pro_seq_emb, pro_ss_emb, pro_2_emb, pro_dens1, pro_dens2], dim=-1)
        
        # Apply sequence masks
        encode_peptide *= x_seqmask_pep.unsqueeze(-1)
        encode_protein *= x_seqmask_pro.unsqueeze(-1)
        
        # GCN encoding
        encode_peptide_batch = self._apply_gcn(encode_peptide, x_edge_pep, x_seqmask_pep, batch_size, self.gcn_pep_1, self.gcn_pep_2)
        encode_protein_batch = self._apply_gcn(encode_protein, x_edge_pro, x_seqmask_pro, batch_size, self.gcn_pro_1, self.gcn_pro_2)
        
        # Stack encoded features
        feature_pep = torch.stack(encode_peptide_batch, dim=0)
        feature_pro = torch.stack(encode_protein_batch, dim=0)
        
        # Peptide and Protein Contextual Mapping
        peptide_vecs = self._apply_fc_trans(self.pep_trans, feature_pep, transpose_dims=(1, 2), unsqueeze_dim=3)
        protein_vecs = self._apply_fc_trans(self.pro_trans, feature_pro, transpose_dims=(1, 2), unsqueeze_dim=2)
        
        return peptide_vecs, protein_vecs

    def _create_fc_net_layers(self, n_layers, input_dim, output_dim):
        return nn.ModuleList([FCNet([input_dim, output_dim], act='ReLU', dropout=0.2) for _ in range(n_layers)])
    
    def _apply_gcn(self, encoded_data, edge_data, mask, batch_size, gcn_layer_1, gcn_layer_2):
        batch_encoded = []
        for i in range(batch_size):
            edge_index, _ = dense_to_sparse(edge_data[i])
            encoded_i = gcn_layer_1(encoded_data[i], edge_index)
            encoded_i = F.relu(encoded_i)
            encoded_i = gcn_layer_2(encoded_i, edge_index)
            encoded_i = F.relu(encoded_data[i] + encoded_i)
            batch_encoded.append(encoded_i * mask[i].unsqueeze(-1))
        return batch_encoded

    def _apply_fc_trans(self, fc_layers, feature, transpose_dims, unsqueeze_dim):
        return [fc_layer(feature).transpose(*transpose_dims).unsqueeze(unsqueeze_dim) for fc_layer in fc_layers]
    
# %%
class OutContextRepresentation(nn.Module):
    """
    Realisation of Extra-linguistic Contextual Representations
    """
    def __init__(self):
        super(OutContextRepresentation, self).__init__()
        # Predefined adjacency matrix
        adj = torch.Tensor(np.array([[1.   , -0.649, -0.268, -0.223, -0.038, -0.022, -0.008],
                                     [-0.649,  1.   , -0.203, -0.2  , -0.074, -0.057, -0.006],
                                     [-0.268, -0.203,  1.   ,  0.007, -0.028, -0.019, -0.   ],
                                     [-0.223, -0.2  ,  0.007,  1.   , -0.02 ,  0.011, -0.003],
                                     [-0.038, -0.074, -0.028, -0.02 ,  1.   , -0.007, -0.001],
                                     [-0.022, -0.057, -0.019,  0.011, -0.007,  1.   , -0.001],
                                     [-0.008, -0.006, -0.   , -0.003, -0.001, -0.001,  1.   ]]))
        self.edgs = torch.Tensor(adj).to(device)
        
        # Type probability calculation layers
        self.type_cal_prob = nn.ModuleList([
            weight_norm(nn.Linear(128, 1), dim=None) for _ in range(7)
        ])
        
        # Activate function for model output
        self.activate = nn.Sigmoid()
        
    def forward(self, peptide_vecs, protein_vecs, x_seqmask_pep, x_seqmask_pro):
        # Matrix multiplication for each bond type
        matmul_pair_all = torch.cat([torch.matmul(pep_vec, pro_vec).transpose(1, 2).transpose(2, 3).unsqueeze(-1)
                                     for pep_vec, pro_vec in zip(peptide_vecs, protein_vecs)], -1)

        interaction_result = torch.einsum('injkl,lm->injkm', matmul_pair_all, self.edgs)

        interaction_mask = torch.matmul(x_seqmask_pep.unsqueeze(-1), x_seqmask_pro.unsqueeze(1)).unsqueeze(-1)

        # Calculate type probabilities
        bond_type_probs = [self.activate(self.type_cal_prob[i](interaction_result[:, :, :, :, i])) * interaction_mask 
                      for i in range(7)]
        
        return bond_type_probs

# %%
class KGIPA(nn.Module):
    def __init__(self):
        super(KGIPA, self).__init__()
        
        # Initialize sub-modules for intra-linguistic contextual representations
        self.in_context = InContextRepresentation()
        
        # Initialize sub-modules for extra-linguistic contextual representations
        self.out_context = OutContextRepresentation()
        
    def forward(self, x_pep, x_ss_pep, x_2_pep, x_dense_pep, x_pretrain_pep,
                x_pro, x_ss_pro, x_2_pro, x_dense_pro, x_pretrain_pro,
                x_edge_pep, x_edge_pro, x_seqmask_pep, x_seqmask_pro):
        
        # Intra-linguistic Contextual Representation of peptide and protein
        peptide_vecs, protein_vecs = self.in_context(
            x_pep, x_ss_pep, x_2_pep, x_dense_pep, x_pretrain_pep,
            x_pro, x_ss_pro, x_2_pro, x_dense_pro, x_pretrain_pro,
            x_edge_pep, x_edge_pro, x_seqmask_pep, x_seqmask_pro
        )
        
        bond_type_probs = self.out_context(peptide_vecs, protein_vecs, x_seqmask_pep, x_seqmask_pro)
        
        # Pairwise Non-covalent Mechanism Profiling
        interaction_tensor_type = torch.cat(bond_type_probs, dim=-1)

        # Pairwise Non-covalent Interaction Prediction
        interaction_tensor = torch.max(interaction_tensor_type, dim=-1)[0]

        return interaction_tensor_type, interaction_tensor

# %%

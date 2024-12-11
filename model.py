import torch
import torch.nn as nn
from dgllife.model import AttentiveFPPredictor
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

class BioActHetModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, fp_size, latent_dim=128, dropout=0.3, num_layers=2, n_tasks=11):
        super(BioActHetModel, self).__init__()
        num_timesteps = 2
        graph_feat_size = latent_dim

        self.attentive_fp = AttentiveFPPredictor(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=1,
            dropout=dropout
        )

        self.compound_fc = nn.Sequential(
            nn.Linear(fp_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        # Adjust the final layer to output n_tasks
        self.pred_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, compound_fp):
        graph_emb = self.attentive_fp.gnn(g, node_feats, edge_feats)
        graph_emb = self.attentive_fp.readout(g, graph_emb)
        compound_emb = self.compound_fc(compound_fp)
        combined = torch.cat((graph_emb, compound_emb), dim=1)
        out = self.pred_fc(combined)
        return out


def load_model(model_path, n_tasks=11, device='cpu', load_state=True):
    from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    model = BioActHetModel(
        node_feat_size=atom_featurizer.feat_size('h'),
        edge_feat_size=bond_featurizer.feat_size('e'),
        fp_size=2048,
        latent_dim=128,
        dropout=0.3,
        num_layers=2,
        n_tasks=n_tasks
    ).to(device)

    if load_state:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    return model


def smiles_to_input(smiles):
    # Convert SMILES to graph and fingerprint
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()
    try:
        g = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    except:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return g, arr

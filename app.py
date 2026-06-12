"""
ChEMBL Bioactivity Predictor — Streamlit app

Przewiduje pIC50 dla podanego SMILES używając modeli MLP i GNN.

Uruchomienie:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator, rdchem
import pandas as pd

from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP

st.set_page_config(page_title="ChEMBL Przewidywania", page_icon="🧪", layout="wide")

BASE_DIR     = Path(__file__).parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR   = BASE_DIR / "models"
DEVICE       = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers, prev_dim = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentiveFPModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim,
                 n_layers, n_timesteps=3, dropout=0.3, target_dim=0):
        super().__init__()
        self.target_dim = target_dim
        self.gnn = AttentiveFP(
            in_channels=node_feat_dim, hidden_channels=hidden_dim,
            out_channels=hidden_dim,  edge_dim=edge_feat_dim,
            num_layers=n_layers,      num_timesteps=n_timesteps,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + target_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        if self.target_dim > 0 and hasattr(data, "tgt"):
            x = torch.cat([x, data.tgt], dim=-1)
        return self.head(x).squeeze(-1)


ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
DEGREES         = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES  = [-2, -1, 0, 1, 2]
NUM_HS_VALUES   = [0, 1, 2, 3, 4]
HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED, rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC,
]


def _one_hot(value, allowed_set, with_other=True):
    vec = [0] * (len(allowed_set) + (1 if with_other else 0))
    if value in allowed_set:
        vec[allowed_set.index(value)] = 1
    elif with_other:
        vec[-1] = 1
    return vec


def _atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(),        ATOM_TYPES)
        + _one_hot(atom.GetDegree(),        DEGREES)
        + _one_hot(atom.GetFormalCharge(),  FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(),    NUM_HS_VALUES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)
        + _one_hot(atom.GetChiralTag(),     CHIRAL_TAGS, with_other=False)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def _bond_features(bond):
    bt = bond.GetBondType()
    return [1.0 if bt == t else 0.0 for t in BOND_TYPES] + [
        float(bond.IsInRing()), float(bond.GetIsConjugated())
    ]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_idxs, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_idxs  += [[i, j], [j, i]]
        edge_feats += [bf, bf]
    if edge_idxs:
        edge_index = torch.tensor(edge_idxs,  dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 6), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@st.cache_resource(show_spinner="Ladowanie modeli...")
def load_resources():
    with open(FEATURES_DIR / "mlp_scaler.pkl",         "rb") as f: scaler = pickle.load(f)
    with open(FEATURES_DIR / "mlp_fp_selector.pkl",    "rb") as f: sel4   = pickle.load(f)
    with open(FEATURES_DIR / "mlp_target_encoder.pkl", "rb") as f: tgt_enc, top_tgts = pickle.load(f)

    use_ecfp6 = (FEATURES_DIR / "mlp_fp6_selector.pkl").exists()
    sel6 = None
    if use_ecfp6:
        with open(FEATURES_DIR / "mlp_fp6_selector.pkl", "rb") as f: sel6 = pickle.load(f)

    with open(MODELS_DIR / "target_stats.pkl", "rb") as f: sd = pickle.load(f)

    ck  = torch.load(MODELS_DIR / "mlp_model.pt", map_location="cpu", weights_only=False)
    cfg = ck["model_config"]
    mlp = MLP(cfg["in_dim"], cfg["hidden_dims"], cfg["dropout"]).eval()
    mlp.load_state_dict(ck["model_state_dict"])

    gk  = torch.load(MODELS_DIR / "gnn_model.pt", map_location="cpu", weights_only=False)
    gc  = gk["model_config"]
    gnn = AttentiveFPModel(
        node_feat_dim=gc["node_feat_dim"], edge_feat_dim=gc["edge_feat_dim"],
        hidden_dim=gc["hidden_dim"],       n_layers=gc["n_layers"],
        n_timesteps=gc["n_timesteps"],     dropout=gc["dropout"],
        target_dim=gc["target_dim"],
    ).eval()
    gnn.load_state_dict(gk["model_state_dict"])

    return {
        "scaler": scaler, "sel4": sel4, "sel6": sel6, "use_ecfp6": use_ecfp6,
        "tgt_enc": tgt_enc, "top_tgts": top_tgts,
        "tgt_stats": sd["tgt_stats"], "global_mean": sd["global_mean"], "global_std": sd["global_std"],
        "mlp": mlp, "gnn": gnn,
        "morgan4": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
        "morgan6": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048),
    }


def smiles_to_mlp_features(smiles, target_id, r):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hdon = Descriptors.NumHDonors(mol)
    hacc = Descriptors.NumHAcceptors(mol)
    desc = np.array([[
        mw, mol.GetNumAtoms(), Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol), hdon, hacc, logp,
        Descriptors.TPSA(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol), Descriptors.RingCount(mol),
        sum(a.GetIsAromatic() for a in mol.GetAtoms()) / mol.GetNumAtoms(),
        Descriptors.NumRotatableBonds(mol), Descriptors.NumSaturatedRings(mol),
        Descriptors.FractionCSP3(mol), Descriptors.MolMR(mol),
        int(mw > 500) + int(logp > 5) + int(hdon > 5) + int(hacc > 10),
    ]], dtype=np.float32)
    desc_sc = r["scaler"].transform(desc).astype(np.float32)
    fp4_sel = r["sel4"].transform(r["morgan4"].GetFingerprintAsNumPy(mol).astype(np.float32).reshape(1, -1)).astype(np.float32)
    if r["use_ecfp6"] and r["sel6"] is not None:
        fp6_sel = r["sel6"].transform(r["morgan6"].GetFingerprintAsNumPy(mol).astype(np.float32).reshape(1, -1)).astype(np.float32)
        fp_part = np.concatenate([fp4_sel, fp6_sel], axis=1)
    else:
        fp_part = fp4_sel
    tgt_label = target_id if target_id in r["top_tgts"] else "OTHER"
    tgt_vec   = r["tgt_enc"].transform([[tgt_label]]).astype(np.float32)
    return np.concatenate([desc_sc, fp_part, tgt_vec], axis=1)


def _denorm(pred_norm, target_id, r):
    mean, std = r["tgt_stats"].get(target_id, (r["global_mean"], r["global_std"]))
    return pred_norm * std + mean


@torch.no_grad()
def predict_mlp(smiles, target_id, r):
    feats = smiles_to_mlp_features(smiles, target_id, r)
    if feats is None:
        return None
    return _denorm(r["mlp"](torch.FloatTensor(feats)).item(), target_id, r)


@torch.no_grad()
def predict_gnn(smiles, target_id, r):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    tgt_label = target_id if target_id in r["top_tgts"] else "OTHER"
    graph.tgt  = torch.tensor(r["tgt_enc"].transform([[tgt_label]]).astype(np.float32), dtype=torch.float)
    feats = smiles_to_mlp_features(smiles, target_id, r)
    if feats is not None:
        graph.desc = torch.tensor(feats[:, :17], dtype=torch.float)
    return _denorm(r["gnn"](Batch.from_data_list([graph])).item(), target_id, r)


def pic50_to_ic50(pic50):
    return 10 ** (-pic50 + 9)


def activity_badge(pic50):
    if pic50 >= 8:   return "🟢 Silna (pIC50 >= 8)",      "#1a7a4a"
    elif pic50 >= 6: return "🟡 Umiarkowana (pIC50 6-8)", "#856404"
    else:            return "🔴 Słaba (pIC50 < 6)",        "#842029"


EXAMPLES = {
    "Erlotynib - EGFR":    ("C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",                          "CHEMBL203"),
    "Ibrutynib - BTK":     ("CC(C#C)n1cc(c2ncnc3[nH]ccc23)cc1-c1cccc(NC(=O)/C=C/CN(C)C)c1",       "CHEMBL5251"),
    "Ruksolitynib - JAK1": ("C=CC(=O)Nc1ccn(C2CCCC2)n1",                                          "CHEMBL2835"),
    "Gefitynib - EGFR":    ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",                     "CHEMBL203"),
    "Imatynib":            ("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",         "OTHER"),
    "Aspiryna":            ("CC(=O)Oc1ccccc1C(=O)O",                                               "OTHER"),
}

# UI
st.title("ChEMBL Przewidywania")
st.markdown("Podaj SMILES i wybierz białko docelowe, aby przewidziec **pIC50** modelem **MLP** lub **GNN**.")

r = load_resources()
all_targets = ["OTHER"] + sorted(r["top_tgts"])

if "smiles" not in st.session_state: st.session_state["smiles"] = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
if "target" not in st.session_state: st.session_state["target"] = "CHEMBL203"

with st.sidebar:
    st.header("Opcje")
    run_mlp = st.checkbox("MLP",               value=True)
    run_gnn = st.checkbox("GNN (AttentiveFP)", value=True)
    st.markdown("---")
    st.subheader("Przykładowe cząsteczki")
    for label, (smi, tgt) in EXAMPLES.items():
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state["smiles"] = smi
            st.session_state["target"] = tgt
            st.rerun()

col_in, col_mol = st.columns([3, 2], gap="large")

with col_in:
    st.subheader("Struktura chemiczna")
    smiles_val  = st.text_area("SMILES", value=st.session_state["smiles"], height=90, placeholder="np. CC(=O)Oc1ccccc1C(=O)O")
    tgt_idx     = all_targets.index(st.session_state["target"]) if st.session_state["target"] in all_targets else 0
    target_val  = st.selectbox("Białko docelowe (ChEMBL ID)", all_targets, index=tgt_idx)
    predict_btn = st.button("Przewiduj", type="primary", use_container_width=True)

with col_mol:
    st.subheader("Podgląd 2D")
    mol_preview = Chem.MolFromSmiles(smiles_val)
    if mol_preview:
        st.image(Draw.MolToImage(mol_preview, size=(380, 260)), use_container_width=True)
    else:
        st.warning("Nieprawidłowy SMILES - podgląd niedostępny.")

if predict_btn:
    mol = Chem.MolFromSmiles(smiles_val)
    if mol is None:
        st.error("Nieprawidłowy SMILES. Sprawdź składnię i spróbuj ponownie.")
    else:
        st.session_state["smiles"] = smiles_val
        st.session_state["target"] = target_val

        st.markdown("---")
        st.subheader(f"Wyniki predykcji  |  Target: `{target_val}`")

        rows = []
        if run_mlp:
            with st.spinner("MLP..."):
                p = predict_mlp(smiles_val, target_val, r)
            if p is not None:
                rows.append({"Model": "MLP", "pIC50": p, "IC50 [nM]": pic50_to_ic50(p)})

        if run_gnn:
            with st.spinner("GNN..."):
                p = predict_gnn(smiles_val, target_val, r)
            if p is not None:
                rows.append({"Model": "GNN", "pIC50": p, "IC50 [nM]": pic50_to_ic50(p)})

        if not rows:
            st.warning("Zaznacz co najmniej jeden model w panelu bocznym.")
        else:
            df      = pd.DataFrame(rows)
            avg_pic = float(np.mean(df["pIC50"]))

            col_tbl, col_met = st.columns([3, 2], gap="large")

            with col_tbl:
                df_disp = df.copy()
                df_disp["pIC50"]     = df_disp["pIC50"].map("{:.3f}".format)
                df_disp["IC50 [nM]"] = df_disp["IC50 [nM]"].map("{:,.1f}".format)
                st.dataframe(df_disp, use_container_width=True, hide_index=True)
                if len(rows) == 2:
                    st.info(f"Średnia MLP + GNN:  pIC50 = **{avg_pic:.3f}**  |  IC50 = **{pic50_to_ic50(avg_pic):,.1f} nM**")

            with col_met:
                badge_text, badge_color = activity_badge(avg_pic)
                st.markdown(
                    f"<div style='background:{badge_color}22; border-left:4px solid {badge_color}; "
                    f"padding:14px 16px; border-radius:6px; margin-bottom:12px'>"
                    f"<b>Przewidywana aktywność</b><br>{badge_text}</div>",
                    unsafe_allow_html=True,
                )
                st.metric("pIC50", f"{avg_pic:.3f}")
                st.metric("IC50",  f"{pic50_to_ic50(avg_pic):,.1f} nM")

        st.markdown("---")
        st.subheader("Właściwości fizykochemiczne (Lipinski Ro5)")

        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hdon = Descriptors.NumHDonors(mol)
        hacc = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        nrng = Descriptors.RingCount(mol)
        fsp3 = Descriptors.FractionCSP3(mol)

        props = [
            ("MW",          f"{mw:.1f} Da",   mw   <= 500),
            ("LogP",        f"{logp:.2f}",    logp <= 5),
            ("H-donors",    str(hdon),        hdon <= 5),
            ("H-acceptors", str(hacc),        hacc <= 10),
            ("TPSA",        f"{tpsa:.1f} A2", tpsa <= 140),
            ("Rot. bonds",  str(rotb),        rotb <= 10),
            ("Rings",       str(nrng),        True),
            ("Fsp3",        f"{fsp3:.2f}",    fsp3 >= 0.25),
        ]

        prop_cols = st.columns(4)
        for i, (name, val, ok) in enumerate(props):
            with prop_cols[i % 4]:
                st.metric(f"{'OK' if ok else '!'} {name}", val)

        n_viol = sum(1 for _, _, ok in props[:4] if not ok)
        if n_viol == 0:
            st.success("Spełnia reguły Lipińskiego (Ro5).")
        else:
            st.warning(f"{n_viol} naruszenie{'a' if n_viol > 1 else ''} reguł Lipińskiego.")

        with st.expander("Szczegóły molekuły"):
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            st.markdown(f"""
| łaściwość | Wartość |
|---|---|
| SMILES | `{smiles_val}` |
| Wzór sumaryczny | `{formula}` |
| Dokładna masa | {Descriptors.ExactMolWt(mol):.4f} Da |
| Ciężkie atomy | {Descriptors.HeavyAtomCount(mol)} |
| Atomy ogółem | {mol.GetNumAtoms()} |
| Wiązania | {mol.GetNumBonds()} |
""")

st.markdown("---")
st.caption("Wykonano przez Łukasz Małucha na podstawie danych ChEMBL w ramach projektu z przedmiotu 'Warsztaty Sztucznej Inteligencji'.")
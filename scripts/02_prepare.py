"""Day 1: raw ChEMBL IC50 -> standardised, de-duplicated, labelled compounds + features.

Outputs (data/processed/):
  herg_compounds.csv   one row per InChIKey: smiles, median pIC50, n_meas, label, scaffold
  features.npz         Morgan r=2/2048 bits + MW, logP, TPSA for the labelled compounds
  summary.json         every filtering step with counts, and the label-threshold rationale
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "chembl240_ic50.csv.gz"
OUT = ROOT / "data" / "processed"

THRESHOLD = 5.0  # pIC50 5 = 10 uM, the usual hERG safety cut-off
BAND = 0.3       # compounds within +/-0.3 log units of the threshold are left unlabelled

_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()
_morgan = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
FEATURE_NAMES = [f"bit_{i}" for i in range(2048)] + ["MW", "logP", "TPSA"]


def standardise(smiles):
    """Cleanup -> keep largest fragment (salt strip) -> neutralise. None if RDKit fails."""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = rdMolStandardize.Cleanup(m)
        m = _lfc.choose(m)
        m = _unc.uncharge(m)
        m = rdMolStandardize.Cleanup(m)
        return Chem.MolToSmiles(m), Chem.MolToInchiKey(m)
    except Exception:
        return None


def featurise(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fp = np.array([_morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
    desc = np.array([[Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m)] for m in mols])
    return np.hstack([fp, desc]).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"source": json.loads((RAW.parent / "download_meta.json").read_text())}
    d = pd.read_csv(RAW, low_memory=False)
    log["n_raw"] = len(d)

    steps = [
        ("exact '=' relation", d.standard_relation == "="),
        ("units nM, value > 0", (d.standard_units == "nM") & (d.standard_value > 0)),
        ("has SMILES", d.canonical_smiles.notna()),
        ("wild-type channel (no assay_variant_mutation)", d.assay_variant_mutation.isna()),
        ("no ChEMBL validity flag", d.data_validity_comment.isna()),
    ]
    log["filters"] = []
    for name, mask in steps:
        before = len(d)
        d = d[mask.loc[d.index]]
        log["filters"].append({"step": name, "kept": len(d), "dropped": before - len(d)})

    d = d.assign(pIC50=9 - np.log10(d.standard_value))
    before = len(d)
    d = d[(d.pIC50 > 2) & (d.pIC50 < 12)]
    log["filters"].append({"step": "2 < pIC50 < 12", "kept": len(d), "dropped": before - len(d)})

    std = d.canonical_smiles.map(standardise)
    log["n_standardisation_failed"] = int(std.isna().sum())
    d = d[std.notna()].assign(smiles=std.dropna().str[0], inchikey=std.dropna().str[1])
    log["n_measurements"] = len(d)

    rep = d.groupby("inchikey").pIC50.agg(["std", "count"])
    rep = rep[rep["count"] > 1]
    log["replicates"] = {"n_compounds": len(rep),
                         "median_sd": round(float(rep["std"].median()), 3),
                         "p75_sd": round(float(rep["std"].quantile(0.75)), 3),
                         "p90_sd": round(float(rep["std"].quantile(0.90)), 3)}

    c = (d.groupby("inchikey")
           .agg(smiles=("smiles", "first"), pIC50=("pIC50", "median"), n_meas=("pIC50", "size"),
                molecule_chembl_id=("molecule_chembl_id", "first"))
           .reset_index())
    log["n_unique_compounds"] = len(c)
    log["n_at_exactly_10uM"] = int(np.isclose(c.pIC50, THRESHOLD).sum())

    ambiguous = (c.pIC50 - THRESHOLD).abs() < BAND
    c["label"] = np.where(ambiguous, -1, (c.pIC50 >= THRESHOLD).astype(int))
    c["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in c.smiles]
    labelled = c[c.label >= 0].reset_index(drop=True)
    log["label"] = {"threshold_pIC50": THRESHOLD, "band": BAND,
                    "n_ambiguous": int(ambiguous.sum()),
                    "n_labelled": len(labelled),
                    "n_blocker": int(labelled.label.sum()),
                    "n_non_blocker": int((labelled.label == 0).sum()),
                    "n_scaffolds": int(labelled.scaffold.nunique()),
                    "rationale": ("pIC50 5 (10 uM) is the common early-discovery hERG flag. "
                                  "The +/-0.3 band exceeds the p75 replicate SD, so a compound "
                                  "inside it could land on either side on re-measurement.")}

    c.to_csv(OUT / "herg_compounds.csv", index=False)
    X = featurise(labelled.smiles)
    np.savez_compressed(OUT / "features.npz", X=X, y=labelled.label.values,
                        inchikey=labelled.inchikey.values, scaffold=labelled.scaffold.values,
                        feature_names=np.array(FEATURE_NAMES))
    log["features"] = {"shape": list(X.shape), "morgan": "radius 2, 2048 bits", "descriptors": ["MW", "logP", "TPSA"]}
    (OUT / "summary.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

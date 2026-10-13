# hERG cardiotoxicity classifier

**Question.** Can hERG channel blockade be predicted from structure early in discovery?

## Data
ChEMBL hERG (KCNH2) IC50 bioactivity data.

## Approach
1. IC50 (nM, relation '=') → pIC50, median of replicates
2. Blocker threshold pIC50 ≥ 5 (10 µM) with an excluded uncertainty band
3. RDKit standardisation: salt stripping, MolStandardize, InChIKey de-duplication
4. Morgan fingerprints (r=2, 2048 bits) + MW, logP, TPSA
5. Bemis–Murcko scaffold split vs random split; RandomForest and LightGBM, 5-fold CV
6. ROC-AUC, PR-AUC, MCC, calibration curve, applicability domain (nearest-neighbour Tanimoto), SHAP

## Deliverables
Clean dataset, model, ROC/PR and SHAP figures, write-up.

## Limitations
- ChEMBL assays are heterogeneous (cell lines, protocols); label noise caps performance.
- Random splits overstate performance; the scaffold split is the headline number.

## Reproduce
```bash
python scripts/01_fetch_chembl.py   # all ChEMBL IC50 records for CHEMBL240, checkpointed and resumable
python scripts/02_prepare.py        # standardise, de-duplicate, label, featurise
```
Every filtering step and its counts are in `data/processed/summary.json`.

## Status
Day 1 done (ChEMBL 37, 19,130 IC50 records -> 9,586 unique compounds -> 6,378 labelled: 4,646 blockers, 1,732 non-blockers, 3,470 scaffolds). Modelling next.
Part of a 30-project computational biology and scientific software portfolio.

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

## Results
Held-out test set, 1,275 compounds (75% blockers). The scaffold split shares no Bemis–Murcko scaffold between train and test; it is the honest number.

| Split | Model | CV ROC-AUC (5-fold) | Test ROC-AUC | Test PR-AUC | Test MCC | Brier |
|---|---|---|---|---|---|---|
| Scaffold | RandomForest | 0.869 ± 0.022 | **0.889** | 0.955 | 0.539 | 0.117 |
| Scaffold | LightGBM | 0.857 ± 0.021 | 0.883 | 0.955 | 0.541 | 0.115 |
| Random | RandomForest | 0.908 ± 0.011 | 0.923 | 0.969 | 0.641 | 0.098 |
| Random | LightGBM | 0.896 ± 0.012 | 0.917 | 0.965 | 0.643 | 0.096 |

- **Random splits overstate performance** by ~0.03–0.04 ROC-AUC and ~0.10 MCC: 468 test scaffolds also appear in training.
- **Applicability domain matters.** Scaffold-test ROC-AUC (LightGBM) falls from 0.92 for compounds with a close training neighbour (Tanimoto ≥ 0.6, n = 856) to 0.78 below 0.4 (n = 182). Treat predictions for novel chemotypes as weak evidence.
- **Label band.** Dropping compounds within ±0.3 log units of pIC50 5 removes a third of the data (3,208 compounds). Without the band the scaffold-split ROC-AUC is 0.800 on 9,586 compounds; with ±0.5 it is 0.907 on 4,468. Part of the gain is easier, more separated labels, not a better model (`results/band_sensitivity.csv`).
- **What drives predictions.** logP and TPSA dominate the SHAP ranking: lipophilic, low-polarity compounds are flagged as blockers, consistent with known hERG structure–activity trends (`figures/fig2_shap_summary.png`).
- **Calibration.** LightGBM probabilities track observed blocker rates on the scaffold test; RandomForest is under-confident in the 0.5–0.8 range (`figures/fig1_roc_pr_calibration.png`).

`models/lightgbm_scaffold.joblib` is the model used downstream as a hERG safety flag; featurise inputs with `scripts/02_prepare.py:featurise`.

## Reproduce
```bash
python scripts/01_fetch_chembl.py   # all ChEMBL IC50 records for CHEMBL240, checkpointed and resumable
python scripts/02_prepare.py        # standardise, de-duplicate, label, featurise
python scripts/03_model.py          # splits, RandomForest + LightGBM, metrics, figures, SHAP
```
`notebooks/herg_results.ipynb` walks through the outputs.
Every filtering step and its counts are in `data/processed/summary.json`.

## Status
Complete. Part of a 30-project computational biology and scientific software portfolio.

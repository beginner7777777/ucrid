# Temperature Calibration Comparison

## Setup

- Pipeline: `run_ucrid.py` Stage1+2 (no `--use_llm`)
- Checkpoints:
  - CLINC150: `outputs/stage1/ucrid_stage1_clinc150_rerun_20260319_095845/best_model.pt`
  - Banking77: `outputs/stage1/ucrid_stage1_banking77_rerun_20260320_1220/best_model.pt`
- Temp ON: default config (`routing.calibrate_temperature=true`)
- Temp OFF: `*_no_temp_calib.yaml` + `--disable_temperature`

## Stage1+2 Metrics

| Dataset | Setting | Acc | ID Acc | OOS Prec | OOS Rec | OOS F1 | LLM Call Rate | Router Temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLINC150 | Temp ON | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.05% | 0.8129 |
| CLINC150 | Temp OFF | 90.87% | 95.76% | 92.36% | 68.90% | 0.7892 | 9.05% | 1.0000 |
| Banking77 | Temp ON | 89.94% | 92.47% | 95.38% | 74.40% | 0.8360 | 7.21% | 0.8755 |
| Banking77 | Temp OFF | 90.89% | 92.27% | 91.76% | 82.40% | 0.8683 | 8.52% | 1.0000 |

## Delta (Temp OFF - Temp ON)

| Dataset | Δ OOS F1 | Δ LLM Call Rate | Δ Acc | Δ ID Acc |
|---|---:|---:|---:|---:|
| CLINC150 | +0.0521 | +1.00% | +1.33% | -0.20% |
| Banking77 | +0.0323 | +1.31% | +0.95% | -0.19% |

## Observations

- CLINC150: Temp OFF has higher OOS F1 (0.7371 vs 0.7892).
- Banking77: Temp OFF has higher OOS F1 (0.8360 vs 0.8683).

## Result Files

- CLINC150 Temp ON: `outputs/ucrid/clinc150_calibration_cmp_temp_on_20260331/ucrid_results.json`
- CLINC150 Temp OFF: `outputs/ucrid/clinc150_calibration_cmp_temp_off_20260331/ucrid_results.json`
- Banking77 Temp ON: `outputs/ucrid/banking77_calibration_cmp_temp_on_20260331/ucrid_results.json`
- Banking77 Temp OFF: `outputs/ucrid/banking77_calibration_cmp_temp_off_20260331/ucrid_results.json`

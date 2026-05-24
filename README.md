

## Run the full experiment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/vkr_pipeline.py \
  --start 2019-01-01 \
  --end 2025-12-31 \
  --max-articles-per-month 20 \
  --language-filter English \
  --model qwen-plus \
  --auditor-model qwen3-max
```





## Outputs

- `outputs/market_data.csv` - FRED Brent, WTI and Henry Hub time series.
- `outputs/news_annotations.csv` - filtered GDELT headlines and Qwen semantic annotations.
- `outputs/annotation_audit.csv` - Qwen auditor sample check.
- `outputs/daily_model_dataset.csv` - joined market and news-factor dataset.
- `outputs/predictions.csv` - rolling out-of-sample predictions by model.
- `outputs/rolling_metrics.csv` - monthly rolling metrics by model.
- `outputs/metrics.csv` - benchmark, hybrid, and volatility metrics.
- `outputs/feature_ablation.csv` - contribution analysis of news-factor groups.
- `outputs/stat_tests.csv` - McNemar and Diebold-Mariano checks.
- `outputs/language_robustness.csv` - GDELT language distribution before filtering.
- `outputs/run_summary.json` - reproducibility metadata and headline run summary.

## Tests

```bash
.venv/bin/python -m pytest tests
```

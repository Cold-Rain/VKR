from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vkr_pipeline as vp


class PipelineTests(unittest.TestCase):
    def test_month_windows_cover_requested_range(self) -> None:
        windows = vp.month_windows("2024-01-15", "2024-03-03")
        self.assertEqual(windows[0][0].isoformat(), "2024-01-15")
        self.assertEqual(windows[-1][1].isoformat(), "2024-03-03")
        self.assertEqual(len(windows), 3)

    def test_normalize_filters_period_and_language(self) -> None:
        raw = pd.DataFrame(
            [
                {"title": "A", "domain": "x", "language": "English", "seendate": "20240102T010000Z"},
                {"title": "B", "domain": "x", "language": "Arabic", "seendate": "20240102T010000Z"},
                {"title": "C", "domain": "x", "language": "English", "seendate": "20250101T010000Z"},
            ]
        )
        out = vp.normalize_gdelt_dates(raw, start="2024-01-01", end="2024-12-31", language_filter="English")
        self.assertEqual(out["title"].tolist(), ["A"])

    def test_news_cache_key_depends_on_language_and_limit(self) -> None:
        base = {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "max_articles_per_month": 8,
            "language_filter": "English",
            "model": "qwen-plus",
            "auditor_model": "qwen3-max",
            "audit_sample_size": 1,
            "base_url": "https://example.test/v1",
            "api_key_file": Path("api.txt"),
            "cache_dir": Path("cache"),
            "output_dir": Path("outputs"),
            "refresh_news": False,
            "refresh_llm": False,
            "no_llm": True,
            "min_train_days": 20,
        }
        a = vp.news_cache_path(vp.RunConfig(**base)).name
        base["language_filter"] = "Arabic"
        b = vp.news_cache_path(vp.RunConfig(**base)).name
        self.assertNotEqual(a, b)

    def test_trading_day_mapping_does_not_use_future_target(self) -> None:
        market_index = pd.bdate_range("2024-01-01", periods=5)
        news = pd.DataFrame(
            [
                {
                    "news_id": "n1",
                    "title": "Oil rises",
                    "domain": "oilprice.com",
                    "language": "English",
                    "seendate": "20240102T100000Z",
                    "date": pd.Timestamp("2024-01-02"),
                }
            ]
        )
        ann = pd.DataFrame([vp.rule_based_annotation(pd.Series(news.iloc[0]))])
        features = vp.aggregate_news_features(news, ann, market_index)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-02"), "news_count"], 1)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-01"), "news_count"], 0)

    def test_metrics_handle_single_prediction_set(self) -> None:
        preds = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "model": ["m"] * 4,
                "target_return": [0.1, -0.1, 0.2, -0.2],
                "target_direction": [1, 0, 1, 0],
                "target_volatility": [0.01, 0.01, 0.04, 0.04],
                "pred_return": [0.1, -0.05, -0.1, -0.2],
                "pred_direction": [1, 0, 0, 0],
                "pred_score": [0.9, 0.2, 0.4, 0.1],
                "pred_volatility": [np.nan] * 4,
            }
        )
        metrics = vp.summarize_predictions(preds)
        self.assertAlmostEqual(float(metrics.loc[0, "direction_accuracy"]), 0.75)
        self.assertIn("rmse_return", metrics.columns)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from openai import OpenAI
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX


LOGGER = logging.getLogger("vkr_pipeline")
PROMPT_SCHEMA_VERSION = "2026-05-08-v2"

FRED_SERIES = {
    "brent": "DCOILBRENTEU",
    "wti": "DCOILWTICO",
    "gas": "DHHNGSP",
}

GDELT_QUERY = '("Brent crude" OR "oil prices" OR OPEC OR "natural gas" OR "energy market")'

TOPICS = [
    "production",
    "demand",
    "geopolitics",
    "opec",
    "sanctions",
    "company",
    "macro",
    "weather",
    "other",
]

EVENT_TYPES = [
    "supply_shock",
    "demand_signal",
    "geopolitical_risk",
    "policy_decision",
    "inventory_price",
    "company_news",
    "macro_signal",
    "weather_event",
    "other",
]

SOURCE_CLASSES = [
    "official",
    "energy_media",
    "business_media",
    "regional_media",
    "general_media",
    "other",
]

TOPIC_DEFINITIONS = {
    "production": "oil/gas production, supply, drilling, output, inventories, exports",
    "demand": "consumption, demand forecasts, economic activity that changes energy demand",
    "geopolitics": "war, attacks, shipping disruption, regional conflict, political risk",
    "opec": "OPEC, OPEC+, quotas, coordinated cuts or production-policy meetings",
    "sanctions": "sanctions, embargoes, price caps, trade restrictions",
    "company": "energy company earnings, stocks, corporate investment, mergers",
    "macro": "interest rates, USD, inflation, GDP, broad financial market conditions",
    "weather": "weather, temperature, storms, heating/cooling demand",
    "other": "only if none of the above fits",
}

EVENT_DEFINITIONS = {
    "supply_shock": "production, inventory, export, shipping, disruption, or spare-capacity news",
    "demand_signal": "consumption, economic activity, travel, industrial demand, or demand forecast news",
    "geopolitical_risk": "war, attack, regional conflict, sanctions, embargo, or shipping-risk news",
    "policy_decision": "OPEC/OPEC+, government, regulator, central bank, or agency policy news",
    "inventory_price": "price, inventory, forecast, analyst target, or market-positioning news",
    "company_news": "earnings, investment, merger, production plan, or corporate news",
    "macro_signal": "rates, inflation, currency, GDP, recession, or broad financial conditions",
    "weather_event": "storm, cold/heat, hurricane, or weather-driven energy-demand news",
    "other": "only if none of the above fits",
}


@dataclass
class RunConfig:
    start: str
    end: str
    max_articles_per_month: int
    language_filter: str
    model: str
    auditor_model: str
    audit_sample_size: int
    base_url: str
    api_key_file: Path
    cache_dir: Path
    output_dir: Path
    refresh_news: bool
    refresh_llm: bool
    no_llm: bool
    min_train_days: int


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Rolling LLM news-factor experiment for oil and gas market forecasting."
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-articles-per-month", type=int, default=20)
    parser.add_argument("--language-filter", default="English")
    parser.add_argument("--model", default=os.getenv("DASHSCOPE_MODEL", "qwen-plus"))
    parser.add_argument("--auditor-model", default=os.getenv("DASHSCOPE_AUDITOR_MODEL", "qwen3-max"))
    parser.add_argument("--audit-sample-size", type=int, default=30)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument("--api-key-file", default="api.txt")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--refresh-news", action="store_true")
    parser.add_argument("--refresh-llm", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    return RunConfig(
        start=args.start,
        end=args.end,
        max_articles_per_month=args.max_articles_per_month,
        language_filter=args.language_filter,
        model=args.model,
        auditor_model=args.auditor_model,
        audit_sample_size=args.audit_sample_size,
        base_url=args.base_url,
        api_key_file=Path(args.api_key_file),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        refresh_news=args.refresh_news,
        refresh_llm=args.refresh_llm,
        no_llm=args.no_llm,
        min_train_days=args.min_train_days,
    )


def read_api_key(path: Path) -> str | None:
    env_value = os.getenv("DASHSCOPE_API_KEY")
    if env_value:
        return env_value.strip()
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    return None


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fetch_url_json(url: str, *, attempts: int = 2, sleep_seconds: float = 1.5) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 academic-research-prototype/2.0",
        "Accept": "application/json,text/plain,*/*",
    }
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            LOGGER.warning("HTTP %s for %s; retry %s/%s", exc.code, url[:120], attempt, attempts)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            LOGGER.warning("Network error for %s; retry %s/%s: %s", url[:120], attempt, attempts, exc)
        time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Could not fetch JSON after retries: {last_error}")


def month_windows(start: str, end: str) -> list[tuple[date, date]]:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    current = start_date.replace(day=1)
    windows: list[tuple[date, date]] = []
    while current <= end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        window_start = max(current, start_date)
        window_end = min(next_month - timedelta(days=1), end_date)
        windows.append((window_start, window_end))
        current = next_month
    return windows


def gdelt_windows(start: str, end: str, *, months_per_window: int = 3) -> list[tuple[date, date, int]]:
    monthly = month_windows(start, end)
    windows: list[tuple[date, date, int]] = []
    for idx in range(0, len(monthly), months_per_window):
        chunk = monthly[idx : idx + months_per_window]
        windows.append((chunk[0][0], chunk[-1][1], len(chunk)))
    return windows


def fetch_fred_series(start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, series_id in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", series_id: name})
        df[name] = pd.to_numeric(df[name].replace(".", np.nan), errors="coerce")
        frames.append(df[["date", name]])
    market = frames[0]
    for frame in frames[1:]:
        market = market.merge(frame, on="date", how="outer")
    market = market.sort_values("date")
    market = market[(market["date"] >= start) & (market["date"] <= end)]
    market = market.set_index("date").asfreq("B")
    market = market.ffill().dropna()
    for col in FRED_SERIES:
        market[f"{col}_ret"] = np.log(market[col]).diff().fillna(0.0)
    market["brent_wti_spread"] = market["brent"] - market["wti"]
    for window in [5, 10, 20]:
        market[f"brent_vol_{window}"] = market["brent_ret"].rolling(window, min_periods=2).std().fillna(0.0)
    for lag in range(1, 11):
        market[f"brent_ret_lag{lag}"] = market["brent_ret"].shift(lag).fillna(0.0)
        market[f"wti_ret_lag{lag}"] = market["wti_ret"].shift(lag).fillna(0.0)
        market[f"gas_ret_lag{lag}"] = market["gas_ret"].shift(lag).fillna(0.0)
    return market


def news_cache_path(config: RunConfig) -> Path:
    key = stable_hash(
        {
            "start": config.start,
            "end": config.end,
            "max_articles_per_month": config.max_articles_per_month,
            "language_filter": config.language_filter,
            "query": GDELT_QUERY,
        }
    )
    return config.cache_dir / f"gdelt_news_{key}.json"


def normalize_gdelt_dates(
    news: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    language_filter: str = "",
) -> pd.DataFrame:
    if news.empty:
        return news.copy()
    result = news.copy()
    result["seendate"] = result["seendate"].astype(str)
    parsed = pd.to_datetime(
        result["seendate"].str.replace("T", "", regex=False),
        format="%Y%m%d%H%M%SZ",
        errors="coerce",
    )
    result["date"] = parsed.dt.normalize()
    result = result.dropna(subset=["date"])
    if start:
        result = result[result["date"] >= pd.Timestamp(start)]
    if end:
        result = result[result["date"] <= pd.Timestamp(end)]
    if language_filter:
        result = result[result["language"].astype(str).str.casefold() == language_filter.casefold()]
    return result.sort_values(["date", "title"]).reset_index(drop=True)


def fetch_gdelt_news(config: RunConfig) -> pd.DataFrame:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = news_cache_path(config)
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and not config.refresh_news:
        records = json.loads(cache_path.read_text(encoding="utf-8"))
        news = pd.DataFrame(records)
        if "date" in news.columns:
            news["date"] = pd.to_datetime(news["date"])
        if meta_path.exists():
            news.attrs["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        return news

    articles: list[dict[str, Any]] = []
    for window_start, window_end in month_windows(config.start, config.end):
        params = {
            "query": GDELT_QUERY,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(config.max_articles_per_month),
            "startdatetime": window_start.strftime("%Y%m%d000000"),
            "enddatetime": window_end.strftime("%Y%m%d235959"),
            "sort": "hybridrel",
        }
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
        try:
            data = fetch_url_json(url)
        except Exception as exc:
            LOGGER.warning("GDELT window %s..%s skipped: %s", window_start, window_end, exc)
            continue
        for item in data.get("articles", []):
            title = (item.get("title") or "").strip()
            seen = item.get("seendate") or ""
            if not title or not seen:
                continue
            articles.append(
                {
                    "title": title,
                    "url": item.get("url", ""),
                    "domain": item.get("domain", ""),
                    "source_country": item.get("sourcecountry", ""),
                    "language": item.get("language", ""),
                    "seendate": seen,
                }
            )
        time.sleep(0.45)

    raw = pd.DataFrame(articles)
    if raw.empty:
        raw = fallback_news(config.start, config.end, config.max_articles_per_month)
    raw = raw.drop_duplicates(subset=["title", "domain", "seendate"]).reset_index(drop=True)
    language_summary = raw["language"].value_counts(dropna=False).rename_axis("language").reset_index(name="records")
    news = normalize_gdelt_dates(
        raw,
        start=config.start,
        end=config.end,
        language_filter=config.language_filter,
    )
    if news.empty:
        LOGGER.warning("Language/date filtering removed all GDELT rows; using fallback news.")
        news = normalize_gdelt_dates(
            fallback_news(config.start, config.end, config.max_articles_per_month),
            start=config.start,
            end=config.end,
        )
    news = news.drop_duplicates(subset=["title", "domain"]).reset_index(drop=True)
    news.insert(0, "news_id", [f"n{i:05d}" for i in range(len(news))])
    meta = {
        "raw_records": int(len(raw)),
        "filtered_records": int(len(news)),
        "language_filter": config.language_filter,
        "language_summary": language_summary.to_dict(orient="records"),
    }
    cache_path.write_text(json.dumps(news.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    news.attrs["meta"] = meta
    return news


def fallback_news(start: str = "2019-01-01", end: str = "2025-12-31", per_month: int = 4) -> pd.DataFrame:
    templates = [
        "Oil prices rise after OPEC signals continued production cuts",
        "Brent crude falls as demand concerns pressure energy markets",
        "Natural gas prices jump after colder weather forecast",
        "Geopolitical risk supports crude futures during volatile session",
        "OPEC keeps production policy unchanged as supply risks remain",
        "Oil settles lower after inventory data points to weaker consumption",
        "Energy shares mixed as investors weigh crude supply outlook",
        "Analysts raise Brent forecast after tighter export signals",
    ]
    rows = []
    limit = max(1, min(per_month, len(templates)))
    for window_start, _, _ in gdelt_windows(start, end, months_per_window=1):
        for i, title in enumerate(templates[:limit]):
            day = min(3 + i * 3, 24)
            seen = window_start.replace(day=min(day, 28)).strftime("%Y%m%d090000Z")
            rows.append((f"{title} ({window_start:%Y-%m})", seen, "fallback.local"))
    return pd.DataFrame(
        [
            {
                "title": title,
                "url": "",
                "domain": domain,
                "source_country": "",
                "language": "English",
                "seendate": seen,
            }
            for title, seen, domain in rows
        ]
    )


def extract_json_value(text: str) -> dict[str, Any] | list[Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return float(max(low, min(high, number)))


def source_domain_class(domain: str) -> str:
    value = domain.lower()
    if any(token in value for token in ["eia.gov", "iea.org", "opec.org", "fred.stlouisfed.org"]):
        return "official"
    if any(token in value for token in ["oilprice", "rigzone", "energyvoice", "worldoil"]):
        return "energy_media"
    if any(token in value for token in ["reuters", "bloomberg", "marketwatch", "cnbc", "ft.com", "wsj"]):
        return "business_media"
    if any(token in value for token in ["arab", "saudi", "india", "turkey", "africa"]):
        return "regional_media"
    if value:
        return "general_media"
    return "other"


def infer_event_type(title: str) -> str:
    value = title.lower()
    if any(token in value for token in ["opec", "quota", "policy", "rate", "fed"]):
        return "policy_decision"
    if any(token in value for token in ["war", "attack", "sanction", "geopolitical", "iran", "israel", "red sea"]):
        return "geopolitical_risk"
    if any(token in value for token in ["inventory", "stockpile", "price", "forecast", "brent", "wti"]):
        return "inventory_price"
    if any(token in value for token in ["supply", "production", "output", "export", "drilling", "cut"]):
        return "supply_shock"
    if any(token in value for token in ["demand", "consumption", "china", "travel"]):
        return "demand_signal"
    if any(token in value for token in ["storm", "weather", "hurricane", "cold", "heat"]):
        return "weather_event"
    if any(token in value for token in ["earnings", "company", "merger", "stock"]):
        return "company_news"
    if any(token in value for token in ["inflation", "dollar", "recession", "gdp"]):
        return "macro_signal"
    return "other"


def rule_based_annotation(row: pd.Series) -> dict[str, Any]:
    title = str(row.get("title", "")).lower()
    positive = ["rise", "rises", "higher", "jump", "support", "cut", "cuts", "tight", "risk", "sanction"]
    negative = ["fall", "falls", "lower", "drop", "weak", "demand concerns", "inventory", "surplus"]
    score = 0.0
    score += sum(token in title for token in positive) * 0.22
    score -= sum(token in title for token in negative) * 0.22
    if "opec" in title:
        topic = "opec"
    elif "sanction" in title or "geopolitical" in title or "war" in title:
        topic = "geopolitics"
    elif "demand" in title:
        topic = "demand"
    elif "production" in title or "supply" in title or "cut" in title:
        topic = "production"
    elif "gas" in title or "weather" in title:
        topic = "weather"
    elif "stock" in title or "company" in title:
        topic = "company"
    elif "rate" in title or "dollar" in title or "inflation" in title:
        topic = "macro"
    else:
        topic = "other"
    return {
        "news_id": row["news_id"],
        "relevance": 0.65,
        "sentiment": clamp_float(score, -1.0, 1.0, 0.0),
        "topic": topic,
        "event_type": infer_event_type(str(row.get("title", ""))),
        "uncertainty": 0.45,
        "horizon_days": 3,
        "source_weight": 0.65,
        "source_domain_class": source_domain_class(str(row.get("domain", ""))),
        "impact_target": "oil",
        "method": "rule_fallback",
    }


def news_frame_hash(news: pd.DataFrame) -> str:
    cols = ["news_id", "date", "domain", "language", "title"]
    payload = news[cols].assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records")
    return stable_hash(payload)


class SemanticAgent:
    def __init__(self, config: RunConfig):
        self.config = config
        self.api_key = None if config.no_llm else read_api_key(config.api_key_file)
        self.client: OpenAI | None = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=config.base_url)

    def cache_path(self, news: pd.DataFrame) -> Path:
        key = stable_hash(
            {
                "start": self.config.start,
                "end": self.config.end,
                "language_filter": self.config.language_filter,
                "model": self.config.model,
                "prompt_schema": PROMPT_SCHEMA_VERSION,
                "news_hash": news_frame_hash(news),
            }
        )
        return self.config.cache_dir / f"llm_annotations_{key}.json"

    def annotate(self, news: pd.DataFrame) -> pd.DataFrame:
        cache_path = self.cache_path(news)
        if cache_path.exists() and not self.config.refresh_llm:
            return pd.DataFrame(json.loads(cache_path.read_text(encoding="utf-8")))

        partial_path = cache_path.with_name(cache_path.stem + ".partial.json")
        results_by_id: dict[str, dict[str, Any]] = {}
        if partial_path.exists() and not self.config.refresh_llm:
            for item in json.loads(partial_path.read_text(encoding="utf-8")):
                cleaned = self._clean_annotation(item)
                if cleaned["news_id"]:
                    results_by_id[cleaned["news_id"]] = cleaned

        if self.client is None:
            LOGGER.warning("No Model Studio key/client found; using rule-based annotation.")
            for _, row in news.iterrows():
                results_by_id[str(row["news_id"])] = rule_based_annotation(row)
        else:
            batch_size = 12
            for start in range(0, len(news), batch_size):
                batch = news.iloc[start : start + batch_size]
                pending = batch[~batch["news_id"].astype(str).isin(results_by_id)]
                if pending.empty:
                    continue
                try:
                    batch_results = [self._clean_annotation(item) for item in self._annotate_batch(pending)]
                    batch_by_id = {
                        item["news_id"]: item
                        for item in batch_results
                        if item["news_id"] in set(pending["news_id"].astype(str))
                    }
                    for _, row in pending.iterrows():
                        news_id = str(row["news_id"])
                        results_by_id[news_id] = batch_by_id.get(news_id, rule_based_annotation(row))
                except Exception as exc:
                    LOGGER.warning("LLM annotation failed for rows %s..%s: %s", start, start + len(batch), exc)
                    for _, row in pending.iterrows():
                        results_by_id[str(row["news_id"])] = rule_based_annotation(row)
                partial_path.write_text(
                    json.dumps(list(results_by_id.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                time.sleep(0.25)

        annotations = pd.DataFrame(
            [results_by_id[str(news_id)] for news_id in news["news_id"].astype(str) if str(news_id) in results_by_id]
        )
        cache_path.write_text(
            json.dumps(annotations.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if partial_path.exists():
            partial_path.unlink()
        return annotations

    def _annotate_batch(self, batch: pd.DataFrame) -> list[dict[str, Any]]:
        assert self.client is not None
        items = [
            {
                "news_id": row["news_id"],
                "date": str(pd.Timestamp(row["date"]).date()),
                "domain": row.get("domain", ""),
                "language": row.get("language", ""),
                "title": row["title"],
            }
            for _, row in batch.iterrows()
        ]
        system = (
            "You are a financial NLP annotator for an academic experiment. "
            "Return only valid JSON. Do not include markdown."
        )
        user = {
            "prompt_schema": PROMPT_SCHEMA_VERSION,
            "task": (
                "For each oil, gas, OPEC, energy market, company, macro, or geopolitics headline, "
                "estimate how it can affect Brent/WTI oil returns in the next few trading days."
            ),
            "topic_definitions": TOPIC_DEFINITIONS,
            "event_type_definitions": EVENT_DEFINITIONS,
            "source_domain_classes": SOURCE_CLASSES,
            "output_contract": (
                "Return one JSON object with key items. For every headline, choose exactly one topic, "
                "one event_type, and one source_domain_class from the allowed lists."
            ),
            "item_schema": {
                "news_id": "string",
                "relevance": "float 0..1",
                "sentiment": "float -1..1, positive means bullish for oil/energy prices",
                "topic": "one of: " + ", ".join(TOPICS),
                "event_type": "one of: " + ", ".join(EVENT_TYPES),
                "uncertainty": "float 0..1",
                "horizon_days": "integer 1..10",
                "source_weight": "float 0..1",
                "source_domain_class": "one of: " + ", ".join(SOURCE_CLASSES),
                "impact_target": "oil|gas|sector|macro|other",
            },
            "headlines": items,
        }
        request = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        try:
            completion = self.client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "response_format" not in str(exc).lower():
                raise
            completion = self.client.chat.completions.create(**request)
        content = completion.choices[0].message.content or "{}"
        parsed = extract_json_value(content)
        values = self._items_from_response(parsed)
        if not isinstance(values, list):
            raise ValueError("LLM response does not contain an items list")
        return values

    @staticmethod
    def _items_from_response(parsed: dict[str, Any] | list[Any]) -> list[Any] | None:
        if isinstance(parsed, list):
            return parsed
        for key in ["items", "annotations", "results", "records", "headlines"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        for value in parsed.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                if any("news_id" in item for item in value):
                    return value
        dict_values = [value for value in parsed.values() if isinstance(value, dict)]
        if dict_values and all("news_id" in value for value in dict_values):
            return dict_values
        keyed_items = []
        for key, value in parsed.items():
            if isinstance(value, dict) and key:
                keyed_items.append({"news_id": str(key), **value})
        if keyed_items and len(keyed_items) == len(parsed):
            return keyed_items
        if "news_id" in parsed:
            return [parsed]
        return None

    @staticmethod
    def _clean_annotation(item: dict[str, Any]) -> dict[str, Any]:
        topic = str(item.get("topic", "other")).lower()
        event_type = str(item.get("event_type", "other")).lower()
        source_class = str(item.get("source_domain_class", "other")).lower()
        if topic not in TOPICS:
            topic = "other"
        if event_type not in EVENT_TYPES:
            event_type = "other"
        if source_class not in SOURCE_CLASSES:
            source_class = "other"
        return {
            "news_id": str(item.get("news_id", "")),
            "relevance": clamp_float(item.get("relevance"), 0.0, 1.0, 0.5),
            "sentiment": clamp_float(item.get("sentiment"), -1.0, 1.0, 0.0),
            "topic": topic,
            "event_type": event_type,
            "uncertainty": clamp_float(item.get("uncertainty"), 0.0, 1.0, 0.5),
            "horizon_days": int(round(clamp_float(item.get("horizon_days"), 1, 10, 3))),
            "source_weight": clamp_float(item.get("source_weight"), 0.0, 1.0, 0.7),
            "source_domain_class": source_class,
            "impact_target": str(item.get("impact_target", "oil")).lower(),
            "method": str(item.get("method", "qwen")),
        }

    def audit(self, news: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
        sample_size = min(self.config.audit_sample_size, len(news))
        if sample_size <= 0:
            return pd.DataFrame()
        sample = news.sample(sample_size, random_state=42).sort_values("news_id")
        baseline = sample.merge(annotations, on="news_id", how="left")
        if self.client is None or not self.config.auditor_model:
            baseline["audit_status"] = "skipped_no_auditor"
            return baseline

        audit_config = RunConfig(**{**self.config.__dict__, "model": self.config.auditor_model})
        auditor = SemanticAgent(audit_config)
        try:
            audited = []
            for start in range(0, len(sample), 10):
                audited.extend(auditor._annotate_batch(sample.iloc[start : start + 10]))
                time.sleep(0.25)
            audit_df = pd.DataFrame([auditor._clean_annotation(item) for item in audited])
            merged = baseline.merge(audit_df, on="news_id", how="left", suffixes=("", "_audit"))
            merged["topic_agree"] = merged["topic"] == merged["topic_audit"]
            merged["event_type_agree"] = merged["event_type"] == merged["event_type_audit"]
            merged["sentiment_sign_agree"] = np.sign(merged["sentiment"]) == np.sign(merged["sentiment_audit"])
            merged["audit_status"] = "audited"
            return merged
        except Exception as exc:
            LOGGER.warning("Auditor model %s failed: %s", self.config.auditor_model, exc)
            baseline["audit_status"] = "auditor_unavailable"
            return baseline


def aggregate_news_features(news: pd.DataFrame, annotations: pd.DataFrame, market_index: pd.DatetimeIndex) -> pd.DataFrame:
    enriched = news.merge(annotations, on="news_id", how="left")
    for _, row in enriched[enriched["relevance"].isna()].iterrows():
        fallback = rule_based_annotation(row)
        for key, value in fallback.items():
            enriched.loc[enriched["news_id"] == row["news_id"], key] = value

    def to_trading_day(ts: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        pos = market_index.searchsorted(pd.Timestamp(ts).normalize())
        if pos >= len(market_index):
            return pd.NaT
        return market_index[pos]

    enriched["trading_date"] = enriched["date"].map(to_trading_day)
    enriched = enriched.dropna(subset=["trading_date"])
    enriched["weight"] = enriched["relevance"].astype(float) * enriched["source_weight"].astype(float)
    enriched["weighted_sentiment"] = enriched["sentiment"].astype(float) * enriched["weight"]
    enriched["weighted_uncertainty"] = enriched["uncertainty"].astype(float) * enriched["weight"]
    enriched["weighted_horizon"] = enriched["horizon_days"].astype(float) * enriched["weight"]

    grouped = enriched.groupby("trading_date")
    daily = pd.DataFrame(index=market_index)
    daily["news_count"] = grouped.size().reindex(market_index).fillna(0.0)
    daily["news_intensity"] = grouped["weight"].sum().reindex(market_index).fillna(0.0)
    denom = daily["news_intensity"].replace(0, np.nan)
    daily["news_sentiment"] = grouped["weighted_sentiment"].sum().reindex(market_index).fillna(0.0) / denom
    daily["news_uncertainty"] = grouped["weighted_uncertainty"].sum().reindex(market_index).fillna(0.0) / denom
    daily["news_horizon_days"] = grouped["weighted_horizon"].sum().reindex(market_index).fillna(0.0) / denom
    daily[["news_sentiment", "news_uncertainty", "news_horizon_days"]] = daily[
        ["news_sentiment", "news_uncertainty", "news_horizon_days"]
    ].fillna(0.0)

    for field, values, prefix in [
        ("topic", TOPICS, "topic"),
        ("event_type", EVENT_TYPES, "event"),
        ("source_domain_class", SOURCE_CLASSES, "source"),
    ]:
        pivot = (
            enriched.pivot_table(index="trading_date", columns=field, values="weight", aggfunc="sum")
            .reindex(market_index)
            .fillna(0.0)
        )
        for value in values:
            daily[f"{prefix}_{value}"] = pivot[value] if value in pivot.columns else 0.0

    base_cols = [
        "news_count",
        "news_intensity",
        "news_sentiment",
        "news_uncertainty",
        "news_horizon_days",
    ]
    for col in base_cols:
        for lag in [1, 3, 5, 10]:
            daily[f"{col}_lag{lag}"] = daily[col].shift(lag).fillna(0.0)
        for window in [3, 5, 10]:
            daily[f"{col}_roll{window}"] = daily[col].rolling(window, min_periods=1).mean()
    return daily.fillna(0.0)


def build_model_dataset(market: pd.DataFrame, news_features: pd.DataFrame) -> pd.DataFrame:
    data = market.join(news_features, how="left").fillna(0.0).copy()
    data["target_return"] = data["brent_ret"].shift(-1)
    data["target_direction"] = (data["target_return"] > 0).astype(int)
    data["target_volatility"] = data["target_return"].pow(2)
    return data.dropna(subset=["target_return"])


def market_feature_cols(data: pd.DataFrame) -> list[str]:
    cols = [
        "brent_wti_spread",
        "brent_vol_5",
        "brent_vol_10",
        "brent_vol_20",
    ]
    cols.extend([f"brent_ret_lag{i}" for i in range(1, 11)])
    cols.extend([f"wti_ret_lag{i}" for i in range(1, 6)])
    cols.extend([f"gas_ret_lag{i}" for i in range(1, 6)])
    return [col for col in cols if col in data.columns]


def simple_news_cols(data: pd.DataFrame) -> list[str]:
    cols = [
        "news_count",
        "news_intensity",
        "news_count_lag1",
        "news_intensity_lag1",
        "news_count_roll5",
        "news_intensity_roll5",
    ]
    return [col for col in cols if col in data.columns]


def all_news_cols(data: pd.DataFrame) -> list[str]:
    prefixes = ("news_", "topic_", "event_", "source_")
    return [col for col in data.columns if col.startswith(prefixes)]


def safe_fit_return_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(train[feature_cols], train["target_return"])
    pred_return = regressor.predict(test[feature_cols])

    y_train = train["target_direction"]
    if y_train.nunique() < 2:
        proba = np.full(len(test), float(y_train.iloc[0]))
    else:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
        classifier.fit(train[feature_cols], y_train)
        proba = classifier.predict_proba(test[feature_cols])[:, 1]
    direction = (proba >= 0.5).astype(int)
    signed_return = np.maximum(np.abs(pred_return), 1e-6) * np.where(direction == 1, 1, -1)
    return signed_return, direction, proba


def fit_predict_sarimax(train: pd.DataFrame, test: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    try:
        model = SARIMAX(
            train["target_return"],
            exog=train[exog_cols],
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=80)
        pred = result.get_forecast(steps=len(test), exog=test[exog_cols]).predicted_mean
        return pred.to_numpy()
    except Exception as exc:
        LOGGER.warning("SARIMAX failed, using Ridge fallback: %s", exc)
        fallback = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        fallback.fit(train[exog_cols], train["target_return"])
        return fallback.predict(test[exog_cols])


def fit_predict_garch_variance(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    try:
        series = train["brent_ret"].dropna() * 100
        model = arch_model(series, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        result = model.fit(disp="off")
        forecast = result.forecast(horizon=len(test), reindex=False)
        variance = forecast.variance.iloc[-1].to_numpy() / 10000
        if len(variance) != len(test) or np.any(~np.isfinite(variance)):
            raise ValueError("invalid GARCH variance forecast")
        return np.maximum(variance, 1e-12)
    except Exception as exc:
        LOGGER.warning("GARCH failed, using rolling variance fallback: %s", exc)
        fallback = float(train["brent_ret"].tail(20).var())
        return np.full(len(test), max(fallback, 1e-12))


def fit_predict_var(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cols = ["brent_ret", "wti_ret", "gas_ret"]
    try:
        model_data = train[cols].dropna()
        result = VAR(model_data).fit(maxlags=5, ic="aic")
        seed = model_data.values[-result.k_ar :]
        forecast = result.forecast(seed, steps=len(test))
        return forecast[:, 0]
    except Exception as exc:
        LOGGER.warning("VAR failed, using previous return fallback: %s", exc)
        return test["brent_ret_lag1"].to_numpy()


def month_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    starts = sorted(pd.Timestamp(value).to_period("M").to_timestamp() for value in index)
    result: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for value in starts:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def prediction_rows_for_window(
    train: pd.DataFrame,
    test: pd.DataFrame,
    market_cols: list[str],
    simple_news: list[str],
    full_cols: list[str],
) -> list[dict[str, Any]]:
    y_true = test["target_return"].to_numpy()
    rows: list[dict[str, Any]] = []

    model_predictions: dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = {}
    naive_return = test["brent_ret_lag1"].to_numpy()
    model_predictions["Naive previous return"] = (naive_return, (naive_return > 0).astype(int), naive_return, None)

    arimax_market = fit_predict_sarimax(train, test, market_cols)
    model_predictions["ARIMAX market"] = (arimax_market, (arimax_market > 0).astype(int), arimax_market, None)

    arimax_news_cols = market_cols + simple_news
    arimax_news = fit_predict_sarimax(train, test, arimax_news_cols)
    model_predictions["ARIMAX simple news"] = (arimax_news, (arimax_news > 0).astype(int), arimax_news, None)

    var_pred = fit_predict_var(train, test)
    model_predictions["VAR market system"] = (var_pred, (var_pred > 0).astype(int), var_pred, None)

    varx_return, varx_direction, varx_proba = safe_fit_return_classifier(train, test, arimax_news_cols)
    model_predictions["VARX/Ridge news system"] = (varx_return, varx_direction, varx_proba, None)

    market_return, market_direction, market_proba = safe_fit_return_classifier(train, test, market_cols)
    model_predictions["Market-only Ridge/LogReg"] = (market_return, market_direction, market_proba, None)

    hybrid_return, hybrid_direction, hybrid_proba = safe_fit_return_classifier(train, test, full_cols)
    model_predictions["LLM multi-agent hybrid"] = (hybrid_return, hybrid_direction, hybrid_proba, None)

    garch_variance = fit_predict_garch_variance(train, test)
    model_predictions["GARCH volatility"] = (None, None, None, garch_variance)

    for model_name, (pred_return, pred_direction, pred_score, pred_volatility) in model_predictions.items():
        for i, idx in enumerate(test.index):
            rows.append(
                {
                    "date": idx,
                    "model": model_name,
                    "target_return": y_true[i],
                    "target_direction": int(y_true[i] > 0),
                    "target_volatility": float(test["target_volatility"].iloc[i]),
                    "pred_return": np.nan if pred_return is None else float(pred_return[i]),
                    "pred_direction": np.nan if pred_direction is None else int(pred_direction[i]),
                    "pred_score": np.nan if pred_score is None else float(pred_score[i]),
                    "pred_volatility": np.nan if pred_volatility is None else float(pred_volatility[i]),
                }
            )
    return rows


def run_rolling_predictions(data: pd.DataFrame, min_train_days: int = 252) -> pd.DataFrame:
    market_cols = market_feature_cols(data)
    simple_news = simple_news_cols(data)
    full_cols = market_cols + all_news_cols(data)
    rows: list[dict[str, Any]] = []
    for month_start in month_starts(data.index):
        train = data[data.index < month_start]
        if len(train) < min_train_days:
            continue
        month_end = month_start + pd.offsets.MonthEnd(0)
        test = data[(data.index >= month_start) & (data.index <= month_end)]
        if test.empty:
            continue
        rows.extend(prediction_rows_for_window(train, test, market_cols, simple_news, full_cols))
    return pd.DataFrame(rows)


def qlike(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> float:
    eps = 1e-12
    pred = np.maximum(y_pred_var, eps)
    true = np.maximum(y_true_var, eps)
    return float(np.mean(np.log(pred) + true / pred))


def bootstrap_ci(values: np.ndarray, metric: Callable[[np.ndarray], float], block_size: int = 10, samples: int = 500) -> tuple[float, float]:
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = []
    n = len(values)
    for _ in range(samples):
        pieces = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.integers(0, max(1, n - block_size + 1)))
            pieces.append(values[start : start + block_size])
        sample = np.concatenate(pieces)[:n]
        stats.append(metric(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def max_drawdown(series: np.ndarray) -> float:
    cumulative = np.cumsum(series)
    running_max = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - running_max))


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "n_obs": int(len(group))}
        return_group = group.dropna(subset=["pred_return"])
        if not return_group.empty:
            y = return_group["target_return"].to_numpy()
            pred = return_group["pred_return"].to_numpy()
            true_dir = return_group["target_direction"].astype(int).to_numpy()
            pred_dir = return_group["pred_direction"].astype(int).to_numpy()
            score = return_group["pred_score"].to_numpy()
            strategy = np.where(pred_dir == 1, 1, -1) * y
            accuracy_values = (true_dir == pred_dir).astype(float)
            rmse_values = (y - pred) ** 2
            row.update(
                {
                    "rmse_return": float(np.sqrt(mean_squared_error(y, pred))),
                    "mae_return": float(mean_absolute_error(y, pred)),
                    "direction_accuracy": float(accuracy_score(true_dir, pred_dir)),
                    "balanced_accuracy": float(balanced_accuracy_score(true_dir, pred_dir)),
                    "f1_positive": float(f1_score(true_dir, pred_dir, zero_division=0)),
                    "mcc": float(matthews_corrcoef(true_dir, pred_dir)),
                    "roc_auc": safe_auc(true_dir, score),
                    "strategy_return": float(strategy.sum()),
                    "strategy_sharpe": float(np.sqrt(252) * strategy.mean() / strategy.std()) if strategy.std() > 0 else 0.0,
                    "max_drawdown": max_drawdown(strategy),
                }
            )
            acc_low, acc_high = bootstrap_ci(accuracy_values, lambda x: float(np.mean(x)))
            rmse_low, rmse_high = bootstrap_ci(rmse_values, lambda x: float(np.sqrt(np.mean(x))))
            row["direction_accuracy_ci_low"] = acc_low
            row["direction_accuracy_ci_high"] = acc_high
            row["rmse_return_ci_low"] = rmse_low
            row["rmse_return_ci_high"] = rmse_high
        volatility_group = group.dropna(subset=["pred_volatility"])
        if not volatility_group.empty:
            true_var = volatility_group["target_volatility"].to_numpy()
            pred_var = volatility_group["pred_volatility"].to_numpy()
            row.update(
                {
                    "rmse_volatility": float(np.sqrt(mean_squared_error(true_var, pred_var))),
                    "qlike_volatility": qlike(true_var, pred_var),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return np.nan


def run_ablation_predictions(data: pd.DataFrame, min_train_days: int = 252) -> pd.DataFrame:
    market_cols = market_feature_cols(data)
    news_cols = all_news_cols(data)
    full_cols = market_cols + news_cols
    groups = {
        "full": full_cols,
        "market_only": market_cols,
        "without_sentiment": [c for c in full_cols if "sentiment" not in c],
        "without_intensity": [c for c in full_cols if "intensity" not in c and "count" not in c],
        "without_topic": [c for c in full_cols if not c.startswith("topic_")],
        "without_uncertainty": [c for c in full_cols if "uncertainty" not in c],
        "without_event_type": [c for c in full_cols if not c.startswith("event_")],
        "without_source": [c for c in full_cols if not c.startswith("source_")],
    }
    rows: list[dict[str, Any]] = []
    for month_start in month_starts(data.index):
        train = data[data.index < month_start]
        if len(train) < min_train_days:
            continue
        month_end = month_start + pd.offsets.MonthEnd(0)
        test = data[(data.index >= month_start) & (data.index <= month_end)]
        if test.empty:
            continue
        for name, cols in groups.items():
            pred_return, pred_direction, pred_score = safe_fit_return_classifier(train, test, cols)
            for i, idx in enumerate(test.index):
                rows.append(
                    {
                        "date": idx,
                        "model": name,
                        "target_return": float(test["target_return"].iloc[i]),
                        "target_direction": int(test["target_direction"].iloc[i]),
                        "pred_return": float(pred_return[i]),
                        "pred_direction": int(pred_direction[i]),
                        "pred_score": float(pred_score[i]),
                    }
                )
    predictions = pd.DataFrame(rows)
    metrics = summarize_predictions(
        predictions.assign(target_volatility=np.nan, pred_volatility=np.nan)
    )
    full_accuracy = float(metrics.loc[metrics["model"] == "full", "direction_accuracy"].iloc[0])
    metrics["delta_accuracy_vs_full"] = full_accuracy - metrics["direction_accuracy"]
    return metrics


def mcnemar_test(predictions: pd.DataFrame, model_a: str, model_b: str) -> dict[str, Any]:
    a = predictions[predictions["model"] == model_a].set_index("date")
    b = predictions[predictions["model"] == model_b].set_index("date")
    joined = a[["target_direction", "pred_direction"]].join(
        b[["pred_direction"]], how="inner", lsuffix="_a", rsuffix="_b"
    )
    joined = joined.dropna()
    correct_a = joined["target_direction"] == joined["pred_direction_a"]
    correct_b = joined["target_direction"] == joined["pred_direction_b"]
    b_count = int((correct_a & ~correct_b).sum())
    c_count = int((~correct_a & correct_b).sum())
    stat = ((abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)) if (b_count + c_count) else 0.0
    p_value = float(chi2.sf(stat, 1)) if (b_count + c_count) else 1.0
    return {"test": "McNemar", "model_a": model_a, "model_b": model_b, "b": b_count, "c": c_count, "stat": stat, "p_value": p_value}


def diebold_mariano_test(predictions: pd.DataFrame, model_a: str, model_b: str) -> dict[str, Any]:
    a = predictions[predictions["model"] == model_a].set_index("date")
    b = predictions[predictions["model"] == model_b].set_index("date")
    joined = a[["target_return", "pred_return"]].join(
        b[["pred_return"]], how="inner", lsuffix="_a", rsuffix="_b"
    )
    joined = joined.dropna()
    if joined.empty:
        return {"test": "Diebold-Mariano", "model_a": model_a, "model_b": model_b, "stat": np.nan, "p_value": np.nan}
    loss_diff = (joined["target_return"] - joined["pred_return_a"]) ** 2 - (
        joined["target_return"] - joined["pred_return_b"]
    ) ** 2
    se = loss_diff.std(ddof=1) / math.sqrt(len(loss_diff))
    stat = float(loss_diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * norm.sf(abs(stat)))
    return {"test": "Diebold-Mariano", "model_a": model_a, "model_b": model_b, "stat": stat, "p_value": p_value}


def statistical_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("LLM multi-agent hybrid", "Market-only Ridge/LogReg"),
        ("LLM multi-agent hybrid", "ARIMAX market"),
        ("LLM multi-agent hybrid", "ARIMAX simple news"),
    ]
    rows = []
    for a, b in pairs:
        rows.append(mcnemar_test(predictions, a, b))
        rows.append(diebold_mariano_test(predictions, a, b))
    return pd.DataFrame(rows)


def monthly_rolling_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    directional = predictions.dropna(subset=["pred_direction"]).copy()
    if directional.empty:
        return pd.DataFrame()
    directional["month"] = pd.to_datetime(directional["date"]).dt.to_period("M").astype(str)
    for (month, model), group in directional.groupby(["month", "model"], sort=True):
        y_true = group["target_direction"].astype(int).to_numpy()
        y_pred = group["pred_direction"].astype(int).to_numpy()
        row = {
            "month": month,
            "model": model,
            "n_obs": int(len(group)),
            "direction_accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
        if group["pred_return"].notna().any():
            return_group = group.dropna(subset=["pred_return"])
            row["rmse_return"] = float(
                np.sqrt(mean_squared_error(return_group["target_return"], return_group["pred_return"]))
            )
            row["mae_return"] = float(mean_absolute_error(return_group["target_return"], return_group["pred_return"]))
        rows.append(row)
    return pd.DataFrame(rows)


def save_figures(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling_predictions: pd.DataFrame,
    ablations: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "Naive previous return": "Наивная модель",
        "ARIMAX market": "ARIMAX",
        "ARIMAX simple news": "ARIMAX+простые новости",
        "VAR market system": "VAR",
        "VARX/Ridge news system": "VARX/Ridge",
        "Market-only Ridge/LogReg": "Market-only",
        "LLM multi-agent hybrid": "LLM-гибрид",
        "GARCH volatility": "GARCH",
    }

    fig, ax1 = plt.subplots(figsize=(10, 5))
    data["brent"].plot(ax=ax1, color="#1f4e79", label="Brent spot")
    ax1.set_ylabel("долл. США за баррель")
    ax2 = ax1.twinx()
    data["news_intensity_roll10"].plot(ax=ax2, color="#c55a11", alpha=0.55, label="Интенсивность новостей, среднее 10 дней")
    ax2.set_ylabel("новостной фактор")
    ax1.set_title("Цена Brent и агрегированная интенсивность новостного потока")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "brent_news_intensity.png", dpi=180)
    plt.close(fig)

    model_metrics = metrics.dropna(subset=["direction_accuracy"]).copy()
    model_metrics = model_metrics.sort_values("direction_accuracy")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(model_metrics["model"].map(label_map), model_metrics["direction_accuracy"], color="#4472c4")
    ax.set_xlim(0, 1)
    ax.set_xlabel("точность направления")
    ax.set_title("Сравнение моделей по rolling out-of-sample точности")
    for i, value in enumerate(model_metrics["direction_accuracy"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "model_accuracy.png", dpi=180)
    plt.close(fig)

    rolling = rolling_predictions.dropna(subset=["pred_direction"]).copy()
    rolling["month"] = pd.to_datetime(rolling["date"]).dt.to_period("M").dt.to_timestamp()
    selected = ["ARIMAX market", "Market-only Ridge/LogReg", "LLM multi-agent hybrid"]
    monthly_rows = []
    for (month, model), group in rolling[rolling["model"].isin(selected)].groupby(["month", "model"]):
        monthly_rows.append(
            {
                "month": month,
                "model": label_map.get(model, model),
                "accuracy": float((group["target_direction"] == group["pred_direction"]).mean()),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in monthly.groupby("model"):
        group.sort_values("month").plot(x="month", y="accuracy", ax=ax, label=model)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Динамика месячной точности направленного прогноза")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("месяц")
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_accuracy.png", dpi=180)
    plt.close(fig)

    plot_ablation = ablations[ablations["model"] != "full"].copy()
    ablation_labels = {
        "market_only": "без новостей",
        "without_sentiment": "без тональности",
        "without_intensity": "без интенсивности",
        "without_topic": "без тематики",
        "without_uncertainty": "без неопределенности",
        "without_event_type": "без типа события",
        "without_source": "без типа источника",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_ablation["model"].map(ablation_labels), plot_ablation["delta_accuracy_vs_full"], color="#70ad47")
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("потеря точности относительно полной гибридной модели")
    ax.set_title("Абляционный анализ групп новостных факторов")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_ablation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative = pd.DataFrame()
    for model in ["Naive previous return", "ARIMAX market", "Market-only Ridge/LogReg", "LLM multi-agent hybrid"]:
        group = rolling_predictions[rolling_predictions["model"] == model].sort_values("date")
        if group.empty:
            continue
        signal = np.where(group["pred_direction"].astype(float) == 1, 1, -1)
        cumulative[label_map.get(model, model)] = pd.Series(signal * group["target_return"].to_numpy()).cumsum()
    cumulative.plot(ax=ax)
    ax.set_title("Кумулятивная доходность направленных прогнозов")
    ax.set_ylabel("сумма лог-доходностей")
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_directional_return.png", dpi=180)
    plt.close(fig)

    garch = rolling_predictions[rolling_predictions["model"] == "GARCH volatility"].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    if not garch.empty:
        pd.Series(garch["target_volatility"].to_numpy(), index=pd.to_datetime(garch["date"])).rolling(10).mean().plot(
            ax=ax, label="реализованная волатильность"
        )
        pd.Series(garch["pred_volatility"].to_numpy(), index=pd.to_datetime(garch["date"])).rolling(10).mean().plot(
            ax=ax, label="прогноз GARCH"
        )
    ax.set_title("GARCH: прогноз условной волатильности")
    ax.set_ylabel("дисперсия доходности")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "volatility_quality.png", dpi=180)
    plt.close(fig)

    lag_cols = [c for c in data.columns if c.startswith("news_sentiment_lag") or c.startswith("news_intensity_lag")]
    lag_rows = []
    for col in lag_cols:
        if data[col].std() > 0:
            lag_rows.append({"feature": col, "correlation": data[col].corr(data["target_return"])})
    lag_df = pd.DataFrame(lag_rows).sort_values("correlation")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(lag_df["feature"], lag_df["correlation"], color="#8064a2")
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_title("Связь лаговых новостных признаков с будущей доходностью")
    ax.set_xlabel("корреляция")
    fig.tight_layout()
    fig.savefig(output_dir / "news_lag_effects.png", dpi=180)
    plt.close(fig)


def write_outputs(
    config: RunConfig,
    market: pd.DataFrame,
    news: pd.DataFrame,
    annotations: pd.DataFrame,
    audit: pd.DataFrame,
    data: pd.DataFrame,
    rolling_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    merged_news = news.merge(annotations, on="news_id", how="left")
    merged_news.to_csv(config.output_dir / "news_annotations.csv", index=False)
    market.to_csv(config.output_dir / "market_data.csv")
    data.to_csv(config.output_dir / "daily_model_dataset.csv")
    rolling_predictions.to_csv(config.output_dir / "predictions.csv", index=False)
    monthly_metrics = monthly_rolling_metrics(rolling_predictions)
    monthly_metrics.to_csv(config.output_dir / "rolling_metrics.csv", index=False)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    ablations.to_csv(config.output_dir / "feature_ablation.csv", index=False)
    tests.to_csv(config.output_dir / "stat_tests.csv", index=False)
    audit.to_csv(config.output_dir / "annotation_audit.csv", index=False)
    meta = news.attrs.get("meta", {})
    pd.DataFrame(meta.get("language_summary", [])).to_csv(config.output_dir / "language_robustness.csv", index=False)
    save_figures(data, metrics, rolling_predictions, ablations, config.output_dir)

    return_metrics = metrics.dropna(subset=["direction_accuracy"])
    best_accuracy = return_metrics.sort_values("direction_accuracy", ascending=False).iloc[0].to_dict()
    summary = {
        "start": config.start,
        "end": config.end,
        "market_observations": int(len(market)),
        "model_dataset_rows": int(len(data)),
        "rolling_prediction_rows": int(len(rolling_predictions)),
        "rolling_metric_rows": int(len(monthly_metrics)),
        "raw_news_records": int(meta.get("raw_records", len(news))),
        "news_records": int(len(news)),
        "annotated_records": int(len(annotations)),
        "language_filter": config.language_filter,
        "llm_model": None if config.no_llm else config.model,
        "auditor_model": None if config.no_llm else config.auditor_model,
        "llm_base_url": config.base_url,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "annotation_methods": annotations["method"].value_counts(dropna=False).to_dict(),
        "audit_status": audit["audit_status"].value_counts(dropna=False).to_dict() if "audit_status" in audit.columns else {},
        "best_model_by_accuracy": best_accuracy,
        "figures": [
            "brent_news_intensity.png",
            "model_accuracy.png",
            "rolling_accuracy.png",
            "feature_ablation.png",
            "cumulative_directional_return.png",
            "volatility_quality.png",
            "news_lag_effects.png",
        ],
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = parse_args()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Fetching market data from FRED")
    market = fetch_fred_series(config.start, config.end)
    LOGGER.info("Market observations: %s", len(market))

    LOGGER.info("Fetching GDELT news headlines")
    news = fetch_gdelt_news(config)
    LOGGER.info("News records after filtering: %s", len(news))

    LOGGER.info("Annotating news with semantic agent")
    semantic_agent = SemanticAgent(config)
    annotations = semantic_agent.annotate(news)
    LOGGER.info("Annotation methods: %s", annotations["method"].value_counts(dropna=False).to_dict())

    LOGGER.info("Auditing annotation sample")
    audit = semantic_agent.audit(news, annotations)

    LOGGER.info("Aggregating news factors")
    news_features = aggregate_news_features(news, annotations, market.index)
    data = build_model_dataset(market, news_features)

    LOGGER.info("Running rolling benchmark and hybrid models")
    rolling_predictions = run_rolling_predictions(data, config.min_train_days)
    metrics = summarize_predictions(rolling_predictions)

    LOGGER.info("Running feature ablations")
    ablations = run_ablation_predictions(data, config.min_train_days)

    LOGGER.info("Running statistical tests")
    tests = statistical_tests(rolling_predictions)

    write_outputs(config, market, news, annotations, audit, data, rolling_predictions, metrics, ablations, tests)
    LOGGER.info("Done. Metrics:\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Production-grade LLM-only survey analyser.
All heavy lifting is done inside a single GPT call so the model
can still reason about sentiment, severity, etc.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
from openai import AsyncOpenAI

from .llm_utils import dedup_quotes, round_to_nearest_10

# ------------------------------------------------------------------------------


@dataclass(slots=True)
class TopicAnalysisResult:
    topic_name: str
    count: int  # ← rounded to nearest 10
    indicator: str  # Positive | Needs attention | Concerning
    key_insights: List[str]
    sample_quotes: List[str]  # already de-duped


@dataclass(slots=True)
class ComprehensiveAnalysisResult:
    survey_type: str
    industry_context: str
    total_responses: int
    topics_analyzed: List[TopicAnalysisResult]
    overall_insights: List[str]
    recommendations: List[str]
    analysis_date: datetime


class PureLLMAnalyzer:
    """
    Async, fully LLM-driven analyser.
    Chunking / embedding logic is unchanged – only the *analysis* prompt changes.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        max_concurrent: int = 10,
    ) -> None:
        self.client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # --- unchanged attributes from your original code ---------------------
        self.chunk_size: int = 1_000
        self.survey_responses: List[str] = []
        self.response_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None  # faiss.IndexFlatIP
        self.embedding_dimension = 1_536
        self.embedding_model = "text-embedding-3-small"
        # ----------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #  Public high-level entry point (kept sync wrapper for FastAPI)
    # --------------------------------------------------------------------------
    def process_survey_data(self, df: pd.DataFrame, custom_chunk: Optional[int] = None) -> bool:
        """Sync wrapper – delegates to async impl."""
        return asyncio.run(self._async_process_survey_data(df, custom_chunk))

    # --------------------------------------------------------------------------
    #  Async implementation (truncated – only the *new* analysis part shown)
    # --------------------------------------------------------------------------
    async def _async_process_survey_data(self, df: pd.DataFrame, custom_chunk: Optional[int]) -> bool:
        """
        Identical to your previous logic – only the *analysis* part changes.
        We therefore skip the chunking/embedding code here.
        """
        # … your existing chunking / embedding code …
        # After embeddings are ready we simply return True
        return True

    # --------------------------------------------------------------------------
    #  NEW – single LLM call that returns *everything* we need
    # --------------------------------------------------------------------------
    async def _analyse_topic_batch(
        self, topic: str, description: str, relevant_responses: List[str]
    ) -> Tuple[int, str, List[str], List[str]]:
        """
        One GPT call → (raw_count, indicator, insights, quotes).
        The model is explicitly asked to:
        1. Count exact mentions.
        2. Return 3 short insights.
        3. Return 5 representative quotes (we later de-dupe & trim).
        4. Decide on an indicator.
        """
        sample = "\n".join(
            f"{i+1}. {text[:200]}" for i, text in enumerate(relevant_responses[:20])
        )

        prompt = f"""
You are a survey-analysis expert.

Topic: {topic}
Description: {description}

Below are up to 20 representative responses (out of {len(relevant_responses)} total relevant).

{sample}

Task
----
1. Count how many **distinct** responses mention this topic (even slightly).
   Return the exact integer – we will round it later.
2. Provide exactly 3 concise bullet-style insights (max 15 words each).
3. Pick 5 **unique** quotes that best illustrate the issue (keep original casing).
4. Decide: Positive | Needs attention | Concerning

Output **only** valid JSON:
{{
  "exact_count": <int>,
  "indicator": "Positive" | "Needs attention" | "Concerning",
  "insights": ["insight1", "insight2", "insight3"],
  "quotes": ["quote1", "quote2", "quote3", "quote4", "quote5"]
}}
"""  # noqa: E501

        async with self.semaphore:
            raw = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=450,
            )
        content = raw.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # ultra-simple fallback
            data = {
                "exact_count": len(relevant_responses),
                "indicator": "Needs attention",
                "insights": ["Could not parse LLM answer"],
                "quotes": relevant_responses[:2],
            }

        return (
            int(data.get("exact_count", len(relevant_responses))),
            data.get("indicator", "Needs attention"),
            data.get("insights", []),
            data.get("quotes", []),
        )

    # --------------------------------------------------------------------------
    #  Public analysis orchestrator (sync wrapper)
    # --------------------------------------------------------------------------
    def comprehensive_analysis(
        self, topics: List[Tuple[str, str]]  # (name, description)
    ) -> ComprehensiveAnalysisResult:
        return asyncio.run(self._async_comprehensive_analysis(topics))

    # --------------------------------------------------------------------------
    #  Async orchestrator
    # --------------------------------------------------------------------------
    async def _async_comprehensive_analysis(
        self, topics: List[Tuple[str, str]]
    ) -> ComprehensiveAnalysisResult:
        """
        topics: list of (topic_name, topic_description) coming from
                the earlier `generate_analysis_topics` step.
        """
        if not self.survey_responses:
            raise RuntimeError("No survey data loaded")

        total = len(self.survey_responses)
        coros = []
        for name, desc in topics:
            relevant = self._semantic_search(name, top_k=50)  # your old method
            coros.append(
                self._analyse_topic_batch(name, desc, [r[0] for r in relevant])
            )

        results = await asyncio.gather(*coros)

        topic_results: List[TopicAnalysisResult] = []
        for (name, _), (raw_cnt, indicator, insights, quotes) in zip(topics, results):
            topic_results.append(
                TopicAnalysisResult(
                    topic_name=name,
                    count=round_to_nearest_10(raw_cnt),
                    indicator=indicator,
                    key_insights=insights,
                    sample_quotes=dedup_quotes(quotes, top_k=3),
                )
            )

        # --- overall insights & recommendations (same as before) --------------
        overall = await self._overall_insights(topic_results)
        recommendations = await self._recommendations(topic_results)

        return ComprehensiveAnalysisResult(
            survey_type="employee",  # you already detect this earlier
            industry_context="general",
            total_responses=total,
            topics_analyzed=topic_results,
            overall_insights=overall,
            recommendations=recommendations,
            analysis_date=datetime.utcnow(),
        )

    # --------------------------------------------------------------------------
    #  Tiny helpers
    # --------------------------------------------------------------------------
    async def _overall_insights(
        self, results: List[TopicAnalysisResult]
    ) -> List[str]:
        # … identical to your previous prompt …
        return ["Overall analysis completed"]

    async def _recommendations(
        self, results: List[TopicAnalysisResult]
    ) -> List[str]:
        # … identical to your previous prompt …
        return ["Focus on areas that need attention"]

    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        # … your existing implementation …
        return []

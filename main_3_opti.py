from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import pandas as pd
import io
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import faiss
from dataclasses import dataclass
import openai
from openai import OpenAI
import json
import random
import asyncio
import time

from services.na_handler import NAHandler
import logging
import math

load_dotenv()

# Basic logging setup; override with LOG_LEVEL env var (e.g., DEBUG, INFO)
_LOG_LEVEL = getattr(logging, os.getenv(
    'LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL, format='[%(asctime)s] %(levelname)s - %(message)s')

app = FastAPI(
    title="Survey Analysis API v4.0 - Pure LLM-Powered Dynamic Analysis",
    description="Truly dynamic LLM-powered analysis with user-customizable topics",
    version="4.0-dynamic"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class AnalysisTopic:
    name: str
    description: str
    search_terms: str
    importance: str  # high, medium, low
    source: str  # "llm_generated" or "user_defined"


@dataclass
class TopicAnalysisResult:
    topic_name: str
    count: int  # raw mention count
    # percentage of non-placeholder responses mentioning topic (0-100)
    percentage: float
    key_insights: List[str]
    sample_quotes: List[str]
    indicator: str  # "Positive", "Needs attention", "Concerning"


@dataclass
class ComprehensiveAnalysisResult:
    survey_type: str
    industry_context: str
    total_responses: int
    topics_analyzed: List[TopicAnalysisResult]
    overall_insights: List[str]
    recommendations: List[str]
    analysis_date: datetime


class PureLLMAnalyzer:
    def __init__(self, chunk_size: int = 1000, openai_api_key: Optional[str] = None):
        self.logger = logging.getLogger("SurveyAnalyzer")
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        # Model selection (tune via env):
        # - CLASSIFIER_MODEL: used for per-response classification (faster model recommended)
        # - INSIGHTS_MODEL: used for generating insights/recs (can be richer model)
        self.classifier_model = os.getenv('CLASSIFIER_MODEL', 'gpt-4o-mini')
        self.insights_model = os.getenv('INSIGHTS_MODEL', 'gpt-4')

        # accuracy toggles – **ON by default**
        self.calibrate_threshold = True
        self.deduplicate_freq = True
        self.sentiment_sample_size = 60

        self.chunk_size = chunk_size
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_vectors: Dict[int, np.ndarray] = {}
        self.use_direct_analysis = False

        self.survey_responses: List[str] = []
        self.response_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        self.survey_type: Optional[str] = None
        self.industry_context: Optional[str] = None
        self.generated_topics: List[AnalysisTopic] = []
        self.user_topics: List[AnalysisTopic] = []

        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.allow_random_embedding_fallback: bool = False
        # count mentions by classifying ALL responses via LLM
        self.full_llm_count_all: bool = True
        self._overall_sentiment_results: Optional[List[Dict[str, Any]]] = None
        self._topic_cls_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._topic_cls_cached_topics: set = set()

    # ------------- helpers / retries / parsing -------------
    async def _chat(self, messages: List[Dict[str, Any]], model: str = "gpt-4", temperature: float = 0.3, max_retries: int = 3, base_delay: float = 0.5) -> str:
        last_err = None
        for attempt in range(max_retries):
            try:
                def _call():
                    return self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                resp = await run_in_threadpool(_call)
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        raise last_err

    def _safe_json_extract(self, text: str, expect: str = "object") -> Any:
        # remove code fences/backticks
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        # try direct
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        # regex fallback
        import re
        pattern = r"\{.*\}" if expect == "object" else r"\[.*\]"
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise ValueError("Failed to parse JSON from LLM output")

    def _normalize_indicator(self, indicator: Optional[str], sentiment_score: int) -> str:
        if not indicator:
            indicator = ""
        s = indicator.strip().lower()
        pos = {"positive", "good", "favorable", "favourable",
               "satisfied", "satisfactory", "on track"}
        mid = {"needs attention", "neutral", "mixed", "watch",
               "monitor", "needs improvement", "average"}
        neg = {"concerning", "negative", "poor", "bad", "at risk", "critical"}
        if s in pos:
            return "Positive"
        if s in mid:
            return "Needs attention"
        if s in neg:
            return "Concerning"
        # derive from score if unknown
        if sentiment_score >= 66:
            return "Positive"
        if sentiment_score <= 40:
            return "Concerning"
        return "Needs attention"

    def _get_employee_id(self, row: pd.Series, idx: int) -> str:
        for key in [
                'Participant', 'Participant ID', 'ParticipantID', 'Participant Id', 'Participant_id']:
            val = row.get(key)
            if pd.notna(val):
                return str(val)
        return f'emp_{idx}'

    # ------------- calibration helper -------------
    def _calibrate_and_filter(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        scores = sorted([s for _, s in candidates], reverse=True)
        idx = int(0.30 * len(scores))
        thresh = max(0.35, scores[idx] if scores else 0.35)
        return [(t, s) for t, s in candidates if s >= thresh]

    def _round_about(self, n: int) -> int:
        """Round to the nearest 5 (e.g., 8→10, 11→10, 13→15, 18→20)."""
        return int(round(n / 5) * 5)

    def _format_count_about(self, n: int) -> str:
        """Exact if <10 else 'about N' using nearest-5 rounding."""
        return str(n) if n < 10 else f"about {self._round_about(n)}"

    def _ensure_sentiment_anchors(self):
        """Initialize sentiment anchor vectors for vector-anchored sentiment once."""
        if hasattr(self, "_sent_pos") and hasattr(self, "_sent_neg") and self._sent_pos is not None and self._sent_neg is not None:
            return
        pos_refs = [
            "I am happy and satisfied with my work",
            "Great team support and recognition",
            "Positive environment and good work-life balance",
        ]
        neg_refs = [
            "I am unhappy and dissatisfied with my work",
            "Lack of support and recognition",
            "Negative environment and poor work-life balance",
        ]
        embs = self._get_embeddings_batch(pos_refs + neg_refs)
        pos = np.mean(embs[: len(pos_refs)], axis=0)
        neg = np.mean(embs[len(pos_refs):], axis=0)
        # normalize anchors

        def _nz_norm(x):
            n = np.linalg.norm(x)
            return x if n == 0 else x / n
        self._sent_pos = _nz_norm(pos)
        self._sent_neg = _nz_norm(neg)

    def _semantic_search_indices(self, query: str, top_k: int = 5000) -> List[Tuple[int, float]]:
        """Return (index, score) pairs for FAISS search to enable respondent-level counting."""
        if self.faiss_index is None or self.embeddings is None:
            return []
        try:
            q_emb = self._get_embeddings_batch([query])[0]
            q_emb = q_emb / np.linalg.norm(q_emb)
            scores, indices = self.faiss_index.search(
                q_emb.reshape(1, -1).astype('float32'), top_k
            )
            pairs = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])
                     if 0 <= int(idx) < len(self.survey_responses)]
            # Apply same calibration thresholding as text path
            if self.calibrate_threshold and pairs:
                sc = sorted([s for _, s in pairs], reverse=True)
                i = int(0.30 * len(sc))
                thresh = max(0.35, sc[i] if sc else 0.35)
                pairs = [(i, s) for i, s in pairs if s >= thresh]
            return pairs
        except Exception:
            return []

    # ------------- chunking / embeddings -------------
    def calculate_optimal_chunk_size(self, dataset_size: int, custom_chunk_size: Optional[int] = None) -> Tuple[int, bool]:
        if dataset_size < 100:
            return 100, True
        if custom_chunk_size is not None:
            chunk_size = max(100, custom_chunk_size)
            return chunk_size, False
        calculated = max(100, int(dataset_size * 0.1))
        calculated = round(calculated / 100) * 100
        return calculated, False

    def update_chunk_size(self, new_chunk_size: int, dataset_size: int):
        if dataset_size < 100:
            self.use_direct_analysis = True
            self.chunk_size = 100
        else:
            self.chunk_size = max(100, new_chunk_size)
            self.use_direct_analysis = False

    def _divide_into_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        chunks = []
        total_rows = len(df)
        for i in range(0, total_rows, self.chunk_size):
            chunk_df = df.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk_df)
        return chunks

    def _process_chunk(self, chunk_df: pd.DataFrame, chunk_id: int) -> Dict[str, Any]:
        possible_response_cols = [
            'How do you feel about the work-life balance in the organization?',
            'Rate your overall satisfaction with the team support.',
            'What improvements would you suggest for the workplace?',
            'Any additional comments or feedback?',
            'How would you rate your overall job satisfaction?'
        ]
        available_cols = [
            col for col in possible_response_cols if col in chunk_df.columns]
        if not available_cols:
            available_cols = [col for col in chunk_df.columns
                              if any(keyword in col.lower()
                                     for keyword in ['feel', 'satisfaction', 'rate', 'support', 'improve', 'comment'])]

        chunk_responses = []
        chunk_metadata = []
        for idx, row in chunk_df.iterrows():
            employee_id = self._get_employee_id(row, idx)
            for col in available_cols:
                response = str(row[col]) if pd.notna(row[col]) else ""
                if response and response.lower() not in ['nan', 'none', '']:
                    chunk_responses.append(response)
                    chunk_metadata.append({
                        'employee_id': employee_id,
                        'question_type': col,
                        'response_text': response,
                        'chunk_id': chunk_id,
                        'response_index': len(chunk_responses) - 1
                    })

        if chunk_responses:
            chunk_embeddings = self._get_embeddings_batch(chunk_responses)
            chunk_summary_vector = np.mean(chunk_embeddings, axis=0)
            return {
                'chunk_id': chunk_id,
                'responses': chunk_responses,
                'embeddings': chunk_embeddings,
                'metadata': chunk_metadata,
                'summary_vector': chunk_summary_vector,
                'summary_text': f"Chunk {chunk_id}: {len(chunk_responses)} responses",
                'employee_count': len(set(meta['employee_id'] for meta in chunk_metadata)),
                'response_count': len(chunk_responses)
            }
        return None

    def _aggregate_chunks(self) -> bool:
        if not self.chunks:
            return False
        all_embeddings = []
        all_metadata = []
        for chunk_data in self.chunks:
            if chunk_data and 'embeddings' in chunk_data:
                all_embeddings.append(chunk_data['embeddings'])
                all_metadata.extend(chunk_data['metadata'])
                self.chunk_vectors[chunk_data['chunk_id']
                                   ] = chunk_data['summary_vector']
        if not all_embeddings:
            return False
        self.embeddings = np.vstack(all_embeddings)
        self.response_metadata = all_metadata
        self.survey_responses = [meta['response_text']
                                 for meta in all_metadata]

        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_vectors = self.embeddings / norms
        self.faiss_index.add(normalized_vectors.astype('float32'))
        return True

    def process_survey_data(self, df: pd.DataFrame, custom_chunk_size: Optional[int] = None) -> bool:
        try:
            self.logger.info(
                f"Starting survey data processing. Rows={len(df)}")
            self.survey_responses = []
            self.response_metadata = []
            self.chunks = []
            self._cache.clear()
            self._overall_sentiment_results = None
            self._topic_cls_cache.clear()
            self._topic_cls_cached_topics.clear()

            dataset_size = len(df)
            calculated_chunk_size, use_direct = self.calculate_optimal_chunk_size(
                dataset_size, custom_chunk_size)
            self.update_chunk_size(calculated_chunk_size, dataset_size)
            self.logger.info(
                f"Chunking decided. chunk_size={self.chunk_size}, direct={self.use_direct_analysis}")

            if self.use_direct_analysis:
                return self._process_small_dataset(df)
            else:
                return self._process_large_dataset(df)
        except Exception as e:
            print(f"❌ Error processing survey data: {e}")
            return False

    def _process_small_dataset(self, df: pd.DataFrame) -> bool:
        possible_response_cols = [
            'How do you feel about the work-life balance in the organization?',
            'Rate your overall satisfaction with the team support.',
            'What improvements would you suggest for the workplace?',
            'Any additional comments or feedback?',
            'How would you rate your overall job satisfaction?'
        ]
        available_cols = [
            col for col in possible_response_cols if col in df.columns]
        if not available_cols:
            available_cols = [col for col in df.columns
                              if any(keyword in col.lower()
                                     for keyword in ['feel', 'satisfaction', 'rate', 'support', 'improve', 'comment'])]

        for idx, row in df.iterrows():
            employee_id = self._get_employee_id(row, idx)
            for col in available_cols:
                response = str(row[col]) if pd.notna(row[col]) else ""
                if response and response.lower() not in ['nan', 'none', '']:
                    self.survey_responses.append(response)
                    self.response_metadata.append({
                        'employee_id': employee_id,
                        'question_type': col,
                        'response_text': response
                    })

        if self.survey_responses:
            self.embeddings = self._get_embeddings_batch(self.survey_responses)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_vectors = self.embeddings / norms
            self.faiss_index.add(normalized_vectors.astype('float32'))
            return True
        return False

    def _process_large_dataset(self, df: pd.DataFrame) -> bool:
        try:
            chunk_dataframes = self._divide_into_chunks(df)
            self.chunks = []
            for i, chunk_df in enumerate(chunk_dataframes):
                chunk_result = self._process_chunk(chunk_df, i)
                if chunk_result:
                    self.chunks.append(chunk_result)
            if not self.chunks:
                return False
            return self._aggregate_chunks()
        except Exception as e:
            print(f"❌ Error processing large dataset: {e}")
            return False

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            last_err = None
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch
                    )
                    batch_embeddings = [
                        embedding.embedding for embedding in response.data]
                    embeddings.extend(batch_embeddings)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(0.5 * (2 ** attempt))
            if last_err is not None:
                if self.allow_random_embedding_fallback:
                    print(
                        f"Embedding failed, using random fallback for batch starting {i}: {last_err}")
                    embeddings.extend([
                        np.random.normal(0, 0.1, self.embedding_dimension).tolist() for _ in batch
                    ])
                else:
                    raise last_err
        return np.array(embeddings)

    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        if self.faiss_index is None or self.embeddings is None:
            return []
        try:
            query_embedding = self._get_embeddings_batch([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), top_k
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.survey_responses):
                    results.append((self.survey_responses[idx], float(score)))

            if self.calibrate_threshold:
                results = self._calibrate_and_filter(results)
            if self.deduplicate_freq:
                seen = dict()
                deduped = []
                for text, sc in results:
                    if text not in seen:
                        seen[text] = sc
                        deduped.append((text, sc))
                results = deduped
            return [(text, score) for text, score in results if score > (0.3 if not self.calibrate_threshold else -1.0)]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    async def generate_analysis_topics(self, additional_topics: Optional[List[str]] = None) -> List[AnalysisTopic]:
        if not self.survey_responses:
            raise ValueError("No survey data available. Process data first.")
        # unbiased sample (exclude explicit placeholders if present)
        non_placeholder = [r for r in self.survey_responses if r and r.strip(
        ) and r.strip().lower() != 'no response provided']
        pool = non_placeholder if non_placeholder else self.survey_responses
        sample_size = min(20, len(pool))
        sample_responses = random.sample(pool, sample_size)
        sample_text = "\n".join(
            [f"{i+1}. {resp[:150]}" for i, resp in enumerate(sample_responses)])
        prompt = f"""
Analyze these survey responses and generate the most relevant analysis topics.
Sample responses:
{sample_text}
Based on these responses, provide ONLY a valid JSON object with:
{{
    "survey_type": "employee|customer|candidate|student|other",
    "industry_context": "detected industry context",
    "analysis_topics": [
        {{
            "name": "topic_name",
            "description": "what this topic measures",
            "search_terms": "keywords for finding related responses",
            "importance": "high|medium|low"
        }}
    ]
}}
Generate 6-10 topics that are most relevant and actionable for this specific survey.
Focus on topics that appear frequently in the responses.
"""
        try:
            response_content = await self._chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            try:
                result = self._safe_json_extract(
                    response_content, expect="object")
            except Exception:
                result = {
                    "survey_type": "employee",
                    "industry_context": "general",
                    "analysis_topics": [
                        {"name": "overall_satisfaction", "description": "Overall satisfaction levels",
                         "search_terms": "satisfaction happy pleased", "importance": "high"}
                    ]
                }
            self.survey_type = result["survey_type"]
            self.industry_context = result["industry_context"]
            self.generated_topics = [
                AnalysisTopic(
                    name=topic["name"],
                    description=topic["description"],
                    search_terms=topic["search_terms"],
                    importance=topic["importance"],
                    source="llm_generated"
                )
                for topic in result["analysis_topics"]
            ]
            self.user_topics = []
            if additional_topics:
                for topic_name in additional_topics:
                    self.user_topics.append(AnalysisTopic(
                        name=topic_name.lower().replace(" ", "_"),
                        description=f"User-defined analysis for {topic_name}",
                        search_terms=topic_name.lower(),
                        importance="high",
                        source="user_defined"
                    ))
            return self.generated_topics + self.user_topics
        except Exception as e:
            print(f"❌ Error generating topics: {e}")
            return [AnalysisTopic(
                name="overall_satisfaction",
                description="Overall satisfaction levels",
                search_terms="satisfaction happy pleased content",
                importance="high",
                source="fallback"
            )]

    # ------------------------------------------------------
    #  NEW  –  per-quote sentiment scoring
    # ------------------------------------------------------
    async def analyze_topic(self, topic: AnalysisTopic) -> TopicAnalysisResult:
        # New path: classify ALL responses via LLM once for multiple topics and count every relevant mention
        if self.full_llm_count_all:
            try:
                await self._ensure_multi_topic_classification([topic])
                # {id: {relevant, sentiment, conf}}
                per_topic = self._topic_cls_cache.get(topic.name, {})
                relevant_ids = [
                    rid for rid, r in per_topic.items() if r.get('relevant') == 1]
                mention_count = len(relevant_ids)
                total_non_placeholder = sum(1 for r in self.survey_responses if r and r.strip(
                ).lower() != 'no response provided')
                percentage = round(100.0 * mention_count /
                                   max(1, total_non_placeholder), 1)
                if mention_count == 0:
                    return TopicAnalysisResult(
                        topic_name=topic.name,
                        count=0,
                        percentage=0.0,
                        key_insights=["No relevant responses found"],
                        sample_quotes=[],
                        indicator="No data"
                    )

                # sentiment score from relevant responses
                try:
                    rel_items = [per_topic[rid] for rid in relevant_ids]
                    avg = np.average([r['sentiment'] for r in rel_items], weights=[
                                     max(0.01, r.get('conf', 0.5)) for r in rel_items])
                    sentiment_score = int(50 + 50 * avg)
                except Exception:
                    sentiment_score = 50

                # build small set of example quotes for insights prompt (limit length)
                relevant_texts = [self.survey_responses[rid]
                                  for rid in relevant_ids if 0 <= rid < len(self.survey_responses)]
                sample_for_prompt = relevant_texts[: self.sentiment_sample_size]
                responses_text = "\n".join(
                    [f"{i}. {t[:300]}" for i, t in enumerate(sample_for_prompt)])

                insights_prompt = f"""
Based on these {len(sample_for_prompt)} responses for topic: {topic.description}
and overall sentiment score {sentiment_score}/100,
provide ONLY a JSON object:
{{
    "key_insights": ["insight1", "insight2", "insight3"],
    "sample_quotes": ["quote1", "quote2"],
    "indicator": "Positive|Needs attention|Concerning"
}}

Responses:
{responses_text}
"""
                resp2_content = await self._chat(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": insights_prompt}],
                    temperature=0.3
                )
                try:
                    result2 = self._safe_json_extract(
                        resp2_content, expect="object")
                except Exception:
                    result2 = {"key_insights": [],
                               "sample_quotes": [], "indicator": ""}

                normalized_indicator = self._normalize_indicator(
                    result2.get("indicator"), sentiment_score)

                # choose sample quotes: prefer GPT-provided, else from relevant texts
                sample_quotes = result2.get(
                    "sample_quotes") or relevant_texts[:2]

                return TopicAnalysisResult(
                    topic_name=topic.name,
                    count=mention_count,
                    percentage=percentage,
                    key_insights=result2.get("key_insights", []),
                    sample_quotes=sample_quotes,
                    indicator=normalized_indicator
                )
            except Exception as e:
                print(
                    f"❌ Error in full LLM counting for topic {topic.name}: {e}")
                return TopicAnalysisResult(
                    topic_name=topic.name,
                    count=0,
                    percentage=0.0,
                    key_insights=[f"Analysis error: {str(e)}"],
                    sample_quotes=[],
                    indicator="Error"
                )

        # Vector path: semantic search on combined topic query (fast, respondent-level dedupe)
        query = f"{topic.name}. {topic.description}. {topic.search_terms}"
        pairs = self._semantic_search_indices(
            query, top_k=min(len(self.survey_responses), 10000))
        if not pairs:
            return TopicAnalysisResult(
                topic_name=topic.name,
                count=0,
                percentage=0.0,
                key_insights=["No relevant responses found"],
                sample_quotes=[],
                indicator="No data"
            )
        unique_emp_ids = set()
        for rid, _ in pairs:
            if 0 <= rid < len(self.response_metadata):
                emp_id = self.response_metadata[rid].get('employee_id')
                txt = self.survey_responses[rid] if 0 <= rid < len(
                    self.survey_responses) else None
                if emp_id is not None and txt and txt.strip().lower() != 'no response provided':
                    unique_emp_ids.add(str(emp_id))
        mention_count = len(unique_emp_ids)
        total_employees = len({m.get('employee_id')
                              for m in self.response_metadata})
        percentage = round(100.0 * mention_count / max(1, total_employees), 1)

        # sentiment via sample of relevant
        # Select sample quotes from top results
        sample_indices = [rid for rid, _ in pairs[:self.sentiment_sample_size]]
        response_texts = []
        seen_txt = set()
        for rid in sample_indices:
            if 0 <= rid < len(self.survey_responses):
                txt = self.survey_responses[rid]
                if not txt or txt.strip().lower() == 'no response provided':
                    continue
                if txt in seen_txt:
                    continue
                seen_txt.add(txt)
                response_texts.append(txt)
        sample_size = len(response_texts)
        responses_text = "\n".join(
            [f"{i}. {t[:300]}" for i, t in enumerate(response_texts)])
        prompt = f"""
You are a survey-sentiment engine.
For EACH of the {len(response_texts)} responses below give exactly one JSON line:
{{"id":<index>,"sentiment":<-1,0,1>,"conf":<0-1>}}
-1 = negative, 0 = neutral, 1 = positive.
"conf" is your confidence.
Do NOT write anything except the JSON lines.

Responses:
{responses_text}
"""
        try:
            response_content = await self._chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            try:
                scores = []
                for line in response_content.strip().splitlines():
                    line = line.strip().strip(',')
                    if not line or line.startswith('```') or line.startswith('`'):
                        continue
                    try:
                        obj = json.loads(line)
                        if 'sentiment' in obj and 'conf' in obj:
                            scores.append((obj['sentiment'], obj['conf']))
                    except Exception:
                        continue
                if not scores:
                    raise ValueError("No scores returned")
                avg = np.average([s for s, _ in scores],
                                 weights=[c for _, c in scores])
                sentiment_score = int(50 + 50 * avg)
            except Exception:
                sentiment_score = 50

            insights_prompt = f"""
Based on these {len(response_texts)} responses for topic: {topic.description}
and overall sentiment score {sentiment_score}/100,
provide ONLY a JSON object:
{{
    "key_insights": ["insight1", "insight2", "insight3"],
    "sample_quotes": ["quote1", "quote2"],
    "indicator": "Positive|Needs attention|Concerning"
}}

Responses:
{responses_text}
"""
            resp2_content = await self._chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": insights_prompt}],
                temperature=0.3
            )
            try:
                result2 = self._safe_json_extract(
                    resp2_content, expect="object")
            except Exception:
                result2 = {"key_insights": [],
                           "sample_quotes": [], "indicator": ""}
            normalized_indicator = self._normalize_indicator(
                result2.get("indicator"), sentiment_score)
            return TopicAnalysisResult(
                topic_name=topic.name,
                count=mention_count,
                percentage=percentage,
                key_insights=result2.get("key_insights", []),
                sample_quotes=result2.get("sample_quotes", []),
                indicator=normalized_indicator
            )

        except Exception as e:
            print(f"❌ Error analyzing topic {topic.name}: {e}")
            return TopicAnalysisResult(
                topic_name=topic.name,
                count=0,
                percentage=0.0,
                key_insights=[f"Analysis error: {str(e)}"],
                sample_quotes=[],
                indicator="Error"
            )

    async def _llm_classify_all_responses_for_topic(self, topic: AnalysisTopic, batch_size: int = 80) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(self.survey_responses)
        for start in range(0, total, batch_size):
            batch = self.survey_responses[start:start+batch_size]
            # skip placeholders to save tokens, but keep IDs aligned
            entries = [(start + i, txt) for i, txt in enumerate(batch)
                       if txt and txt.strip().lower() != 'no response provided']
            if not entries:
                continue
            responses_text = "\n".join(
                [f"{idx}: {text[:300]}" for idx, text in entries])
            prompt = f"""
You are classifying relevance for the topic.
Topic name: {topic.name}
Topic description: {topic.description}
Search terms (hints): {topic.search_terms}
For EACH response below output exactly one JSON line:
{{"id":<global_index>,"relevant":<0|1>,"sentiment":<-1,0,1>,"conf":<0-1>}}
- Determine relevance strictly based on the topic meaning (synonyms included).
- Do NOT output anything other than the JSON lines.

Responses (format: <global_index>: <text>):
{responses_text}
"""
            content = await self._chat(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            for line in content.strip().splitlines():
                line = line.strip().strip(',')
                if not line or line.startswith('```') or line.startswith('`'):
                    continue
                try:
                    obj = json.loads(line)
                    if 'id' in obj and 'relevant' in obj:
                        # sanitize
                        obj['id'] = int(obj['id'])
                        obj['relevant'] = 1 if int(obj['relevant']) == 1 else 0
                        if 'sentiment' in obj:
                            try:
                                obj['sentiment'] = int(obj['sentiment'])
                            except Exception:
                                obj['sentiment'] = 0
                        else:
                            obj['sentiment'] = 0
                        obj['conf'] = float(obj.get('conf', 0.5))
                        results.append(obj)
                except Exception:
                    continue
        return results

    async def _llm_classify_topics_for_all_responses(self, topics: List[AnalysisTopic], batch_size: int = 40) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Classify multiple topics in a single pass per batch of responses.
        Returns mapping: {topic_name: {id: {rel, sent, conf}}}
        """
        if not topics:
            return {}
        topic_keys = [t.name for t in topics]
        # build topic schema text
        topics_desc = []
        for t in topics:
            topics_desc.append(
                f"- {t.name}: {t.description}. Hints: {t.search_terms}")
        topics_block = "\n".join(topics_desc)

        # initialize result container
        out: Dict[str, Dict[int, Dict[str, Any]]] = {k: {} for k in topic_keys}

        total = len(self.survey_responses)
        num_batches = math.ceil(total / batch_size) if batch_size else 1
        self.logger.info(
            f"Classifying topics for all responses. topics={len(topics)} total_responses={total} batches={num_batches} batch_size={batch_size}")
        for bi, start in enumerate(range(0, total, batch_size), start=1):
            batch = self.survey_responses[start:start+batch_size]
            entries = [(start + i, txt) for i, txt in enumerate(batch)
                       if txt and txt.strip().lower() != 'no response provided']
            if not entries:
                continue
            responses_text = "\n".join(
                [f"{idx}: {text[:160]}" for idx, text in entries])
            t0 = time.time()
            prompt = f"""
You are classifying relevance and sentiment for multiple topics.
Topics (judge each independently):
{topics_block}

For EACH response below, output exactly one JSON line in this format:
{{
  "id": <global_index>,
  "topics": {{
    "{topic_keys[0]}": {{"rel": 0|1, "sent": -1|0|1, "conf": 0-1}},
    ... for each topic listed above ...
  }}
}}
Do NOT output anything other than the JSON lines. If a topic is not relevant, set rel=0 and sent=0.
Be balanced and precise: if sentiment is unclear or mixed, set sent=0. Do not assume negative without clear evidence. Neutral/no-content answers (e.g., N/A) should be rel=0.

Responses (format: <global_index>: <text>):
{responses_text}
"""
            content = await self._chat(
                model=self.classifier_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            parsed_lines = 0
            ids_in_batch = {idx for idx, _ in entries}
            covered_ids = set()
            for line in content.strip().splitlines():
                line = line.strip().strip(',')
                if not line or line.startswith('```') or line.startswith('`'):
                    continue
                try:
                    obj = json.loads(line)
                    rid = int(obj.get('id'))
                    topics_obj = obj.get('topics', {})
                    if not isinstance(topics_obj, dict):
                        continue
                    for tname in topic_keys:
                        entry = topics_obj.get(tname, {})
                        rel = 1 if int(entry.get('rel', 0)) == 1 else 0
                        try:
                            sent = int(entry.get('sent', 0))
                        except Exception:
                            sent = 0
                        try:
                            conf = float(entry.get('conf', 0.5))
                        except Exception:
                            conf = 0.5
                        out[tname][rid] = {"relevant": rel,
                                           "sentiment": sent, "conf": conf}
                    parsed_lines += 1
                    covered_ids.add(rid)
                except Exception:
                    continue
            dt = time.time() - t0
            self.logger.info(
                f"Multi-topic batch {bi}/{num_batches} processed. responses={len(entries)} parsed={parsed_lines} elapsed={dt:.2f}s")
            # Retry missing subset if any ids not covered
            missing_ids = list(ids_in_batch - covered_ids)
            if missing_ids:
                self.logger.warning(
                    f"Multi-topic batch {bi}: {len(missing_ids)} missing; retrying subset")
                sub_entries = [
                    (idx, next(text for j, text in entries if j == idx)) for idx in missing_ids]
                sub_text = "\n".join(
                    [f"{idx}: {text[:160]}" for idx, text in sub_entries])
                sub_prompt = f"""
You are classifying relevance and sentiment for multiple topics.
Topics (judge each independently):
{topics_block}

For EACH response below, output exactly one JSON line:
{{"id": <global_index>, "topics": {{"{topic_keys[0]}": {{"rel": 0|1, "sent": -1|0|1, "conf": 0-1}}, ...}}}}
If a topic is not relevant, set rel=0 and sent=0. Use sent=0 for mixed/unclear cases.

Responses (format: <global_index>: <text>):
{sub_text}
"""
                sub_content = await self._chat(
                    model=self.classifier_model,
                    messages=[{"role": "user", "content": sub_prompt}],
                    temperature=0.1
                )
                sub_parsed = 0
                for line in sub_content.strip().splitlines():
                    line = line.strip().strip(',')
                    if not line or line.startswith('```') or line.startswith('`'):
                        continue
                    try:
                        obj = json.loads(line)
                        rid = int(obj.get('id'))
                        topics_obj = obj.get('topics', {})
                        if not isinstance(topics_obj, dict):
                            continue
                        for tname in topic_keys:
                            entry = topics_obj.get(tname, {})
                            rel = 1 if int(entry.get('rel', 0)) == 1 else 0
                            try:
                                sent = int(entry.get('sent', 0))
                            except Exception:
                                sent = 0
                            try:
                                conf = float(entry.get('conf', 0.5))
                            except Exception:
                                conf = 0.5
                            out[tname][rid] = {"relevant": rel,
                                               "sentiment": sent, "conf": conf}
                        sub_parsed += 1
                    except Exception:
                        continue
                self.logger.info(
                    f"Multi-topic batch {bi} retry parsed={sub_parsed} of {len(missing_ids)}")
        return out

    async def _ensure_multi_topic_classification(self, topics: List[AnalysisTopic]):
        """Ensure we have cached per-response classification for all given topics."""
        missing = [
            t for t in topics if t.name not in self._topic_cls_cached_topics]
        if not missing:
            return
        self.logger.info(
            f"Ensuring classification for {len(missing)} topics: {', '.join(t.name for t in missing)}")
        results = await self._llm_classify_topics_for_all_responses(missing)
        for t in missing:
            self._topic_cls_cache[t.name] = results.get(t.name, {})
            self._topic_cls_cached_topics.add(t.name)

    async def _llm_sentiment_all_responses(self, batch_size: int = 80) -> List[Dict[str, Any]]:
        """Classify sentiment (-1,0,1) with confidence for ALL responses (no topic)."""
        if self._overall_sentiment_results is not None:
            return self._overall_sentiment_results
        results: List[Dict[str, Any]] = []
        total = len(self.survey_responses)
        num_batches = math.ceil(total / batch_size) if batch_size else 1
        self.logger.info(
            f"Classifying overall sentiment. total_responses={total} batches={num_batches} batch_size={batch_size}")
        for bi, start in enumerate(range(0, total, batch_size), start=1):
            batch = self.survey_responses[start:start+batch_size]
            entries = [(start + i, txt) for i, txt in enumerate(batch)
                       if txt and txt.strip().lower() != 'no response provided']
            if not entries:
                continue
            responses_text = "\n".join(
                [f"{idx}: {text[:160]}" for idx, text in entries])
            t0 = time.time()
            prompt = f"""
You are a sentiment classifier for employee survey responses.
For EACH response below output exactly one JSON line:
{{"id":<global_index>,"sentiment":<-1,0,1>,"conf":<0-1>}}
-1 = negative, 0 = neutral, 1 = positive. "conf" is your confidence.
Do NOT output anything besides the JSON lines. Be balanced: if unclear or mixed, use 0 (neutral); do not assume negative.

Responses (format: <global_index>: <text>):
{responses_text}
"""
            content = await self._chat(
                model=self.classifier_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            parsed_lines = 0
            ids_in_batch = {idx for idx, _ in entries}
            covered_ids = set()
            for line in content.strip().splitlines():
                line = line.strip().strip(',')
                if not line or line.startswith('```') or line.startswith('`'):
                    continue
                try:
                    obj = json.loads(line)
                    if 'id' in obj and 'sentiment' in obj:
                        obj['id'] = int(obj['id'])
                        try:
                            obj['sentiment'] = int(obj['sentiment'])
                        except Exception:
                            obj['sentiment'] = 0
                        obj['conf'] = float(obj.get('conf', 0.5))
                        results.append(obj)
                        parsed_lines += 1
                        covered_ids.add(obj['id'])
                except Exception:
                    continue
            dt = time.time() - t0
            self.logger.info(
                f"Overall-sentiment batch {bi}/{num_batches} processed. responses={len(entries)} parsed={parsed_lines} elapsed={dt:.2f}s")
            missing_ids = list(ids_in_batch - covered_ids)
            if missing_ids:
                self.logger.warning(
                    f"Overall-sentiment batch {bi}: {len(missing_ids)} missing; retrying subset")
                sub_entries = [
                    (idx, next(text for j, text in entries if j == idx)) for idx in missing_ids]
                sub_text = "\n".join(
                    [f"{idx}: {text[:160]}" for idx, text in sub_entries])
                sub_prompt = f"""
You are a sentiment classifier for employee survey responses.
For EACH response below output exactly one JSON line:
{{"id":<global_index>,"sentiment":<-1,0,1>,"conf":<0-1>}}
-1 = negative, 0 = neutral, 1 = positive. Be balanced: if unclear or mixed, use 0 (neutral).

Responses (format: <global_index>: <text>):
{sub_text}
"""
                sub_content = await self._chat(
                    model=self.classifier_model,
                    messages=[{"role": "user", "content": sub_prompt}],
                    temperature=0.1
                )
                sub_parsed = 0
                for line in sub_content.strip().splitlines():
                    line = line.strip().strip(',')
                    if not line or line.startswith('```') or line.startswith('`'):
                        continue
                    try:
                        obj = json.loads(line)
                        if 'id' in obj and 'sentiment' in obj:
                            obj['id'] = int(obj['id'])
                            try:
                                obj['sentiment'] = int(obj['sentiment'])
                            except Exception:
                                obj['sentiment'] = 0
                            obj['conf'] = float(obj.get('conf', 0.5))
                            results.append(obj)
                            sub_parsed += 1
                    except Exception:
                        continue
                self.logger.info(
                    f"Overall-sentiment batch {bi} retry parsed={sub_parsed} of {len(missing_ids)}")
        self._overall_sentiment_results = results
        return results

    async def overall_sentiment(self) -> Dict[str, Any]:
        if not self.survey_responses:
            raise ValueError("No survey data available. Process data first.")
        classified = await self._llm_sentiment_all_responses()
        # Aggregate counts (use confidence for sorting only)
        non_placeholder_total = sum(1 for r in self.survey_responses if r and r.strip(
        ).lower() != 'no response provided')
        pos = [r for r in classified if r.get('sentiment') == 1]
        neg = [r for r in classified if r.get('sentiment') == -1]
        neu = [r for r in classified if r.get('sentiment') == 0]

        def pct(n):
            return round(100.0 * n / max(1, non_placeholder_total), 1)
        # Pick top confident quotes

        def top_quotes(items, k=3):
            items_sorted = sorted(
                items, key=lambda x: x.get('conf', 0), reverse=True)
            quotes = []
            for it in items_sorted[:k]:
                idx = it['id']
                if 0 <= idx < len(self.survey_responses):
                    text = self.survey_responses[idx]
                    if text and text.strip().lower() != 'no response provided':
                        quotes.append(text)
            return quotes
        return {
            "total_responses": len(self.survey_responses),
            "total_non_placeholder": non_placeholder_total,
            "positive": {
                "count": len(pos),
                "percentage": pct(len(pos)),
                "sample_quotes": top_quotes(pos)
            },
            "negative": {
                "count": len(neg),
                "percentage": pct(len(neg)),
                "sample_quotes": top_quotes(neg)
            },
            "neutral": {
                "count": len(neu),
                "percentage": pct(len(neu))
            }
        }

    async def overall_sentiment_vector(self) -> Dict[str, Any]:
        """Overall sentiment via vector-anchored method, aggregated per-employee with 'about N' counts."""
        if not self.survey_responses or self.embeddings is None:
            raise ValueError("No survey data available. Process data first.")
        self._ensure_sentiment_anchors()
        # Normalize embeddings
        E = self.embeddings
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1
        E_norm = E / norms
        # Response-level scores: cos(pos) - cos(neg)
        pos_scores = E_norm @ self._sent_pos.astype(E_norm.dtype)
        neg_scores = E_norm @ self._sent_neg.astype(E_norm.dtype)
        diffs = pos_scores - neg_scores
        # Map to employees (skip placeholders)
        emp_to_scores: Dict[str, List[float]] = {}
        for idx, meta in enumerate(self.response_metadata):
            txt = meta.get('response_text')
            if not txt or txt.strip().lower() == 'no response provided':
                continue
            emp = str(meta.get('employee_id'))
            emp_to_scores.setdefault(emp, []).append(float(diffs[idx]))
        # Aggregate per employee and classify
        POS_T = 0.15
        NEG_T = -0.15
        emp_labels: Dict[str, int] = {}
        for emp, vals in emp_to_scores.items():
            avg = float(np.mean(vals))
            if avg >= POS_T:
                emp_labels[emp] = 1
            elif avg <= NEG_T:
                emp_labels[emp] = -1
            else:
                emp_labels[emp] = 0
        total_employees = len({str(m.get('employee_id'))
                              for m in self.response_metadata})
        pos_emps = [e for e, lab in emp_labels.items() if lab == 1]
        neg_emps = [e for e, lab in emp_labels.items() if lab == -1]
        neu_emps = [e for e, lab in emp_labels.items() if lab == 0]

        def pct(n):
            return round(100.0 * n / max(1, total_employees), 1)

        # Sample quotes by top/bottom diffs
        def top_quotes_by_sign(sign: int, k=3):
            indices = np.argsort(diffs)  # ascending
            out = []
            seen = set()
            it = reversed(indices) if sign > 0 else iter(indices)
            for rid in it:
                if len(out) >= k:
                    break
                if 0 <= rid < len(self.survey_responses):
                    txt = self.survey_responses[rid]
                    if not txt or txt.strip().lower() == 'no response provided':
                        continue
                    if txt in seen:
                        continue
                    seen.add(txt)
                    out.append(txt)
            return out

        return {
            "total_responses": len(self.survey_responses),
            "total_non_placeholder": total_employees,
            "positive": {
                "count": self._format_count_about(len(pos_emps)),
                "percentage": pct(len(pos_emps)),
                "sample_quotes": top_quotes_by_sign(+1)
            },
            "negative": {
                "count": self._format_count_about(len(neg_emps)),
                "percentage": pct(len(neg_emps)),
                "sample_quotes": top_quotes_by_sign(-1)
            },
            "neutral": {
                "count": self._format_count_about(len(neu_emps)),
                "percentage": pct(len(neu_emps))
            }
        }

    async def comprehensive_analysis(self) -> ComprehensiveAnalysisResult:
        if not self.generated_topics and not self.user_topics:
            raise ValueError(
                "No analysis topics available. Generate topics first.")
        all_topics = self.generated_topics + self.user_topics
        # Ensure multi-topic classification once for speed
        if self.full_llm_count_all:
            await self._ensure_multi_topic_classification(all_topics)
        # Analyze topics concurrently (mainly insights generation)
        tasks = [self.analyze_topic(topic) for topic in all_topics]
        topic_results = await asyncio.gather(*tasks)
        overall_insights = await self._generate_overall_insights(topic_results)
        recommendations = await self._generate_recommendations(topic_results)
        return ComprehensiveAnalysisResult(
            survey_type=self.survey_type or "unknown",
            industry_context=self.industry_context or "unknown",
            total_responses=len(self.survey_responses),
            topics_analyzed=topic_results,
            overall_insights=overall_insights,
            recommendations=recommendations,
            analysis_date=datetime.now()
        )

    async def _generate_overall_insights(self, topic_results: List[TopicAnalysisResult]) -> List[str]:
        summary = [
            f"{r.topic_name}: {r.count} mentions ({r.percentage}%), {r.indicator}" for r in topic_results]
        prompt = f"""
Based on these results:
{chr(10).join(summary)}
Survey type: {self.survey_type}
Total responses: {len(self.survey_responses)}
Provide 3-5 key overall insights as JSON array: ["insight1", "insight2", "insight3"]
"""
        try:
            content = await self._chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            try:
                parsed = self._safe_json_extract(content, expect="array")
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            return [content]
        except Exception:
            return [f"Overall analysis of {len(topic_results)} topics completed"]

    async def _generate_recommendations(self, topic_results: List[TopicAnalysisResult]) -> List[str]:
        concerning = [r for r in topic_results if r.indicator in [
            "Needs attention", "Concerning"]]
        if not concerning:
            return ["Overall survey results are positive - continue current practices"]
        topics_text = "\n".join(
            [f"{t.topic_name}: {t.indicator}" for t in concerning])
        prompt = f"""
Based on these concerning areas:
{topics_text}
Provide 3-5 actionable recommendations as JSON array: ["rec1", "rec2"]
"""
        try:
            content = await self._chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            try:
                parsed = self._safe_json_extract(content, expect="array")
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            return [content]
        except Exception:
            return ["Focus on areas that need attention based on survey feedback"]


# ------------------  FASTAPI ROUTES  ------------------
na_handler = NAHandler()
llm_analyzer = PureLLMAnalyzer(chunk_size=1000)
llm_analyzer.full_llm_count_all = False


@app.get("/")
async def root():
    return {"message": "Survey Analysis API v4.0 - Pure LLM-Powered Dynamic Analysis", "status": "operational", "accuracy_mode": "enhanced"}


@app.post("/analyze/process")
async def process_survey_data(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(
        None, description="Optional custom chunk size")
):
    try:
        contents = await file.read()
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df_test = pd.read_excel(io.BytesIO(contents), header=None, nrows=3)
            if (pd.isna(df_test.iloc[0]).sum() > len(df_test.columns) / 2 or 'Participant' in str(df_test.iloc[1].values)):
                df = pd.read_excel(io.BytesIO(contents), header=1)
            else:
                df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(contents.decode('latin-1')))
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file format")

        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')
        success = llm_analyzer.process_survey_data(df_clean, chunk_size)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data")

        processing_type = "Direct LLM Analysis" if llm_analyzer.use_direct_analysis else "Hierarchical Chunking"
        return {
            "message": "Survey data processed successfully",
            "filename": file.filename,
            "processing_type": processing_type,
            "dataset_size": len(df_clean),
            "chunk_size_used": llm_analyzer.chunk_size,
            "total_responses": len(llm_analyzer.survey_responses),
            "chunks_created": len(llm_analyzer.chunks) if not llm_analyzer.use_direct_analysis else 0,
            "next_step": "Call GET /analyze/topics to generate analysis topics",
            "processing_date": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/analyze/topics")
async def generate_analysis_topics(additional_topics: Optional[str] = None):
    try:
        if not llm_analyzer.survey_responses:
            raise HTTPException(
                status_code=400, detail="No survey data available. Process data first.")
        user_topics = []
        if additional_topics:
            user_topics = [topic.strip()
                           for topic in additional_topics.split(',') if topic.strip()]
        all_topics = await llm_analyzer.generate_analysis_topics(user_topics)
        return {
            "message": "Analysis topics generated successfully",
            "survey_type": llm_analyzer.survey_type,
            "industry_context": llm_analyzer.industry_context,
            "total_topics": len(all_topics),
            "llm_generated_topics": [
                {
                    "name": topic.name,
                    "description": topic.description,
                    "search_terms": topic.search_terms,
                    "importance": topic.importance
                }
                for topic in all_topics if topic.source == "llm_generated"
            ],
            "user_defined_topics": [
                {
                    "name": topic.name,
                    "description": topic.description,
                    "search_terms": topic.search_terms,
                    "importance": topic.importance
                }
                for topic in all_topics if topic.source == "user_defined"
            ],
            "next_step": "Call GET /analyze/comprehensive to get complete analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating topics: {str(e)}")


@app.get("/analyze/comprehensive")
async def get_comprehensive_analysis():
    try:
        if not llm_analyzer.generated_topics and not llm_analyzer.user_topics:
            raise HTTPException(
                status_code=400, detail="No analysis topics available. Generate topics first.")
        result = await llm_analyzer.comprehensive_analysis()
        return {
            "message": "Comprehensive analysis completed successfully",
            "survey_type": result.survey_type,
            "industry_context": result.industry_context,
            "total_responses": result.total_responses,
            "topics_analyzed": len(result.topics_analyzed),
            "topic_results": [
                {
                    "topic": topic.topic_name,
                    "count": (str(topic.count) if topic.count < 10 else f"about {llm_analyzer._round_about(topic.count)}"),
                    "percentage": topic.percentage,
                    "indicator": topic.indicator,
                    "key_insights": topic.key_insights,
                    "sample_quotes": topic.sample_quotes[:2]
                }
                for topic in result.topics_analyzed
            ],
            "overall_insights": result.overall_insights,
            "recommendations": result.recommendations,
            "analysis_date": result.analysis_date
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing comprehensive analysis: {str(e)}")


@app.get("/analyze/overall-sentiment")
async def get_overall_sentiment():
    try:
        if not llm_analyzer.survey_responses:
            raise HTTPException(
                status_code=400, detail="No survey data available. Process data first.")
        result = await llm_analyzer.overall_sentiment_vector()
        return {
            "message": "Overall sentiment analysis completed successfully",
            **result,
            "analysis_date": datetime.now()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing overall sentiment analysis: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

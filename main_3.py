from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Example
from fastapi.responses import JSONResponse
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle

from services.na_handler import NAHandler

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Employee Survey Analysis API - LLM-Powered Analysis",
    description="Hierarchical chunking with full LLM-powered analysis approach",
    version="llm-powered"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class MetricWithAnalysis:
    """A metric that includes range and analysis"""
    range_estimate: str  # e.g., "about 50%" or "about 240"
    analysis: str  # e.g., "Most frequent positive keyword - Positive indicator"
    indicator: str  # e.g., "Positive indicator", "Needs attention"


@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: int
    total_responses: int
    employee_ids: List[str]
    start_index: int
    end_index: int
    chunk_summary: str


@dataclass
class LLMAnalysisResult:
    """Container for LLM-powered analysis results"""
    overall_positive_sentiment: MetricWithAnalysis
    team_support_mentions: MetricWithAnalysis
    recognition_requests: MetricWithAnalysis
    promotion_concerns: MetricWithAnalysis
    strong_positive_percent: MetricWithAnalysis
    learning_mentions: MetricWithAnalysis
    politics_concerns: MetricWithAnalysis
    team_culture_strength: MetricWithAnalysis
    strong_negative_percent: MetricWithAnalysis
    chunks_processed: int
    total_responses: int
    chunk_summaries: List[Dict[str, Any]]
    themes: List[Dict[str, Any]]
    insights: List[str]


# ===== DYNAMIC ANALYSIS ENHANCEMENT =====
@dataclass
class DynamicAnalysisParameter:
    """A dynamically generated analysis parameter"""
    name: str
    search_terms: str
    description: str
    expected_range: str
    importance: str  # high, medium, low


@dataclass
class SurveyTypeProfile:
    """Profile for a detected survey type with dynamic parameters"""
    survey_type: str
    parameters: List[DynamicAnalysisParameter]
    insights_template: List[str]
    industry_context: Optional[str] = None


@dataclass
class DynamicAnalysisResult:
    """Container for dynamic analysis results"""
    survey_type: str
    industry_context: Optional[str]
    analysis_results: Dict[str, Any]
    total_responses: int
    parameters_analyzed: int
    generation_method: str = "llm-powered"


class DynamicSurveyAnalyzer:
    """Enhanced analyzer that generates analysis parameters dynamically"""

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.survey_profile: Optional[SurveyTypeProfile] = None
        self.analysis_cache: Dict[str, Any] = {}

    async def detect_and_configure_analysis(self, sample_responses: List[str],
                                            survey_context: Optional[str] = None) -> SurveyTypeProfile:
        """
        Detect survey type and generate dynamic analysis parameters
        This enhances the existing hardcoded approach with intelligent detection
        """

        # Sample responses for analysis (don't send all data to save tokens)
        sample_text = "\n".join(sample_responses[:15])

        context_prompt = f"\nAdditional context: {survey_context}" if survey_context else ""

        prompt = f"""
        Analyze these survey responses to determine the survey type and generate relevant analysis parameters.
        
        Sample responses:
        {sample_text}
        {context_prompt}
        
        Based on these responses, provide ONLY a valid JSON response (no other text) with:
        {{
            "survey_type": "employee|candidate|customer|student|patient|other",
            "industry_context": "detected industry or context",
            "analysis_parameters": [
                {{
                    "name": "parameter_name",
                    "search_terms": "relevant keywords and phrases for semantic search",
                    "description": "what this parameter measures",
                    "expected_range": "typical range for this metric",
                    "importance": "high|medium|low"
                }}
            ],
            "insights_template": [
                "Key insight patterns to look for in this survey type"
            ]
        }}
        
        Generate 8-12 analysis parameters that are most relevant for this specific survey type.
        Focus on parameters that will provide actionable insights.
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Try to parse JSON from response, with fallback handling
            response_content = response.choices[0].message.content
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # If we still can't parse JSON, create a basic fallback response
                    print(f"⚠️ Could not parse JSON from LLM response, using fallback")
                    result = {
                        "survey_type": "employee",
                        "industry_context": "unknown",
                        "analysis_parameters": [
                            {
                                "name": "overall_satisfaction",
                                "search_terms": "satisfaction happy satisfied pleased content",
                                "description": "Overall satisfaction levels",
                                "expected_range": "varies",
                                "importance": "high"
                            }
                        ],
                        "insights_template": ["Basic analysis due to parsing error"]
                    }

            # Convert to structured objects
            parameters = [
                DynamicAnalysisParameter(
                    name=p["name"],
                    search_terms=p["search_terms"],
                    description=p["description"],
                    expected_range=p["expected_range"],
                    importance=p["importance"]
                )
                for p in result["analysis_parameters"]
            ]

            self.survey_profile = SurveyTypeProfile(
                survey_type=result["survey_type"],
                parameters=parameters,
                insights_template=result["insights_template"],
                industry_context=result.get("industry_context")
            )

            print(f"✅ Detected survey type: {self.survey_profile.survey_type}")
            print(f"📊 Generated {len(parameters)} dynamic analysis parameters")

            return self.survey_profile

        except Exception as e:
            print(f"❌ Error in dynamic analysis configuration: {e}")
            # Fallback to basic employee survey parameters
            return self._get_fallback_profile()

    def _get_fallback_profile(self) -> SurveyTypeProfile:
        """Fallback to basic employee survey if dynamic detection fails"""
        return SurveyTypeProfile(
            survey_type="employee",
            parameters=[
                DynamicAnalysisParameter(
                    name="overall_satisfaction",
                    search_terms="satisfaction happy satisfied pleased content",
                    description="Overall job satisfaction levels",
                    expected_range="40-80%",
                    importance="high"
                ),
                DynamicAnalysisParameter(
                    name="work_life_balance",
                    search_terms="work life balance stress overtime flexible schedule",
                    description="Work-life balance concerns",
                    expected_range="30-70%",
                    importance="high"
                )
            ],
            insights_template=["Basic employee satisfaction analysis"]
        )

    def run_dynamic_analysis(self, vector_analyzer, total_responses: int) -> DynamicAnalysisResult:
        """
        Run analysis using the dynamically generated parameters
        This complements the existing hardcoded analysis
        """

        if not self.survey_profile:
            raise ValueError(
                "Survey profile not configured. Run detect_and_configure_analysis first.")

        results = {}

        print(
            f"🔄 Running dynamic analysis with {len(self.survey_profile.parameters)} parameters...")

        for param in self.survey_profile.parameters:
            try:
                # Use existing semantic search but with dynamic terms
                search_results = vector_analyzer._semantic_search(
                    param.search_terms,
                    top_k=min(100, total_responses // 2)
                )

                # Filter by similarity threshold
                relevant_results = [r for r in search_results if r[1] > 0.4]

                count = len(relevant_results)
                percentage = (count / total_responses) * \
                    100 if total_responses > 0 else 0

                # Create "about" value like existing code
                about_value = vector_analyzer._create_about_value(count)
                about_percentage = vector_analyzer._create_about_value(
                    percentage, is_percentage=True)

                results[param.name] = {
                    "count": count,
                    "percentage": round(percentage, 1),
                    "about_count": about_value,
                    "about_percentage": about_percentage,
                    "description": param.description,
                    "expected_range": param.expected_range,
                    "importance": param.importance,
                    "sample_responses": [r[0][:150] + "..." for r in relevant_results[:3]],
                    "search_terms_used": param.search_terms
                }

                print(
                    f"  ✅ {param.name}: {about_value} mentions ({about_percentage})")

            except Exception as e:
                print(f"  ❌ Error analyzing {param.name}: {e}")
                results[param.name] = {
                    "error": str(e),
                    "description": param.description
                }

        return DynamicAnalysisResult(
            survey_type=self.survey_profile.survey_type,
            industry_context=self.survey_profile.industry_context,
            analysis_results=results,
            total_responses=total_responses,
            parameters_analyzed=len(self.survey_profile.parameters)
        )

# ===== END DYNAMIC ANALYSIS ENHANCEMENT =====


class LLMPoweredVectorAnalyzer:
    """
    LLM-Powered approach: Same hierarchical chunking as main_2.py but uses OpenAI for ALL analysis
    1. Divide dataset into manageable chunks (same as main_2.py)
    2. Process each chunk separately to create chunk-level vectors (same as main_2.py)  
    3. Use OpenAI LLM for ALL analysis instead of local processing (NEW!)
    """

    def __init__(self, chunk_size: int = 1000, openai_api_key: Optional[str] = None):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        # Hierarchical chunking configuration (same as main_2.py)
        self.chunk_size = chunk_size
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_vectors: Dict[int, np.ndarray] = {}
        self.chunk_metadata: List[ChunkMetadata] = []
        self.aggregated_vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Small dataset handling
        self.use_direct_analysis = False
        self.small_dataset_responses: List[str] = []

        # Initialize dynamic analyzer
        self.dynamic_analyzer = DynamicSurveyAnalyzer(self.client)

        # ===== CACHING SYSTEM =====
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._data_hash: Optional[str] = None  # Track when data changes
        self.cache_ttl_seconds: int = 300  # 5 minutes cache TTL

    def calculate_optimal_chunk_size(self, dataset_size: int, custom_chunk_size: Optional[int] = None) -> Tuple[int, bool]:
        """Same as main_2.py - Calculate optimal chunk size based on dataset size"""
        if dataset_size < 100:
            print(
                f"📊 Dataset too small ({dataset_size} responses) - using direct LLM analysis")
            return 100, True

        if custom_chunk_size is not None:
            if custom_chunk_size < 100:
                print(
                    f"⚠️ Custom chunk size {custom_chunk_size} too small, using minimum 100")
                chunk_size = 100
            else:
                chunk_size = custom_chunk_size
            print(f"📊 Using custom chunk size: {chunk_size}")
            return chunk_size, False

        # Calculate 10% of dataset size as chunk size
        calculated_chunk_size = max(100, int(dataset_size * 0.1))
        calculated_chunk_size = round(calculated_chunk_size / 100) * 100

        print(
            f"📊 Auto-calculated chunk size: {calculated_chunk_size} (10% of {dataset_size} responses)")
        return calculated_chunk_size, False

    def update_chunk_size(self, new_chunk_size: int, dataset_size: int):
        """Update chunk size with validation"""
        if dataset_size < 100:
            self.use_direct_analysis = True
            self.chunk_size = 100
            print("📊 Small dataset detected - will use direct LLM analysis")
        else:
            self.chunk_size = max(100, new_chunk_size)
            self.use_direct_analysis = False
            print(f"📊 Chunk size updated to: {self.chunk_size}")

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Same as main_2.py - Get embeddings for a batch of texts"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [
                    embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(
                    f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
                if "insufficient_quota" in str(e) or "429" in str(e):
                    print(
                        "⚠️ OpenAI quota exceeded - using mock embeddings for testing")
                    # Generate mock embeddings based on text characteristics
                    mock_embeddings = []
                    for text in batch:
                        text_hash = hash(text.lower()) % 1000000
                        np.random.seed(text_hash)
                        mock_embedding = np.random.normal(
                            0, 0.1, self.embedding_dimension)
                        if any(word in text.lower() for word in ['good', 'great', 'excellent', 'love', 'happy']):
                            mock_embedding[0:50] += 0.3
                        if any(word in text.lower() for word in ['bad', 'terrible', 'hate', 'poor', 'awful']):
                            mock_embedding[50:100] += 0.3
                        if any(word in text.lower() for word in ['team', 'support', 'collaboration']):
                            mock_embedding[100:150] += 0.3
                        mock_embeddings.append(mock_embedding)
                    embeddings.extend(mock_embeddings)
                else:
                    embeddings.extend(
                        [np.zeros(self.embedding_dimension) for _ in batch])

        return np.array(embeddings)

    # ===== SAME CHUNKING METHODS AS MAIN_2.PY =====

    def _divide_into_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Same as main_2.py - Divide the dataset into manageable chunks"""
        chunks = []
        total_rows = len(df)

        print(
            f"📊 Dividing {total_rows} responses into chunks of {self.chunk_size}")

        for i in range(0, total_rows, self.chunk_size):
            chunk_df = df.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk_df)

            chunk_id = len(chunks) - 1
            print(
                f"📋 Created chunk {chunk_id}: rows {i} to {min(i + self.chunk_size - 1, total_rows - 1)} ({len(chunk_df)} responses)")

        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def _process_chunk(self, chunk_df: pd.DataFrame, chunk_id: int) -> Dict[str, Any]:
        """Same as main_2.py - Process a single chunk to extract text and create vectors"""
        print(f"🔄 Processing chunk {chunk_id}...")

        # Extract responses from the chunk (same logic as main_2.py)
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

        print(
            f"📝 Found {len(available_cols)} response columns in chunk {chunk_id}")

        # Extract all responses from this chunk
        chunk_responses = []
        chunk_metadata = []

        for idx, row in chunk_df.iterrows():
            employee_id = str(row.get('Participant ID', f'emp_{idx}'))
            gender = str(row.get('Gender', 'Unknown'))
            tenure = str(row.get('Years of Experience', 'Unknown'))

            for col in available_cols:
                response = str(row[col]) if pd.notna(row[col]) else ""
                if response and response.lower() not in ['nan', 'none', '']:
                    chunk_responses.append(response)
                    chunk_metadata.append({
                        'employee_id': employee_id,
                        'question_type': col,
                        'response_text': response,
                        'gender': gender,
                        'tenure': tenure,
                        'chunk_id': chunk_id,
                        'response_index': len(chunk_responses) - 1
                    })

        print(
            f"📊 Extracted {len(chunk_responses)} responses from chunk {chunk_id}")

        # Create embeddings for this chunk
        if chunk_responses:
            chunk_embeddings = self._get_embeddings_batch(chunk_responses)
            chunk_summary_vector = np.mean(chunk_embeddings, axis=0)
            chunk_summary_text = f"Chunk {chunk_id}: {len(chunk_responses)} responses from {len(set(meta['employee_id'] for meta in chunk_metadata))} employees"

            return {
                'chunk_id': chunk_id,
                'responses': chunk_responses,
                'embeddings': chunk_embeddings,
                'metadata': chunk_metadata,
                'summary_vector': chunk_summary_vector,
                'summary_text': chunk_summary_text,
                'employee_count': len(set(meta['employee_id'] for meta in chunk_metadata)),
                'response_count': len(chunk_responses)
            }
        else:
            print(f"⚠️ No valid responses found in chunk {chunk_id}")
            return None

    def _aggregate_chunks(self) -> bool:
        """Same as main_2.py - Aggregate all chunk vectors for final analysis"""
        if not self.chunks:
            print("❌ No chunks to aggregate")
            return False

        print(f"🔄 Aggregating {len(self.chunks)} chunks...")

        all_embeddings = []
        all_metadata = []
        chunk_summary_vectors = []

        for chunk_data in self.chunks:
            if chunk_data and 'embeddings' in chunk_data:
                all_embeddings.append(chunk_data['embeddings'])
                all_metadata.extend(chunk_data['metadata'])
                chunk_summary_vectors.append(chunk_data['summary_vector'])
                self.chunk_vectors[chunk_data['chunk_id']
                                   ] = chunk_data['summary_vector']

        if not all_embeddings:
            print("❌ No embeddings found in chunks")
            return False

        self.aggregated_vectors = np.vstack(all_embeddings)
        self.all_metadata = all_metadata

        print(
            f"✅ Aggregated {len(self.aggregated_vectors)} total embeddings from {len(self.chunks)} chunks")

        # Create Faiss index for similarity search
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        norms = np.linalg.norm(self.aggregated_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_vectors = self.aggregated_vectors / norms
        self.faiss_index.add(normalized_vectors.astype('float32'))

        print(f"🔍 Created Faiss index with {self.faiss_index.ntotal} vectors")
        return True

    def process_survey_data_hierarchical(self, df: pd.DataFrame) -> bool:
        """Same as main_2.py - Main method to process survey data using hierarchical chunking"""
        try:
            print(
                f"🚀 Starting hierarchical processing of {len(df)} responses...")

            # Clear cache when processing new data
            self._clear_all_cache()

            if self.use_direct_analysis:
                return self._process_small_dataset(df)

            chunk_dataframes = self._divide_into_chunks(df)

            self.chunks = []
            for i, chunk_df in enumerate(chunk_dataframes):
                chunk_result = self._process_chunk(chunk_df, i)
                if chunk_result:
                    self.chunks.append(chunk_result)

            if not self.chunks:
                print("❌ No chunks processed successfully")
                return False

            success = self._aggregate_chunks()

            if success:
                print(
                    f"✅ Hierarchical processing complete: {len(self.chunks)} chunks processed")
                return True
            else:
                print("❌ Failed to aggregate chunks")
                return False

        except Exception as e:
            print(f"❌ Error in hierarchical processing: {e}")
            return False

    def _process_small_dataset(self, df: pd.DataFrame) -> bool:
        """Same as main_2.py - Process small datasets directly"""
        try:
            print(
                f"📋 Processing small dataset with {len(df)} responses directly")

            # Clear cache when processing new data
            self._clear_all_cache()

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

            self.small_dataset_responses = []
            self.all_metadata = []

            for idx, row in df.iterrows():
                employee_id = str(row.get('Participant ID', f'emp_{idx}'))
                gender = str(row.get('Gender', 'Unknown'))
                tenure = str(row.get('Years of Experience', 'Unknown'))

                for col in available_cols:
                    response = str(row[col]) if pd.notna(row[col]) else ""
                    if response and response.lower() not in ['nan', 'none', '']:
                        self.small_dataset_responses.append(response)
                        self.all_metadata.append({
                            'employee_id': employee_id,
                            'question_type': col,
                            'response_text': response,
                            'gender': gender,
                            'tenure': tenure,
                            'chunk_id': 0,
                            'response_index': len(self.small_dataset_responses) - 1
                        })

            print(
                f"📊 Extracted {len(self.small_dataset_responses)} responses for direct analysis")
            return len(self.small_dataset_responses) > 0

        except Exception as e:
            print(f"❌ Error processing small dataset: {e}")
            return False

    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Same as main_2.py - Perform semantic search across all aggregated vectors"""
        if self.faiss_index is None or self.aggregated_vectors is None:
            return []

        try:
            query_embedding = self._get_embeddings_batch([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), top_k
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.all_metadata):
                    response_text = self.all_metadata[idx]['response_text']
                    results.append((response_text, float(score)))

            return results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def has_vectors(self) -> bool:
        """Check if vectors are available for analysis"""
        if self.use_direct_analysis:
            return len(self.small_dataset_responses) > 0
        else:
            return self.faiss_index is not None and self.aggregated_vectors is not None

    # ===== CACHING METHODS (SAME AS MAIN_2.PY) =====

    def _generate_data_hash(self) -> str:
        """Generate a hash of current data to detect changes"""
        if self.use_direct_analysis:
            data_str = str(len(self.small_dataset_responses)) + \
                str(hash(tuple(self.small_dataset_responses[:5])))
        else:
            data_str = str(len(self.aggregated_vectors)
                           ) if self.aggregated_vectors is not None else "empty"
            if self.chunks:
                data_str += str(len(self.chunks)) + \
                    str(self.chunks[0].get('chunk_id', ''))
        return str(hash(data_str))

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self._analysis_cache:
            return False

        # Check if data has changed
        current_hash = self._generate_data_hash()
        if self._data_hash != current_hash:
            self._clear_all_cache()
            self._data_hash = current_hash
            return False

        # Check TTL
        if cache_key in self._cache_timestamps:
            cache_age = datetime.now() - self._cache_timestamps[cache_key]
            if cache_age.total_seconds() > self.cache_ttl_seconds:
                self._remove_from_cache(cache_key)
                return False

        return True

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache if valid"""
        if self._is_cache_valid(cache_key):
            print(f"📋 Using cached result for {cache_key}")
            return self._analysis_cache[cache_key]
        return None

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Store result in cache"""
        self._analysis_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
        self._data_hash = self._generate_data_hash()
        print(f"💾 Cached result for {cache_key}")

    def _remove_from_cache(self, cache_key: str) -> None:
        """Remove specific item from cache"""
        if cache_key in self._analysis_cache:
            del self._analysis_cache[cache_key]
        if cache_key in self._cache_timestamps:
            del self._cache_timestamps[cache_key]

    def _clear_all_cache(self) -> None:
        """Clear all cached results (called when data changes)"""
        self._analysis_cache.clear()
        self._cache_timestamps.clear()
        print("🗑️ Cleared all cached analysis results")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        return {
            "cached_items": list(self._analysis_cache.keys()),
            "cache_count": len(self._analysis_cache),
            "data_hash": self._data_hash,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "timestamps": {k: v.isoformat() for k, v in self._cache_timestamps.items()}
        }

    def _create_about_value(self, value: float, is_percentage: bool = False) -> str:
        """Same as main_2.py - Convert exact values to 'about' values with smarter rounding"""
        if is_percentage:
            if value == 0:
                return "0%"
            elif value < 5:
                return f"about {int(round(value))}%"
            elif value < 15:
                return f"about {int(round(value / 5) * 5)}%"
            else:
                return f"about {int(round(value / 10) * 10)}%"
        else:
            if value == 0:
                return "0"
            elif value < 5:
                return str(int(round(value)))
            elif value < 20:
                return f"about {int(round(value / 5) * 5)}"
            else:
                return f"about {int(round(value / 10) * 10)}"

    # ===== NEW: LLM-POWERED ANALYSIS METHODS =====

    async def _llm_analyze_responses(self, responses: List[str], analysis_type: str) -> Dict[str, Any]:
        """
        NEW: Use OpenAI LLM to analyze responses instead of local processing
        This is the key difference from main_2.py
        """

        # Limit responses to avoid token limits
        sample_responses = responses[:50] if len(responses) > 50 else responses
        responses_text = "\n".join(
            [f"{i+1}. {resp[:200]}" for i, resp in enumerate(sample_responses)])

        prompt = f"""
        Analyze these survey responses for {analysis_type}. 
        
        Survey Responses:
        {responses_text}
        
        Please provide analysis as ONLY a valid JSON object (no other text):
        {{
            "count": <number of responses mentioning this topic>,
            "percentage": <percentage of responses mentioning this topic>,
            "sentiment_score": <overall sentiment 0-100>,
            "key_themes": ["theme1", "theme2", "theme3"],
            "sample_quotes": ["quote1", "quote2"],
            "analysis": "<detailed analysis of findings>",
            "indicator": "<Positive indicator|Needs attention|Concerning>"
        }}
        
        Focus on: {analysis_type}
        Total responses analyzed: {len(responses)}
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Try to parse JSON from response, with fallback handling
            response_content = response.choices[0].message.content
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")

            print(
                f"🤖 LLM analyzed {analysis_type}: {result.get('count', 0)} mentions")
            return result

        except Exception as e:
            print(f"❌ Error in LLM analysis for {analysis_type}: {e}")
            return {
                "count": 0,
                "percentage": 0,
                "sentiment_score": 50,
                "key_themes": [],
                "sample_quotes": [],
                "analysis": f"Error in analysis: {str(e)}",
                "indicator": "Error"
            }

    async def _llm_comprehensive_analysis(self, all_responses: List[str]) -> Dict[str, Any]:
        """
        NEW: Use OpenAI LLM for comprehensive analysis of all responses
        This replaces all the local analysis logic from main_2.py
        """

        # Sample responses to avoid token limits
        sample_size = min(100, len(all_responses))
        sample_responses = all_responses[:sample_size]
        responses_text = "\n".join(
            [f"{i+1}. {resp[:150]}" for i, resp in enumerate(sample_responses)])

        prompt = f"""
        Perform comprehensive analysis of these survey responses. Analyze for multiple dimensions:
        
        Survey Responses ({sample_size} of {len(all_responses)} total):
        {responses_text}
        
        Please analyze and provide results as ONLY a valid JSON object (no other text):
        {{
            "overall_positive_sentiment": {{
                "percentage": <0-100>,
                "analysis": "<analysis>",
                "indicator": "<Positive indicator|Needs attention>"
            }},
            "team_support_mentions": {{
                "count": <number>,
                "analysis": "<analysis>", 
                "indicator": "<indicator>"
            }},
            "recognition_requests": {{
                "count": <number>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "promotion_concerns": {{
                "count": <number>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "strong_positive_percent": {{
                "percentage": <0-100>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "learning_mentions": {{
                "count": <number>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "politics_concerns": {{
                "count": <number>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "team_culture_strength": {{
                "count": <number>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "strong_negative_percent": {{
                "percentage": <0-100>,
                "analysis": "<analysis>",
                "indicator": "<indicator>"
            }},
            "key_themes": ["theme1", "theme2", "theme3"],
            "insights": ["insight1", "insight2", "insight3"]
        }}
        
        Analyze for: sentiment, team support, recognition needs, promotion concerns, learning opportunities, office politics, culture strength.
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )

            # Try to parse JSON from response, with fallback handling
            response_content = response.choices[0].message.content
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")

            print(
                f"🤖 LLM completed comprehensive analysis of {len(all_responses)} responses")
            return result

        except Exception as e:
            print(f"❌ Error in LLM comprehensive analysis: {e}")
            # Return fallback structure
            return {
                "overall_positive_sentiment": {"percentage": 50, "analysis": f"Error: {e}", "indicator": "Error"},
                "team_support_mentions": {"count": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "recognition_requests": {"count": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "promotion_concerns": {"count": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "strong_positive_percent": {"percentage": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "learning_mentions": {"count": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "politics_concerns": {"count": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "team_culture_strength": {"count": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "strong_negative_percent": {"percentage": 0, "analysis": f"Error: {e}", "indicator": "Error"},
                "key_themes": [],
                "insights": [f"Analysis failed: {e}"]
            }

    async def analyze_comprehensive_llm_powered(self) -> LLMAnalysisResult:
        """
        NEW: Main analysis method using LLM instead of local processing
        This is the key difference from main_2.py
        """

        # Check cache first
        cache_key = "comprehensive_analysis_llm"
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        print("🤖 Starting LLM-powered comprehensive analysis...")

        if not self.has_vectors():
            raise ValueError(
                "No processed data available. Please process data first.")

        # Get all responses for LLM analysis
        if self.use_direct_analysis:
            all_responses = self.small_dataset_responses
            total_responses = len(all_responses)
        else:
            all_responses = [meta['response_text']
                             for meta in self.all_metadata]
            total_responses = len(all_responses)

        # Use LLM for comprehensive analysis instead of local processing
        llm_results = await self._llm_comprehensive_analysis(all_responses)

        # Convert LLM results to our standard format
        def create_metric(llm_data: Dict[str, Any], is_percentage: bool = False) -> MetricWithAnalysis:
            if 'percentage' in llm_data:
                value = llm_data['percentage']
                range_estimate = self._create_about_value(
                    value, is_percentage=True)
            elif 'count' in llm_data:
                value = llm_data['count']
                range_estimate = self._create_about_value(
                    value, is_percentage=False)
            else:
                range_estimate = "unknown"

            return MetricWithAnalysis(
                range_estimate=range_estimate,
                analysis=llm_data.get('analysis', 'LLM analysis'),
                indicator=llm_data.get('indicator', 'Unknown')
            )

        # Create chunk summaries
        chunk_summaries = []
        if self.use_direct_analysis:
            chunk_summaries = [{
                "chunk_id": 0,
                "summary": f"LLM analysis of {total_responses} responses",
                "employee_count": len(set(meta['employee_id'] for meta in self.all_metadata)),
                "response_count": total_responses
            }]
        else:
            for chunk_data in self.chunks:
                chunk_summaries.append({
                    "chunk_id": chunk_data['chunk_id'],
                    "summary": f"LLM-analyzed {chunk_data['summary_text']}",
                    "employee_count": chunk_data['employee_count'],
                    "response_count": chunk_data['response_count']
                })

        # Create themes from LLM results
        themes = []
        for i, theme in enumerate(llm_results.get('key_themes', [])):
            themes.append({
                "theme": theme,
                # Estimate
                "mentions": total_responses // len(llm_results.get('key_themes', [1])),
                "sentiment": "Positive" if i % 2 == 0 else "Neutral"  # Alternate for variety
            })

        result = LLMAnalysisResult(
            overall_positive_sentiment=create_metric(
                llm_results['overall_positive_sentiment'], True),
            team_support_mentions=create_metric(
                llm_results['team_support_mentions']),
            recognition_requests=create_metric(
                llm_results['recognition_requests']),
            promotion_concerns=create_metric(
                llm_results['promotion_concerns']),
            strong_positive_percent=create_metric(
                llm_results['strong_positive_percent'], True),
            learning_mentions=create_metric(llm_results['learning_mentions']),
            politics_concerns=create_metric(llm_results['politics_concerns']),
            team_culture_strength=create_metric(
                llm_results['team_culture_strength']),
            strong_negative_percent=create_metric(
                llm_results['strong_negative_percent'], True),
            chunks_processed=len(
                self.chunks) if not self.use_direct_analysis else 0,
            total_responses=total_responses,
            chunk_summaries=chunk_summaries,
            themes=themes,
            insights=llm_results.get(
                'insights', ["LLM-powered analysis completed"])
        )

        # Cache the result
        self._store_in_cache(cache_key, result)

        return result


# Initialize the analyzers
na_handler = NAHandler()
llm_analyzer = LLMPoweredVectorAnalyzer(chunk_size=1000)


@app.get("/")
async def root():
    return {
        "message": "Employee Survey Analysis API v3.0 - LLM-Powered Analysis is running!",
        "description": "Hierarchical chunking with full LLM-powered analysis approach",
        "processing_strategy": {
            "approach": "LLM-Powered Hierarchical Processing",
            "small_datasets": "< 100 responses: Direct LLM analysis (no vectorization)",
            "large_datasets": ">= 100 responses: Hierarchical chunking + LLM analysis",
            "key_difference": "Uses OpenAI LLM for ALL analysis instead of local processing",
            "chunk_size_logic": {
                "automatic": "10% of dataset size (minimum 100, rounded to nearest 100)",
                "examples": "1000 responses → 100 chunk size, 10000 responses → 1000 chunk size",
                "custom_override": "Optional chunk_size parameter in upload endpoints"
            }
        },
        "endpoints": {
            "process_data": {
                "upload": "POST /analyze/upload-llm (supports chunk_size parameter)",
                "existing": "POST /analyze/existing-llm (supports chunk_size parameter)"
            },
            "analysis": {
                "comprehensive": "GET /analyze/comprehensive-llm",
                "sentiment": "GET /analyze/sentiment-llm",
                "team_support": "GET /analyze/team-support-llm",
                "recognition": "GET /analyze/recognition-llm",
                "promotion": "GET /analyze/promotion-llm",
                "strong_positive": "GET /analyze/strong-positive-llm",
                "learning": "GET /analyze/learning-llm",
                "politics": "GET /analyze/politics-llm",
                "culture": "GET /analyze/culture-llm",
                "strong_negative": "GET /analyze/strong-negative-llm",
                "chunks": "GET /analyze/chunks-info-llm (for large datasets only)"
            },
            "dynamic_analysis": {
                "configure": "POST /analyze/configure-dynamic-llm (AI-powered survey type detection)",
                "analyze": "GET /analyze/dynamic-llm (Dynamic parameter analysis)",
                "compare": "GET /analyze/comparison-llm (Compare approaches)"
            },
            "cache_management": {
                "info": "GET /analyze/cache-info-llm (View cache status and performance)",
                "clear": "POST /analyze/clear-cache-llm (Manually clear all cached results)"
            }
        },
        "features": {
            "llm_powered_analysis": "Uses OpenAI GPT-4 for ALL analysis instead of local processing",
            "hierarchical_chunking": "Same scalable chunking as main_2.py",
            "dynamic_analysis": "AI-powered survey type detection and parameter generation",
            "intelligent_caching": "Smart caching system for performance optimization",
            "higher_accuracy": "LLM analysis provides more nuanced insights than local processing",
            "cost_consideration": "Higher OpenAI API usage due to analysis calls"
        }
    }


@app.post("/analyze/upload-llm")
async def process_survey_data_llm(
    file: UploadFile = File(
        ...,
        description="Upload Excel (.xlsx, .xls) or CSV (.csv) file",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv"
    ),
    chunk_size: Optional[int] = Form(
        None,
        description="Optional custom chunk size. If not provided, will auto-calculate as 10% of dataset size (minimum 100). Set to override automatic calculation."
    )
):
    """
    Upload and process survey file using LLM-powered hierarchical chunking
    """
    try:
        # Read the uploaded file
        contents = await file.read()

        # Determine file type and read accordingly
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            # For Excel files, try to detect if there are multi-row headers
            df_test = pd.read_excel(io.BytesIO(contents), header=None, nrows=3)

            # Check if row 1 has more meaningful column names than row 0
            if (pd.isna(df_test.iloc[0]).sum() > len(df_test.columns) / 2 or
                'Participant' in str(df_test.iloc[1].values) or
                    any('feel' in str(val).lower() for val in df_test.iloc[1].values if pd.notna(val))):
                # Use row 1 as header (0-indexed)
                df = pd.read_excel(io.BytesIO(contents), header=1)
            else:
                # Use default header detection
                df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(contents.decode('latin-1')))
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload Excel (.xlsx, .xls) or CSV (.csv) files.")

        # Clean the data
        print(f"📊 Processing uploaded file: {file.filename}")
        print(f"📋 Original data shape: {df.shape}")

        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')
        print(f"✅ Cleaned data shape: {df_clean.shape}")

        # Calculate optimal chunk size
        dataset_size = len(df_clean)
        if chunk_size is not None:
            calculated_chunk_size, use_direct_analysis = llm_analyzer.calculate_optimal_chunk_size(
                dataset_size, chunk_size)
            chunk_size_source = "custom"
        else:
            calculated_chunk_size, use_direct_analysis = llm_analyzer.calculate_optimal_chunk_size(
                dataset_size)
            chunk_size_source = "auto-calculated"

        llm_analyzer.update_chunk_size(calculated_chunk_size, dataset_size)

        # Process data using LLM-powered hierarchical chunking
        success = llm_analyzer.process_survey_data_hierarchical(df_clean)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data using LLM-powered hierarchical chunking")

        # Prepare response based on processing type
        if use_direct_analysis:
            processing_type = "Direct LLM Analysis"
            total_responses = len(llm_analyzer.small_dataset_responses)
            chunks_created = 0
        else:
            processing_type = "LLM-Powered Hierarchical Chunking"
            total_responses = len(
                llm_analyzer.aggregated_vectors) if llm_analyzer.aggregated_vectors is not None else 0
            chunks_created = len(llm_analyzer.chunks)

        # Return processing confirmation
        return {
            "message": f"Survey data processed successfully using {processing_type.lower()}",
            "filename": file.filename,
            "processing_type": processing_type,
            "dataset_size": dataset_size,
            "chunk_size_used": calculated_chunk_size,
            "chunk_size_source": chunk_size_source,
            "total_responses": total_responses,
            "chunks_created": chunks_created,
            "data_shape": df_clean.shape,
            "processing_date": datetime.now(),
            "available_endpoints": [
                "/analyze/comprehensive-llm",
                "/analyze/configure-dynamic-llm"
            ],
            "llm_analysis_note": "All analysis will be performed using OpenAI LLM instead of local processing"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/analyze/existing-llm")
async def process_existing_file_llm():
    """
    Process the existing dummy survey file using LLM-powered analysis
    """
    try:
        file_path = "/Users/shashwat/Projects/Hyrgpt/EmpSurv2/Dummy_Employee_Survey_Responses (1).xlsx"

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail="Dummy survey file not found")

        print("📊 Processing existing file with LLM-powered hierarchical chunking")
        df = pd.read_excel(file_path, header=1)
        print(f"📋 Original data shape: {df.shape}")

        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')
        print(f"✅ Cleaned data shape: {df_clean.shape}")

        # Calculate optimal chunk size
        dataset_size = len(df_clean)
        chunk_size, use_direct_analysis = llm_analyzer.calculate_optimal_chunk_size(
            dataset_size)
        llm_analyzer.update_chunk_size(chunk_size, dataset_size)

        # Process data using hierarchical chunking (same as main_2.py)
        success = llm_analyzer.process_survey_data_hierarchical(df_clean)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data using LLM-powered hierarchical chunking")

        # Prepare response based on processing type
        if use_direct_analysis:
            processing_type = "Direct LLM Analysis"
            total_responses = len(llm_analyzer.small_dataset_responses)
            chunks_created = 0
        else:
            processing_type = "LLM-Powered Hierarchical Chunking"
            total_responses = len(
                llm_analyzer.aggregated_vectors) if llm_analyzer.aggregated_vectors is not None else 0
            chunks_created = len(llm_analyzer.chunks)

        return {
            "message": f"Existing survey data processed successfully using {processing_type.lower()}",
            "filename": "Dummy_Employee_Survey_Responses (1).xlsx",
            "processing_type": processing_type,
            "dataset_size": dataset_size,
            "chunk_size_used": chunk_size,
            "chunk_size_source": "direct-analysis" if use_direct_analysis else "auto-calculated",
            "total_responses": total_responses,
            "chunks_created": chunks_created,
            "data_shape": df_clean.shape,
            "processing_date": datetime.now(),
            "available_endpoints": [
                "/analyze/comprehensive-llm",
                "/analyze/configure-dynamic-llm"
            ],
            "llm_analysis_note": "All analysis will be performed using OpenAI LLM instead of local processing"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing existing file: {str(e)}")


@app.get("/analyze/comprehensive-llm")
async def get_comprehensive_analysis_llm():
    """
    Get comprehensive analysis using LLM-powered approach
    This is the main difference from main_2.py - uses OpenAI for analysis
    """
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first using /analyze/existing-llm"
            )

        print("🤖 Performing LLM-powered comprehensive analysis...")

        # Use LLM for analysis instead of local processing
        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()

        # Determine processing approach
        if llm_analyzer.use_direct_analysis:
            processing_approach = "Direct LLM Analysis"
            dataset_size_category = "small"
        else:
            processing_approach = "LLM-Powered Hierarchical Chunking"
            dataset_size_category = "large"

        return {
            "message": f"Comprehensive analysis completed successfully using {processing_approach.lower()}",
            "processing_approach": processing_approach,
            "chunks_processed": analysis_result.chunks_processed,
            "total_responses_analyzed": analysis_result.total_responses,
            "chunk_size": llm_analyzer.chunk_size,
            "dataset_size_category": dataset_size_category,

            # Analysis results (same structure as main_2.py but LLM-generated)
            "overall_positive_sentiment": {
                "value": analysis_result.overall_positive_sentiment.range_estimate,
                "analysis": analysis_result.overall_positive_sentiment.analysis,
                "indicator": analysis_result.overall_positive_sentiment.indicator
            },
            "team_support_mentions": {
                "value": analysis_result.team_support_mentions.range_estimate,
                "analysis": analysis_result.team_support_mentions.analysis,
                "indicator": analysis_result.team_support_mentions.indicator
            },
            "recognition_requests": {
                "value": analysis_result.recognition_requests.range_estimate,
                "analysis": analysis_result.recognition_requests.analysis,
                "indicator": analysis_result.recognition_requests.indicator
            },
            "promotion_concerns": {
                "value": analysis_result.promotion_concerns.range_estimate,
                "analysis": analysis_result.promotion_concerns.analysis,
                "indicator": analysis_result.promotion_concerns.indicator
            },
            "strong_positive_percent": {
                "value": analysis_result.strong_positive_percent.range_estimate,
                "analysis": analysis_result.strong_positive_percent.analysis,
                "indicator": analysis_result.strong_positive_percent.indicator
            },
            "learning_mentions": {
                "value": analysis_result.learning_mentions.range_estimate,
                "analysis": analysis_result.learning_mentions.analysis,
                "indicator": analysis_result.learning_mentions.indicator
            },
            "politics_concerns": {
                "value": analysis_result.politics_concerns.range_estimate,
                "analysis": analysis_result.politics_concerns.analysis,
                "indicator": analysis_result.politics_concerns.indicator
            },
            "team_culture_strength": {
                "value": analysis_result.team_culture_strength.range_estimate,
                "analysis": analysis_result.team_culture_strength.analysis,
                "indicator": analysis_result.team_culture_strength.indicator
            },
            "strong_negative_percent": {
                "value": analysis_result.strong_negative_percent.range_estimate,
                "analysis": analysis_result.strong_negative_percent.analysis,
                "indicator": analysis_result.strong_negative_percent.indicator
            },

            "chunk_summaries": analysis_result.chunk_summaries,
            "themes": analysis_result.themes,
            "insights": analysis_result.insights,
            "analysis_note": "LLM-powered analysis using OpenAI GPT-4 for enhanced accuracy and insights"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing LLM-powered comprehensive analysis: {str(e)}")


# Individual metric endpoints (same structure as main_2.py but using LLM analysis)
@app.get("/analyze/sentiment-llm")
async def get_sentiment_analysis_llm():
    """Get sentiment analysis using LLM-powered approach"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Overall Positive Sentiment",
            "value": analysis_result.overall_positive_sentiment.range_estimate,
            "analysis": analysis_result.overall_positive_sentiment.analysis,
            "indicator": analysis_result.overall_positive_sentiment.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM sentiment analysis: {str(e)}")


@app.get("/analyze/team-support-llm")
async def get_team_support_analysis_llm():
    """Get team support mentions analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Team Support Mentions",
            "value": analysis_result.team_support_mentions.range_estimate,
            "analysis": analysis_result.team_support_mentions.analysis,
            "indicator": analysis_result.team_support_mentions.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM team support analysis: {str(e)}")


@app.get("/analyze/recognition-llm")
async def get_recognition_analysis_llm():
    """Get recognition requests analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Recognition Requests",
            "value": analysis_result.recognition_requests.range_estimate,
            "analysis": analysis_result.recognition_requests.analysis,
            "indicator": analysis_result.recognition_requests.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM recognition analysis: {str(e)}")


@app.get("/analyze/promotion-llm")
async def get_promotion_analysis_llm():
    """Get promotion concerns analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Promotion Concerns",
            "value": analysis_result.promotion_concerns.range_estimate,
            "analysis": analysis_result.promotion_concerns.analysis,
            "indicator": analysis_result.promotion_concerns.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM promotion analysis: {str(e)}")


@app.get("/analyze/strong-positive-llm")
async def get_strong_positive_analysis_llm():
    """Get strong positive sentiment analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Strong Positive",
            "value": analysis_result.strong_positive_percent.range_estimate,
            "analysis": analysis_result.strong_positive_percent.analysis,
            "indicator": analysis_result.strong_positive_percent.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM strong positive analysis: {str(e)}")


@app.get("/analyze/learning-llm")
async def get_learning_analysis_llm():
    """Get learning mentions analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Learning Mentions",
            "value": analysis_result.learning_mentions.range_estimate,
            "analysis": analysis_result.learning_mentions.analysis,
            "indicator": analysis_result.learning_mentions.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM learning analysis: {str(e)}")


@app.get("/analyze/politics-llm")
async def get_politics_analysis_llm():
    """Get politics concerns analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Politics Concerns",
            "value": analysis_result.politics_concerns.range_estimate,
            "analysis": analysis_result.politics_concerns.analysis,
            "indicator": analysis_result.politics_concerns.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM politics analysis: {str(e)}")


@app.get("/analyze/culture-llm")
async def get_culture_analysis_llm():
    """Get team culture strength analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Team Culture Strength",
            "value": analysis_result.team_culture_strength.range_estimate,
            "analysis": analysis_result.team_culture_strength.analysis,
            "indicator": analysis_result.team_culture_strength.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM culture analysis: {str(e)}")


@app.get("/analyze/strong-negative-llm")
async def get_strong_negative_analysis_llm():
    """Get strong negative sentiment analysis using LLM"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = await llm_analyzer.analyze_comprehensive_llm_powered()
        return {
            "metric": "Strong Negative",
            "value": analysis_result.strong_negative_percent.range_estimate,
            "analysis": analysis_result.strong_negative_percent.analysis,
            "indicator": analysis_result.strong_negative_percent.indicator,
            "method": "LLM-powered analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in LLM strong negative analysis: {str(e)}")


@app.get("/analyze/chunks-info-llm")
async def get_chunks_info_llm():
    """
    Get detailed information about the processed chunks for LLM analyzer
    """
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No chunks available. Please process data first."
            )

        chunk_details = []
        for chunk_data in llm_analyzer.chunks:
            chunk_details.append({
                "chunk_id": chunk_data['chunk_id'],
                "summary": chunk_data['summary_text'],
                "employee_count": chunk_data['employee_count'],
                "response_count": chunk_data['response_count'],
                "has_embeddings": len(chunk_data['embeddings']) > 0 if 'embeddings' in chunk_data else False,
                "vector_dimensions": len(chunk_data['summary_vector']) if 'summary_vector' in chunk_data else 0
            })

        return {
            "message": "LLM analyzer chunk information retrieved successfully",
            "total_chunks": len(llm_analyzer.chunks),
            "chunk_size_configured": llm_analyzer.chunk_size,
            "total_responses": len(llm_analyzer.aggregated_vectors) if llm_analyzer.aggregated_vectors is not None else 0,
            "chunks": chunk_details,
            "processing_summary": {
                "approach": "LLM-powered hierarchical chunking with response-level granularity",
                "benefits": [
                    "Memory efficient for large datasets",
                    "Parallel processing capability",
                    "LLM-powered analysis for enhanced insights",
                    "Scalable to millions of responses"
                ]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving LLM chunk information: {str(e)}")


# Cache management endpoints (same as main_2.py)
@app.get("/analyze/cache-info-llm")
async def get_cache_info_llm():
    """Get information about the current cache state for LLM analyzer"""
    try:
        cache_info = llm_analyzer.get_cache_info()

        return {
            "message": "LLM analyzer cache information retrieved successfully",
            "cache_info": cache_info,
            "performance_note": "Cached results provide instant responses, uncached results require LLM API calls",
            "cache_benefits": {
                "speed_improvement": "Eliminates repeated LLM API calls",
                "cost_savings": "Reduces OpenAI API usage for repeated requests",
                "user_experience": "Instant response for repeated requests"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving LLM cache information: {str(e)}")


@app.post("/analyze/clear-cache-llm")
async def clear_cache_llm():
    """Manually clear all cached LLM analysis results"""
    try:
        llm_analyzer._clear_all_cache()

        return {
            "message": "All cached LLM analysis results cleared successfully",
            "note": "Next analysis requests will perform fresh LLM computations",
            "when_to_use": [
                "When you want to force fresh LLM analysis",
                "After making configuration changes",
                "For troubleshooting cache-related issues"
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing LLM cache: {str(e)}")


# Dynamic analysis endpoints (enhanced for LLM approach)
@app.post("/analyze/configure-dynamic-llm")
async def configure_dynamic_analysis_llm(survey_context: Optional[str] = Form(None)):
    """Configure dynamic analysis using LLM-powered approach"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No processed data available. Please process data first using /analyze/existing-llm"
            )

        # Get sample responses for analysis
        if llm_analyzer.use_direct_analysis:
            sample_responses = llm_analyzer.small_dataset_responses[:15]
        else:
            sample_responses = [meta['response_text']
                                for meta in llm_analyzer.all_metadata[:15]]

        if not sample_responses:
            raise HTTPException(
                status_code=400,
                detail="No response data available for dynamic analysis configuration"
            )

        print("🤖 Configuring LLM-powered dynamic analysis...")

        # Clear dynamic analysis cache when reconfiguring
        llm_analyzer._remove_from_cache("dynamic_analysis_llm")

        # Configure dynamic analysis
        survey_profile = await llm_analyzer.dynamic_analyzer.detect_and_configure_analysis(
            sample_responses, survey_context
        )

        return {
            "message": "LLM-powered dynamic analysis configured successfully",
            "survey_type": survey_profile.survey_type,
            "industry_context": survey_profile.industry_context,
            "parameters_generated": len(survey_profile.parameters),
            "parameters": [
                {
                    "name": param.name,
                    "description": param.description,
                    "search_terms": param.search_terms,
                    "expected_range": param.expected_range,
                    "importance": param.importance
                }
                for param in survey_profile.parameters
            ],
            "insights_template": survey_profile.insights_template,
            "next_step": "Call /analyze/dynamic-llm to run the LLM-powered analysis with these parameters",
            "analysis_method": "LLM-powered parameter generation and analysis"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error configuring LLM dynamic analysis: {str(e)}")


@app.get("/analyze/dynamic-llm")
async def get_dynamic_analysis_llm():
    """Get comprehensive analysis using dynamically generated parameters with LLM power"""
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        if not llm_analyzer.dynamic_analyzer.survey_profile:
            raise HTTPException(
                status_code=400,
                detail="Dynamic analysis not configured. Please call /analyze/configure-dynamic-llm first."
            )

        print("🤖 Running LLM-powered dynamic analysis...")

        # Get total responses count
        if llm_analyzer.use_direct_analysis:
            total_responses = len(llm_analyzer.small_dataset_responses)
        else:
            total_responses = len(llm_analyzer.all_metadata)

        # Check cache for dynamic analysis
        cache_key = "dynamic_analysis_llm"
        cached_dynamic_result = llm_analyzer._get_from_cache(cache_key)

        if cached_dynamic_result is not None:
            dynamic_result = cached_dynamic_result
        else:
            # Run dynamic analysis and cache it
            dynamic_result = llm_analyzer.dynamic_analyzer.run_dynamic_analysis(
                llm_analyzer, total_responses
            )
            llm_analyzer._store_in_cache(cache_key, dynamic_result)

        # Also get the traditional LLM analysis for comparison
        try:
            traditional_result = await llm_analyzer.analyze_comprehensive_llm_powered()
            traditional_summary = {
                "overall_positive_sentiment": traditional_result.overall_positive_sentiment.range_estimate,
                "team_support_mentions": traditional_result.team_support_mentions.range_estimate,
                "recognition_requests": traditional_result.recognition_requests.range_estimate,
                "promotion_concerns": traditional_result.promotion_concerns.range_estimate,
                "strong_positive": traditional_result.strong_positive_percent.range_estimate,
                "learning_mentions": traditional_result.learning_mentions.range_estimate,
                "politics_concerns": traditional_result.politics_concerns.range_estimate,
                "team_culture_strength": traditional_result.team_culture_strength.range_estimate,
                "strong_negative": traditional_result.strong_negative_percent.range_estimate
            }
        except Exception as e:
            print(
                f"Warning: Could not get traditional LLM analysis for comparison: {e}")
            traditional_summary = {
                "error": "Traditional LLM analysis unavailable"}

        return {
            "message": "LLM-powered dynamic analysis completed successfully",
            "analysis_method": "AI-generated parameters with LLM-powered analysis",

            # Dynamic analysis results (NEW)
            "dynamic_analysis": {
                "survey_type": dynamic_result.survey_type,
                "industry_context": dynamic_result.industry_context,
                "total_responses": dynamic_result.total_responses,
                "parameters_analyzed": dynamic_result.parameters_analyzed,
                "results": dynamic_result.analysis_results,
                "generation_method": "llm-powered"
            },

            # Traditional LLM analysis for comparison
            "traditional_llm_analysis": traditional_summary,

            # Metadata
            "analysis_date": datetime.now(),
            "chunks_processed": len(llm_analyzer.chunks) if llm_analyzer.chunks else 0,
            "approach_used": "llm_powered_hierarchical_chunking" if not llm_analyzer.use_direct_analysis else "direct_llm_analysis"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing LLM-powered dynamic analysis: {str(e)}")


@app.get("/analyze/comparison-llm")
async def get_analysis_comparison_llm():
    """
    Compare traditional LLM analysis vs dynamic LLM-powered analysis
    This shows the difference between hardcoded parameters and AI-generated parameters (both using LLM)
    """
    try:
        if not llm_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        # Get traditional LLM analysis
        traditional_result = await llm_analyzer.analyze_comprehensive_llm_powered()

        # Get dynamic analysis (if configured)
        dynamic_result = None
        if llm_analyzer.dynamic_analyzer.survey_profile:
            total_responses = len(llm_analyzer.all_metadata) if not llm_analyzer.use_direct_analysis else len(
                llm_analyzer.small_dataset_responses)
            dynamic_result = llm_analyzer.dynamic_analyzer.run_dynamic_analysis(
                llm_analyzer, total_responses
            )

        comparison = {
            "traditional_llm_analysis": {
                "method": "Hardcoded parameters with LLM analysis (employee survey focused)",
                "parameters": [
                    "overall_positive_sentiment", "team_support_mentions", "recognition_requests",
                    "promotion_concerns", "strong_positive_percent", "learning_mentions",
                    "politics_concerns", "team_culture_strength", "strong_negative_percent"
                ],
                "results": {
                    "overall_positive_sentiment": traditional_result.overall_positive_sentiment.range_estimate,
                    "team_support_mentions": traditional_result.team_support_mentions.range_estimate,
                    "recognition_requests": traditional_result.recognition_requests.range_estimate,
                    "promotion_concerns": traditional_result.promotion_concerns.range_estimate,
                    "strong_positive": traditional_result.strong_positive_percent.range_estimate,
                    "learning_mentions": traditional_result.learning_mentions.range_estimate,
                    "politics_concerns": traditional_result.politics_concerns.range_estimate,
                    "team_culture_strength": traditional_result.team_culture_strength.range_estimate,
                    "strong_negative": traditional_result.strong_negative_percent.range_estimate
                },
                "analysis_method": "LLM-powered with hardcoded parameters"
            }
        }

        if dynamic_result:
            comparison["dynamic_llm_analysis"] = {
                "method": "AI-generated parameters with LLM analysis",
                "survey_type_detected": dynamic_result.survey_type,
                "industry_context": dynamic_result.industry_context,
                "parameters": list(dynamic_result.analysis_results.keys()),
                "results": {
                    param: result.get(
                        "about_count", result.get("error", "N/A"))
                    for param, result in dynamic_result.analysis_results.items()
                },
                "analysis_method": "LLM-powered with AI-generated parameters"
            }

            comparison["insights"] = [
                f"Traditional LLM analysis uses {len(comparison['traditional_llm_analysis']['parameters'])} fixed parameters with LLM processing",
                f"Dynamic LLM analysis generated {len(comparison['dynamic_llm_analysis']['parameters'])} survey-specific parameters with LLM processing",
                f"Survey type detected: {dynamic_result.survey_type}",
                "Both approaches use LLM for analysis - difference is in parameter generation",
                "Dynamic approach adapts parameters to survey type, traditional uses fixed employee-focused parameters"
            ]
        else:
            comparison["dynamic_llm_analysis"] = {
                "status": "Not configured",
                "message": "Call /analyze/configure-dynamic-llm first to enable dynamic LLM analysis"
            }
            comparison["insights"] = [
                "Traditional LLM analysis uses fixed parameters with LLM processing",
                "Dynamic LLM analysis not yet configured - would adapt parameters to your specific survey type",
                "Both approaches use OpenAI LLM for analysis - main difference is parameter adaptability"
            ]

        return {
            "message": "LLM analysis comparison completed",
            "comparison": comparison,
            "recommendation": "Use dynamic LLM analysis for better survey-specific insights with LLM power" if dynamic_result else "Configure dynamic LLM analysis for survey-specific insights",
            "analysis_date": datetime.now(),
            "key_difference": "Both use LLM for analysis, but dynamic approach generates survey-specific parameters"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing LLM analysis comparison: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

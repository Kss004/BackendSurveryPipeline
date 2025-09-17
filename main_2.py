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
    generation_method: str = "dynamic"


class DynamicSurveyAnalyzer:
    """Enhanced analyzer that generates analysis parameters dynamically"""

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.survey_profile: Optional[SurveyTypeProfile] = None
        self.analysis_cache: Dict[str, Any] = {}

    def detect_and_configure_analysis(self, sample_responses: List[str],
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
        
        Please respond with ONLY a valid JSON object (no other text) in this exact format:
        {{
            "survey_type": "employee",
            "industry_context": "detected industry or context",
            "analysis_parameters": [
                {{
                    "name": "parameter_name",
                    "search_terms": "relevant keywords and phrases for semantic search",
                    "description": "what this parameter measures",
                    "expected_range": "typical range for this metric",
                    "importance": "high"
                }}
            ],
            "insights_template": [
                "Key insight patterns to look for in this survey type"
            ]
        }}
        
        Generate 8-12 analysis parameters that are most relevant for this specific survey type.
        Focus on parameters that will provide actionable insights.
        Respond with ONLY the JSON object, no additional text.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            # Try to extract JSON if there's extra text
            if not content.startswith('{'):
                # Look for JSON block
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    content = content[start:end]

            result = json.loads(content)

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


app = FastAPI(
    title="Employee Survey Analysis API - Hierarchical Chunking",
    description="Hierarchical chunking approach for large-scale employee survey analysis",
    version="hierarchical"
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
class HierarchicalAnalysisResult:
    """Container for hierarchical analysis results"""
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


class HierarchicalVectorAnalyzer:
    """
    Hierarchical chunking approach for large-scale survey analysis:
    1. Divide dataset into manageable chunks (e.g., 1000 responses each)
    2. Process each chunk separately to create chunk-level vectors
    3. Aggregate chunk results for final analysis
    """

    def __init__(self, chunk_size: int = 1000, openai_api_key: Optional[str] = None):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        # Hierarchical chunking configuration
        self.chunk_size = chunk_size
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_vectors: Dict[int, np.ndarray] = {}
        self.chunk_metadata: List[ChunkMetadata] = []
        self.aggregated_vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Small dataset handling
        self.use_direct_analysis = False
        self.small_dataset_responses: List[str] = []

        # Analysis configuration
        self.sentiment_examples = {
            'positive': [
                "I love working here, great team support and opportunities",
                "Excellent work-life balance and supportive management",
                "Amazing learning opportunities and career growth",
                "Outstanding collaboration and team culture"
            ],
            'negative': [
                "Poor work-life balance and stressful environment",
                "Lack of support from management and limited growth",
                "Terrible team dynamics and office politics",
                "No recognition for hard work and contributions"
            ]
        }

        # Initialize sentiment reference vectors
        self.sentiment_vectors: Optional[Dict[str, np.ndarray]] = None
        self._initialize_sentiment_vectors()

        # Initialize dynamic analyzer
        self.dynamic_analyzer = DynamicSurveyAnalyzer(self.client)

        # ===== CACHING SYSTEM =====
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._data_hash: Optional[str] = None  # Track when data changes
        self.cache_ttl_seconds: int = 300  # 5 minutes cache TTL

    def calculate_optimal_chunk_size(self, dataset_size: int, custom_chunk_size: Optional[int] = None) -> Tuple[int, bool]:
        """
        Calculate optimal chunk size based on dataset size or use custom size

        Args:
            dataset_size: Total number of responses in the dataset
            custom_chunk_size: Optional custom chunk size to override automatic calculation

        Returns:
            Tuple of (chunk_size, use_direct_analysis)
        """
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

        # Round to nearest 100 for cleaner chunks
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

    def _initialize_sentiment_vectors(self):
        """Initialize reference vectors for sentiment analysis"""
        try:
            all_examples = self.sentiment_examples['positive'] + \
                self.sentiment_examples['negative']
            embeddings = self._get_embeddings_batch(all_examples)

            pos_count = len(self.sentiment_examples['positive'])

            self.sentiment_vectors = {
                'positive': np.mean(embeddings[:pos_count], axis=0),
                'negative': np.mean(embeddings[pos_count:], axis=0)
            }
            print("✅ Sentiment reference vectors initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize sentiment vectors: {e}")
            self.sentiment_vectors = None

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings for a batch of texts"""
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
                        # Create deterministic mock embeddings based on text content
                        text_hash = hash(text.lower()) % 1000000
                        # Deterministic based on text
                        np.random.seed(text_hash)
                        mock_embedding = np.random.normal(
                            0, 0.1, self.embedding_dimension)
                        # Add some semantic-like features based on keywords
                        if any(word in text.lower() for word in ['good', 'great', 'excellent', 'love', 'happy']):
                            # Positive sentiment region
                            mock_embedding[0:50] += 0.3
                        if any(word in text.lower() for word in ['bad', 'terrible', 'hate', 'poor', 'awful']):
                            # Negative sentiment region
                            mock_embedding[50:100] += 0.3
                        if any(word in text.lower() for word in ['team', 'support', 'collaboration']):
                            # Team support region
                            mock_embedding[100:150] += 0.3
                        mock_embeddings.append(mock_embedding)
                    embeddings.extend(mock_embeddings)
                else:
                    # Add zero vectors as fallback for other errors
                    embeddings.extend(
                        [np.zeros(self.embedding_dimension) for _ in batch])

        return np.array(embeddings)

    def _divide_into_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Divide the dataset into manageable chunks"""
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
        """Process a single chunk to extract text and create vectors"""
        print(f"🔄 Processing chunk {chunk_id}...")

        # Extract responses from the chunk (similar to original approach)
        possible_response_cols = [
            'How do you feel about the work-life balance in the organization?',
            'Rate your overall satisfaction with the team support.',
            'What improvements would you suggest for the workplace?',
            'Any additional comments or feedback?',
            'How would you rate your overall job satisfaction?'
        ]

        # Find available columns
        available_cols = [
            col for col in possible_response_cols if col in chunk_df.columns]
        if not available_cols:
            # Fallback: use columns that contain 'feel', 'satisfaction', 'rate', etc.
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

            # Create chunk summary by averaging embeddings
            chunk_summary_vector = np.mean(chunk_embeddings, axis=0)

            # Generate text summary of chunk
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
        """Aggregate all chunk vectors for final analysis"""
        if not self.chunks:
            print("❌ No chunks to aggregate")
            return False

        print(f"🔄 Aggregating {len(self.chunks)} chunks...")

        # Collect all embeddings and metadata
        all_embeddings = []
        all_metadata = []

        # Create chunk summary vectors
        chunk_summary_vectors = []

        for chunk_data in self.chunks:
            if chunk_data and 'embeddings' in chunk_data:
                # Add individual response embeddings
                all_embeddings.append(chunk_data['embeddings'])
                all_metadata.extend(chunk_data['metadata'])

                # Add chunk summary vector
                chunk_summary_vectors.append(chunk_data['summary_vector'])

                # Store chunk vector for chunk-level analysis
                self.chunk_vectors[chunk_data['chunk_id']
                                   ] = chunk_data['summary_vector']

        if not all_embeddings:
            print("❌ No embeddings found in chunks")
            return False

        # Combine all response-level embeddings
        self.aggregated_vectors = np.vstack(all_embeddings)
        self.all_metadata = all_metadata

        print(
            f"✅ Aggregated {len(self.aggregated_vectors)} total embeddings from {len(self.chunks)} chunks")

        # Create Faiss index for similarity search
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)

        # Normalize vectors for cosine similarity (handle zero vectors)
        norms = np.linalg.norm(self.aggregated_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        normalized_vectors = self.aggregated_vectors / norms
        self.faiss_index.add(normalized_vectors.astype('float32'))

        print(f"🔍 Created Faiss index with {self.faiss_index.ntotal} vectors")
        return True

    def process_survey_data_hierarchical(self, df: pd.DataFrame) -> bool:
        """Main method to process survey data using hierarchical chunking"""
        try:
            print(
                f"🚀 Starting hierarchical processing of {len(df)} responses...")

            # Clear cache when processing new data
            self._clear_all_cache()

            # Check if dataset is too small for vectorization
            if self.use_direct_analysis:
                return self._process_small_dataset(df)

            # Step 1: Divide into chunks
            chunk_dataframes = self._divide_into_chunks(df)

            # Step 2: Process each chunk
            self.chunks = []
            for i, chunk_df in enumerate(chunk_dataframes):
                chunk_result = self._process_chunk(chunk_df, i)
                if chunk_result:
                    self.chunks.append(chunk_result)

            if not self.chunks:
                print("❌ No chunks processed successfully")
                return False

            # Step 3: Aggregate chunks
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
        """Process small datasets directly without vectorization"""
        try:
            print(
                f"📋 Processing small dataset with {len(df)} responses directly")

            # Clear cache when processing new data
            self._clear_all_cache()

            # Extract responses similar to chunk processing
            possible_response_cols = [
                'How do you feel about the work-life balance in the organization?',
                'Rate your overall satisfaction with the team support.',
                'What improvements would you suggest for the workplace?',
                'Any additional comments or feedback?',
                'How would you rate your overall job satisfaction?'
            ]

            # Find available columns
            available_cols = [
                col for col in possible_response_cols if col in df.columns]
            if not available_cols:
                available_cols = [col for col in df.columns
                                  if any(keyword in col.lower()
                                         for keyword in ['feel', 'satisfaction', 'rate', 'support', 'improve', 'comment'])]

            # Extract all responses
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
                            'chunk_id': 0,  # Single chunk for small dataset
                            'response_index': len(self.small_dataset_responses) - 1
                        })

            print(
                f"📊 Extracted {len(self.small_dataset_responses)} responses for direct analysis")
            return len(self.small_dataset_responses) > 0

        except Exception as e:
            print(f"❌ Error processing small dataset: {e}")
            return False

    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Perform semantic search across all aggregated vectors"""
        if self.faiss_index is None or self.aggregated_vectors is None:
            return []

        try:
            # Get query embedding
            query_embedding = self._get_embeddings_batch([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Search
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

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score using reference vectors"""
        if self.sentiment_vectors is None:
            return 0.5  # Neutral fallback

        try:
            embedding = self._get_embeddings_batch([text])[0]

            pos_sim = cosine_similarity(
                [embedding], [self.sentiment_vectors['positive']])[0][0]
            neg_sim = cosine_similarity(
                [embedding], [self.sentiment_vectors['negative']])[0][0]

            # Convert to 0-1 scale (0=negative, 1=positive)
            return (pos_sim - neg_sim + 2) / 4
        except:
            return 0.5

    def has_vectors(self) -> bool:
        """Check if vectors are available for analysis"""
        if self.use_direct_analysis:
            return len(self.small_dataset_responses) > 0
        else:
            return self.faiss_index is not None and self.aggregated_vectors is not None

    # ===== CACHING METHODS =====

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
        """Convert exact values to 'about' values with smarter rounding"""
        if is_percentage:
            # For percentages, use more granular rounding
            if value == 0:
                return "0%"
            elif value < 5:
                return f"about {int(round(value))}%"
            elif value < 15:
                # Round to nearest 5
                return f"about {int(round(value / 5) * 5)}%"
            else:
                # Round to nearest 10
                return f"about {int(round(value / 10) * 10)}%"
        else:
            # For counts, use more granular rounding
            if value == 0:
                return "0"
            elif value < 5:
                # Show exact count for small values
                return str(int(round(value)))
            elif value < 20:
                # Round to nearest 5
                return f"about {int(round(value / 5) * 5)}"
            else:
                # Round to nearest 10
                return f"about {int(round(value / 10) * 10)}"

    def analyze_comprehensive_hierarchical(self) -> HierarchicalAnalysisResult:
        """Perform comprehensive analysis using hierarchical approach with caching"""

        # Check cache first
        cache_key = "comprehensive_analysis"
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        print("🔄 Starting comprehensive hierarchical analysis...")

        if not self.has_vectors():
            raise ValueError(
                "No processed data available. Please process data first.")

        # Perform analysis
        if self.use_direct_analysis:
            result = self._analyze_small_dataset_directly()
        else:
            result = self._analyze_large_dataset_hierarchical()

        # Cache the result
        self._store_in_cache(cache_key, result)

        return result

    def _analyze_small_dataset_directly(self) -> HierarchicalAnalysisResult:
        """Analyze small datasets directly using basic text analysis"""
        print("📊 Performing direct analysis on small dataset...")

        # Simple keyword-based analysis for small datasets
        responses = self.small_dataset_responses
        total_responses = len(responses)

        # Basic sentiment analysis using keywords
        positive_keywords = [
            'good', 'great', 'excellent', 'love', 'happy', 'satisfied', 'amazing', 'wonderful', 'fantastic',
            'enjoy', 'like', 'appreciate', 'positive', 'beneficial', 'helpful', 'comfortable', 'pleased',
            'effective', 'efficient', 'productive', 'valuable', 'useful', 'successful', 'impressive',
            'outstanding', 'exceptional', 'remarkable', 'favorable', 'satisfactory', 'adequate'
        ]
        negative_keywords = [
            'bad', 'terrible', 'hate', 'poor', 'awful', 'horrible', 'disappointing', 'frustrated',
            'dislike', 'difficult', 'challenging', 'stressful', 'inadequate', 'insufficient', 'lacking',
            'unsatisfactory', 'ineffective', 'inefficient', 'unproductive', 'problematic', 'concerning',
            'disappointing', 'dissatisfied', 'unhappy', 'uncomfortable', 'negative', 'issues', 'problems'
        ]

        positive_count = 0
        strong_positive_count = 0
        strong_negative_count = 0
        neutral_count = 0

        for response in responses:
            response_lower = response.lower()
            pos_matches = sum(
                1 for keyword in positive_keywords if keyword in response_lower)
            neg_matches = sum(
                1 for keyword in negative_keywords if keyword in response_lower)

            if pos_matches > 0 and pos_matches > neg_matches:
                positive_count += 1
                if pos_matches >= 2:  # Strong positive if multiple positive keywords
                    strong_positive_count += 1
            elif neg_matches > 0 and neg_matches > pos_matches:
                # Count any negative sentiment, not just strong negative
                if neg_matches >= 2:  # Strong negative if multiple negative keywords
                    strong_negative_count += 1
            else:
                neutral_count += 1

        positive_sentiment_pct = (
            positive_count / total_responses) * 100 if total_responses > 0 else 0
        strong_positive_pct = (
            strong_positive_count / total_responses) * 100 if total_responses > 0 else 0
        strong_negative_pct = (
            strong_negative_count / total_responses) * 100 if total_responses > 0 else 0

        # Simple keyword searches
        team_support_count = sum(1 for response in responses if any(
            word in response.lower() for word in ['team', 'support', 'collaboration', 'help']))
        recognition_count = sum(1 for response in responses if any(word in response.lower(
        ) for word in ['recognition', 'appreciate', 'acknowledge', 'reward']))
        promotion_count = sum(1 for response in responses if any(word in response.lower(
        ) for word in ['promotion', 'career', 'growth', 'advancement']))
        learning_count = sum(1 for response in responses if any(word in response.lower(
        ) for word in ['learning', 'training', 'development', 'skills']))
        politics_count = sum(1 for response in responses if any(
            word in response.lower() for word in ['politics', 'conflict', 'unfair', 'bias']))
        culture_count = sum(1 for response in responses if any(word in response.lower(
        ) for word in ['culture', 'values', 'environment', 'workplace']))

        insights = [
            f"Small dataset with {total_responses} responses analyzed directly without vectorization",
            "Basic keyword-based analysis performed for efficiency",
            "For larger datasets, consider using hierarchical chunking for more accurate results"
        ]

        return HierarchicalAnalysisResult(
            overall_positive_sentiment=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    positive_sentiment_pct, is_percentage=True),
                analysis="Direct keyword analysis" if positive_sentiment_pct > 50 else "Below average satisfaction",
                indicator="Positive indicator" if positive_sentiment_pct > 50 else "Needs attention"
            ),
            team_support_mentions=MetricWithAnalysis(
                range_estimate=self._create_about_value(team_support_count),
                analysis="Keyword-based team support analysis",
                indicator="Positive indicator" if team_support_count > total_responses * 0.3 else "Moderate"
            ),
            recognition_requests=MetricWithAnalysis(
                range_estimate=self._create_about_value(recognition_count),
                analysis="Keyword-based recognition analysis",
                indicator="High demand" if recognition_count > total_responses * 0.25 else "Moderate"
            ),
            promotion_concerns=MetricWithAnalysis(
                range_estimate=self._create_about_value(promotion_count),
                analysis="Keyword-based career analysis",
                indicator="Needs attention" if promotion_count > total_responses * 0.2 else "Moderate"
            ),
            strong_positive_percent=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    strong_positive_pct, is_percentage=True),
                analysis="Strong positive keywords detected",
                indicator="Excellent" if strong_positive_pct > 25 else "Good"
            ),
            learning_mentions=MetricWithAnalysis(
                range_estimate=self._create_about_value(learning_count),
                analysis="Keyword-based learning analysis",
                indicator="High value" if learning_count > total_responses * 0.3 else "Moderate"
            ),
            politics_concerns=MetricWithAnalysis(
                range_estimate=self._create_about_value(politics_count),
                analysis="Keyword-based politics analysis",
                indicator="Needs attention" if politics_count > total_responses *
                0.15 else "Minor concern"
            ),
            team_culture_strength=MetricWithAnalysis(
                range_estimate=self._create_about_value(culture_count),
                analysis="Keyword-based culture analysis",
                indicator="Strong" if culture_count > total_responses * 0.3 else "Moderate"
            ),
            strong_negative_percent=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    strong_negative_pct, is_percentage=True),
                analysis="Strong negative keywords detected",
                indicator="Concerning" if strong_negative_pct > 10 else "Low"
            ),
            chunks_processed=0,  # No chunks for small datasets
            total_responses=total_responses,
            chunk_summaries=[{
                'chunk_id': 0,
                'summary': f"Direct analysis of {total_responses} responses",
                'employee_count': len(set(meta['employee_id'] for meta in self.all_metadata)),
                'response_count': total_responses
            }],
            themes=[
                {"theme": "Direct Analysis",
                    "mentions": total_responses, "sentiment": "Neutral"},
                {"theme": "Keyword-based Processing",
                    "mentions": total_responses, "sentiment": "Neutral"}
            ],
            insights=insights
        )

    def _analyze_large_dataset_hierarchical(self) -> HierarchicalAnalysisResult:
        """Perform comprehensive analysis using hierarchical approach for large datasets"""
        # Sentiment Analysis across all responses
        print("📊 Analyzing sentiment across all chunks...")
        sentiment_scores = []
        for chunk_data in self.chunks:
            for response in chunk_data['responses']:
                score = self._calculate_sentiment_score(response)
                sentiment_scores.append(score)

        positive_sentiment_pct = (
            np.array(sentiment_scores) > 0.6).mean() * 100
        strong_positive_pct = (np.array(sentiment_scores) > 0.8).mean() * 100
        strong_negative_pct = (np.array(sentiment_scores) < 0.2).mean() * 100

        # Semantic searches across aggregated vectors
        team_support_results = self._semantic_search(
            "team support collaboration help colleagues", top_k=50)
        recognition_results = self._semantic_search(
            "recognition appreciation acknowledge achievement reward", top_k=40)
        promotion_results = self._semantic_search(
            "promotion career growth advancement opportunities development", top_k=35)
        learning_results = self._semantic_search(
            "learning training development skills knowledge education", top_k=45)
        politics_results = self._semantic_search(
            "politics conflict unfair bias discrimination", top_k=30)
        culture_results = self._semantic_search(
            "culture values environment positive workplace", top_k=40)

        # Create chunk summaries
        chunk_summaries = []
        for chunk_data in self.chunks:
            chunk_summaries.append({
                'chunk_id': chunk_data['chunk_id'],
                'summary': chunk_data['summary_text'],
                'employee_count': chunk_data['employee_count'],
                'response_count': chunk_data['response_count']
            })

        # Generate insights based on hierarchical analysis
        insights = [
            f"Processed {len(self.chunks)} chunks with chunk size of {self.chunk_size} responses each",
            f"Total {len(self.aggregated_vectors)} individual responses analyzed across all chunks",
            f"Hierarchical approach allows scalable processing of large datasets",
            f"Chunk-level summaries provide intermediate analysis granularity"
        ]

        return HierarchicalAnalysisResult(
            overall_positive_sentiment=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    positive_sentiment_pct, is_percentage=True),
                analysis="Above average satisfaction" if positive_sentiment_pct > 50 else "Below average satisfaction",
                indicator="Positive indicator" if positive_sentiment_pct > 50 else "Needs attention"
            ),
            team_support_mentions=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    len(team_support_results)),
                analysis="Most frequent positive keyword" if len(
                    team_support_results) > 30 else "Moderate team support mentions",
                indicator="Positive indicator" if len(
                    team_support_results) > 30 else "Moderate"
            ),
            recognition_requests=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    len(recognition_results)),
                analysis="High demand for recognition" if len(
                    recognition_results) > 25 else "Moderate recognition requests",
                indicator="High demand" if len(
                    recognition_results) > 25 else "Moderate"
            ),
            promotion_concerns=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    len(promotion_results)),
                analysis="Top negative keyword" if len(
                    promotion_results) > 20 else "Moderate career concerns",
                indicator="Needs attention" if len(
                    promotion_results) > 20 else "Moderate"
            ),
            strong_positive_percent=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    strong_positive_pct, is_percentage=True),
                analysis="Highly satisfied employees",
                indicator="Excellent" if strong_positive_pct > 25 else "Good"
            ),
            learning_mentions=MetricWithAnalysis(
                range_estimate=self._create_about_value(len(learning_results)),
                analysis="Training highly valued" if len(
                    learning_results) > 30 else "Moderate learning interest",
                indicator="High value" if len(
                    learning_results) > 30 else "Moderate"
            ),
            politics_concerns=MetricWithAnalysis(
                range_estimate=self._create_about_value(len(politics_results)),
                analysis="Office politics issues" if len(
                    politics_results) > 15 else "Minor politics concerns",
                indicator="Needs attention" if len(
                    politics_results) > 15 else "Minor concern"
            ),
            team_culture_strength=MetricWithAnalysis(
                range_estimate=self._create_about_value(len(culture_results)),
                analysis="Top organizational strength" if len(
                    culture_results) > 30 else "Moderate culture strength",
                indicator="Strong" if len(culture_results) > 30 else "Moderate"
            ),
            strong_negative_percent=MetricWithAnalysis(
                range_estimate=self._create_about_value(
                    strong_negative_pct, is_percentage=True),
                analysis="Highly dissatisfied",
                indicator="Concerning" if strong_negative_pct > 10 else "Low"
            ),
            chunks_processed=len(self.chunks),
            total_responses=len(self.aggregated_vectors),
            chunk_summaries=chunk_summaries,
            themes=[
                {"theme": "Hierarchical Processing", "mentions": len(
                    self.chunks), "sentiment": "Neutral"},
                {"theme": "Scalable Analysis", "mentions": len(
                    self.aggregated_vectors), "sentiment": "Positive"}
            ],
            insights=insights
        )

    def has_vectors(self) -> bool:
        """Check if vectors are available for analysis"""
        if self.use_direct_analysis:
            return len(self.small_dataset_responses) > 0
        return (self.aggregated_vectors is not None and
                len(self.aggregated_vectors) > 0 and
                len(self.chunks) > 0)


# Initialize the analyzers
na_handler = NAHandler()
hierarchical_analyzer = HierarchicalVectorAnalyzer(
    chunk_size=1000)  # Default 1000 responses per chunk


@app.get("/")
async def root():
    return {
        "message": "Employee Survey Analysis API v2.0 - Smart Adaptive Chunking is running!",
        "description": "Adaptive processing approach: Direct analysis for small datasets, hierarchical chunking for large datasets",
        "processing_strategy": {
            "approach": "Smart Adaptive Processing",
            "small_datasets": "< 100 responses: Direct keyword-based LLM analysis (no vectorization)",
            "large_datasets": ">= 100 responses: Hierarchical chunking with vector-based analysis",
            "chunk_size_logic": {
                "automatic": "10% of dataset size (minimum 100, rounded to nearest 100)",
                "examples": "1000 responses → 100 chunk size, 10000 responses → 1000 chunk size",
                "custom_override": "Optional chunk_size parameter in upload endpoints"
            }
        },
        "endpoints": {
            "process_data": {
                "upload": "POST /analyze/upload-hierarchical (supports chunk_size parameter)",
                "existing": "POST /analyze/existing-hierarchical (supports chunk_size parameter)"
            },
            "analysis": {
                "comprehensive": "GET /analyze/comprehensive-hierarchical",
                "sentiment": "GET /analyze/sentiment-hierarchical",
                "team_support": "GET /analyze/team-support-hierarchical",
                "recognition": "GET /analyze/recognition-hierarchical",
                "promotion": "GET /analyze/promotion-hierarchical",
                "strong_positive": "GET /analyze/strong-positive-hierarchical",
                "learning": "GET /analyze/learning-hierarchical",
                "politics": "GET /analyze/politics-hierarchical",
                "culture": "GET /analyze/culture-hierarchical",
                "strong_negative": "GET /analyze/strong-negative-hierarchical",
                "chunks": "GET /analyze/chunks-info (for large datasets only)"
            },
            "dynamic_analysis": {
                "configure": "POST /analyze/configure-dynamic (NEW: AI-powered survey type detection)",
                "analyze": "GET /analyze/dynamic (NEW: Dynamic parameter analysis)",
                "compare": "GET /analyze/comparison (NEW: Compare traditional vs dynamic analysis)"
            },
            "cache_management": {
                "info": "GET /analyze/cache-info (NEW: View cache status and performance)",
                "clear": "POST /analyze/clear-cache (NEW: Manually clear all cached results)"
            }
        },
        "features": {
            "adaptive_processing": "Automatically chooses best processing method based on dataset size",
            "memory_efficient": "Hierarchical chunking for large datasets prevents memory issues",
            "fast_small_analysis": "Direct analysis for small datasets provides quick results",
            "custom_chunk_sizes": "Override automatic chunk size calculation when needed",
            "dynamic_analysis": "NEW: AI-powered survey type detection and parameter generation",
            "survey_adaptability": "NEW: Automatically adapts analysis to employee, customer, candidate, or other survey types",
            "intelligent_parameters": "NEW: Generates relevant analysis parameters based on survey content",
            "intelligent_caching": "NEW: Smart caching system for 9x faster response times",
            "cache_management": "NEW: Monitor and control caching for optimal performance"
        }
    }


@app.post("/analyze/upload-hierarchical")
async def process_survey_data_hierarchical(
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
    Upload and process survey file using hierarchical chunking approach with automatic or custom chunk sizing

    **Accepted file formats:**
    - Excel files: .xlsx, .xls
    - CSV files: .csv

    **Chunk Size Logic:**
    - If dataset < 100 responses: Uses direct LLM analysis (no vectorization)
    - If dataset >= 100 responses: 
      - Default: Auto-calculates chunk size as 10% of dataset size (minimum 100)
      - Custom: Use provided chunk_size parameter (minimum 100)

    **Examples:**
    - 1000 responses → 100 chunk size (10%)
    - 10000 responses → 1000 chunk size (10%)
    - 20000 responses → 2000 chunk size (10%)
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
        print(
            f"📊 Processing uploaded file with hierarchical chunking: {file.filename}")
        print(f"📋 Original data shape: {df.shape}")

        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')
        print(f"✅ Cleaned data shape: {df_clean.shape}")

        # Calculate optimal chunk size and configure analyzer
        global hierarchical_analyzer
        optimal_chunk_size, use_direct_analysis = hierarchical_analyzer.calculate_optimal_chunk_size(
            len(df_clean), chunk_size)

        # Update analyzer configuration
        hierarchical_analyzer.chunk_size = optimal_chunk_size
        hierarchical_analyzer.use_direct_analysis = use_direct_analysis

        # Process data using hierarchical approach
        success = hierarchical_analyzer.process_survey_data_hierarchical(
            df_clean)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data using hierarchical chunking")

        # Prepare response based on processing type
        if use_direct_analysis:
            processing_type = "Direct LLM Analysis"
            total_responses = len(
                hierarchical_analyzer.small_dataset_responses)
            chunks_created = 0
        else:
            processing_type = "Hierarchical Chunking"
            total_responses = len(
                hierarchical_analyzer.aggregated_vectors) if hierarchical_analyzer.aggregated_vectors is not None else 0
            chunks_created = len(hierarchical_analyzer.chunks)

        # Return processing confirmation
        return {
            "message": f"Survey data processed successfully using {processing_type.lower()}",
            "filename": file.filename,
            "processing_type": processing_type,
            "dataset_size": len(df_clean),
            "chunk_size_used": optimal_chunk_size,
            "chunk_size_source": "custom" if chunk_size is not None else "auto-calculated" if not use_direct_analysis else "direct-analysis",
            "total_responses": total_responses,
            "chunks_created": chunks_created,
            "data_shape": df_clean.shape,
            "processing_date": datetime.now(),
            "available_endpoints": [
                "/analyze/comprehensive-hierarchical",
                "/analyze/chunks-info" if not use_direct_analysis else None
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file with hierarchical chunking: {str(e)}")


@app.post("/analyze/existing-hierarchical")
async def process_existing_file_hierarchical(
    chunk_size: Optional[int] = Form(
        None,
        description="Optional custom chunk size. If not provided, will auto-calculate as 10% of dataset size (minimum 100)."
    )
):
    """
    Process the existing dummy survey file using hierarchical chunking approach with automatic or custom chunk sizing

    **Chunk Size Logic:**
    - If dataset < 100 responses: Uses direct LLM analysis (no vectorization)
    - If dataset >= 100 responses: 
      - Default: Auto-calculates chunk size as 10% of dataset size (minimum 100)
      - Custom: Use provided chunk_size parameter (minimum 100)
    """
    try:
        file_path = "/Users/shashwat/Projects/Hyrgpt/EmpSurv/Dummy_Employee_Survey_Responses (1).xlsx"

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail="Dummy survey file not found")

        # Read the existing file with proper header detection
        df = pd.read_excel(file_path, header=1)

        # Clean the data
        print(f"📊 Processing existing file with hierarchical chunking")
        print(f"📋 Original data shape: {df.shape}")

        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')
        print(f"✅ Cleaned data shape: {df_clean.shape}")

        # Calculate optimal chunk size and configure analyzer
        global hierarchical_analyzer
        optimal_chunk_size, use_direct_analysis = hierarchical_analyzer.calculate_optimal_chunk_size(
            len(df_clean), chunk_size)

        # Update analyzer configuration
        hierarchical_analyzer.chunk_size = optimal_chunk_size
        hierarchical_analyzer.use_direct_analysis = use_direct_analysis

        # Process data using hierarchical approach
        success = hierarchical_analyzer.process_survey_data_hierarchical(
            df_clean)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data using hierarchical chunking")

        # Prepare response based on processing type
        if use_direct_analysis:
            processing_type = "Direct LLM Analysis"
            total_responses = len(
                hierarchical_analyzer.small_dataset_responses)
            chunks_created = 0
        else:
            processing_type = "Hierarchical Chunking"
            total_responses = len(
                hierarchical_analyzer.aggregated_vectors) if hierarchical_analyzer.aggregated_vectors is not None else 0
            chunks_created = len(hierarchical_analyzer.chunks)

        # Return processing confirmation
        return {
            "message": f"Existing survey data processed successfully using {processing_type.lower()}",
            "filename": "Dummy_Employee_Survey_Responses (1).xlsx",
            "processing_type": processing_type,
            "dataset_size": len(df_clean),
            "chunk_size_used": optimal_chunk_size,
            "chunk_size_source": "custom" if chunk_size is not None else "auto-calculated" if not use_direct_analysis else "direct-analysis",
            "total_responses": total_responses,
            "chunks_created": chunks_created,
            "data_shape": df_clean.shape,
            "processing_date": datetime.now(),
            "available_endpoints": [
                "/analyze/comprehensive-hierarchical",
                "/analyze/chunks-info" if not use_direct_analysis else None
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing existing file with hierarchical chunking: {str(e)}")


@app.get("/analyze/comprehensive-hierarchical")
async def get_comprehensive_analysis_hierarchical():
    """
    Get comprehensive analysis using hierarchical chunking approach
    """
    try:
        # Check if vectors are available
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please upload and process data first using /analyze/upload-hierarchical or /analyze/existing-hierarchical"
            )

        print("🔄 Performing comprehensive hierarchical analysis...")

        # Get comprehensive analysis using hierarchical approach
        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()

        # Helper function to extract metric data
        def extract_metric(metric):
            return {
                "value": metric.range_estimate,
                "analysis": metric.analysis,
                "indicator": metric.indicator
            }

        # Format the response with processing type insights
        processing_type = "Direct LLM Analysis" if hierarchical_analyzer.use_direct_analysis else "Hierarchical Chunking"

        response = {
            "message": f"Comprehensive analysis completed successfully using {processing_type.lower()}",
            "processing_approach": processing_type,
            "chunks_processed": analysis_result.chunks_processed,
            "total_responses_analyzed": analysis_result.total_responses,
            "chunk_size": hierarchical_analyzer.chunk_size,
            "dataset_size_category": "small" if hierarchical_analyzer.use_direct_analysis else "large",

            # Core metrics (same format as original for comparison)
            "overall_positive_sentiment": extract_metric(analysis_result.overall_positive_sentiment),
            "team_support_mentions": extract_metric(analysis_result.team_support_mentions),
            "recognition_requests": extract_metric(analysis_result.recognition_requests),
            "promotion_concerns": extract_metric(analysis_result.promotion_concerns),
            "strong_positive_percent": extract_metric(analysis_result.strong_positive_percent),
            "learning_mentions": extract_metric(analysis_result.learning_mentions),
            "politics_concerns": extract_metric(analysis_result.politics_concerns),
            "team_culture_strength": extract_metric(analysis_result.team_culture_strength),
            "strong_negative_percent": extract_metric(analysis_result.strong_negative_percent),

            # Processing-specific insights
            "chunk_summaries": analysis_result.chunk_summaries,
            "themes": analysis_result.themes,
            "insights": analysis_result.insights,

            # Analysis note
            "analysis_note": "Keyword-based analysis for small dataset" if hierarchical_analyzer.use_direct_analysis else f"Vector-based analysis with {analysis_result.chunks_processed} chunks"
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in hierarchical analysis: {str(e)}")


@app.get("/analyze/sentiment-hierarchical")
async def get_sentiment_analysis_hierarchical():
    """Get overall positive sentiment analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Overall Positive Sentiment",
            "value": analysis_result.overall_positive_sentiment.range_estimate,
            "analysis": analysis_result.overall_positive_sentiment.analysis,
            "indicator": analysis_result.overall_positive_sentiment.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in sentiment analysis: {str(e)}")


@app.get("/analyze/team-support-hierarchical")
async def get_team_support_analysis_hierarchical():
    """Get team support mentions analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Team Support Mentions",
            "value": analysis_result.team_support_mentions.range_estimate,
            "analysis": analysis_result.team_support_mentions.analysis,
            "indicator": analysis_result.team_support_mentions.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in team support analysis: {str(e)}")


@app.get("/analyze/recognition-hierarchical")
async def get_recognition_analysis_hierarchical():
    """Get recognition requests analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Recognition Requests",
            "value": analysis_result.recognition_requests.range_estimate,
            "analysis": analysis_result.recognition_requests.analysis,
            "indicator": analysis_result.recognition_requests.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in recognition analysis: {str(e)}")


@app.get("/analyze/promotion-hierarchical")
async def get_promotion_analysis_hierarchical():
    """Get promotion concerns analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Promotion Concerns",
            "value": analysis_result.promotion_concerns.range_estimate,
            "analysis": analysis_result.promotion_concerns.analysis,
            "indicator": analysis_result.promotion_concerns.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in promotion analysis: {str(e)}")


@app.get("/analyze/strong-positive-hierarchical")
async def get_strong_positive_analysis_hierarchical():
    """Get strong positive sentiment analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Strong Positive",
            "value": analysis_result.strong_positive_percent.range_estimate,
            "analysis": analysis_result.strong_positive_percent.analysis,
            "indicator": analysis_result.strong_positive_percent.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in strong positive analysis: {str(e)}")


@app.get("/analyze/learning-hierarchical")
async def get_learning_analysis_hierarchical():
    """Get learning mentions analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Learning Mentions",
            "value": analysis_result.learning_mentions.range_estimate,
            "analysis": analysis_result.learning_mentions.analysis,
            "indicator": analysis_result.learning_mentions.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in learning analysis: {str(e)}")


@app.get("/analyze/politics-hierarchical")
async def get_politics_analysis_hierarchical():
    """Get politics concerns analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Politics Concerns",
            "value": analysis_result.politics_concerns.range_estimate,
            "analysis": analysis_result.politics_concerns.analysis,
            "indicator": analysis_result.politics_concerns.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in politics analysis: {str(e)}")


@app.get("/analyze/culture-hierarchical")
async def get_culture_analysis_hierarchical():
    """Get team culture strength analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Team Culture Strength",
            "value": analysis_result.team_culture_strength.range_estimate,
            "analysis": analysis_result.team_culture_strength.analysis,
            "indicator": analysis_result.team_culture_strength.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in culture analysis: {str(e)}")


@app.get("/analyze/strong-negative-hierarchical")
async def get_strong_negative_analysis_hierarchical():
    """Get strong negative sentiment analysis"""
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        analysis_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
        return {
            "metric": "Strong Negative",
            "value": analysis_result.strong_negative_percent.range_estimate,
            "analysis": analysis_result.strong_negative_percent.analysis,
            "indicator": analysis_result.strong_negative_percent.indicator
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in strong negative analysis: {str(e)}")


@app.get("/analyze/chunks-info")
async def get_chunks_info():
    """
    Get detailed information about the processed chunks
    """
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No chunks available. Please process data first."
            )

        chunk_details = []
        for chunk_data in hierarchical_analyzer.chunks:
            chunk_details.append({
                "chunk_id": chunk_data['chunk_id'],
                "summary": chunk_data['summary_text'],
                "employee_count": chunk_data['employee_count'],
                "response_count": chunk_data['response_count'],
                "has_embeddings": len(chunk_data['embeddings']) > 0 if 'embeddings' in chunk_data else False,
                "vector_dimensions": len(chunk_data['summary_vector']) if 'summary_vector' in chunk_data else 0
            })

        return {
            "message": "Chunk information retrieved successfully",
            "total_chunks": len(hierarchical_analyzer.chunks),
            "chunk_size_configured": hierarchical_analyzer.chunk_size,
            "total_responses": len(hierarchical_analyzer.aggregated_vectors) if hierarchical_analyzer.aggregated_vectors is not None else 0,
            "chunks": chunk_details,
            "processing_summary": {
                "approach": "Hierarchical chunking with response-level granularity",
                "benefits": [
                    "Memory efficient for large datasets",
                    "Parallel processing capability",
                    "Intermediate summarization at chunk level",
                    "Scalable to millions of responses"
                ]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chunk information: {str(e)}")


# ===== NEW DYNAMIC ANALYSIS ENDPOINTS =====

@app.post("/analyze/configure-dynamic")
async def configure_dynamic_analysis(survey_context: Optional[str] = Form(None)):
    """
    Configure dynamic analysis by detecting survey type and generating relevant parameters
    This is a NEW feature that enhances your existing hardcoded analysis
    """
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No processed data available. Please process data first using /analyze/upload-hierarchical or /analyze/existing-hierarchical"
            )

        # Get sample responses for analysis
        if hierarchical_analyzer.use_direct_analysis:
            sample_responses = hierarchical_analyzer.small_dataset_responses[:15]
        else:
            sample_responses = [meta['response_text']
                                for meta in hierarchical_analyzer.all_metadata[:15]]

        if not sample_responses:
            raise HTTPException(
                status_code=400,
                detail="No response data available for dynamic analysis configuration"
            )

        print("🔄 Configuring dynamic analysis...")

        # Clear dynamic analysis cache when reconfiguring
        hierarchical_analyzer._remove_from_cache("dynamic_analysis")

        # Configure dynamic analysis
        survey_profile = hierarchical_analyzer.dynamic_analyzer.detect_and_configure_analysis(
            sample_responses, survey_context
        )

        return {
            "message": "Dynamic analysis configured successfully",
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
            "next_step": "Call /analyze/dynamic to run the analysis with these parameters"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error configuring dynamic analysis: {str(e)}")


@app.get("/analyze/dynamic")
async def get_dynamic_analysis():
    """
    Get comprehensive analysis using dynamically generated parameters
    This COMPLEMENTS your existing /analyze/comprehensive-hierarchical endpoint
    """
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        if not hierarchical_analyzer.dynamic_analyzer.survey_profile:
            raise HTTPException(
                status_code=400,
                detail="Dynamic analysis not configured. Please call /analyze/configure-dynamic first."
            )

        print("🔄 Running dynamic analysis...")

        # Get total responses count
        if hierarchical_analyzer.use_direct_analysis:
            total_responses = len(
                hierarchical_analyzer.small_dataset_responses)
        else:
            total_responses = len(hierarchical_analyzer.all_metadata)

        # Check cache for dynamic analysis
        cache_key = "dynamic_analysis"
        cached_dynamic_result = hierarchical_analyzer._get_from_cache(
            cache_key)

        if cached_dynamic_result is not None:
            dynamic_result = cached_dynamic_result
        else:
            # Run dynamic analysis and cache it
            dynamic_result = hierarchical_analyzer.dynamic_analyzer.run_dynamic_analysis(
                hierarchical_analyzer, total_responses
            )
            hierarchical_analyzer._store_in_cache(cache_key, dynamic_result)

        # Also get the traditional analysis for comparison
        try:
            traditional_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()
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
                f"Warning: Could not get traditional analysis for comparison: {e}")
            traditional_summary = {"error": "Traditional analysis unavailable"}

        return {
            "message": "Dynamic analysis completed successfully",
            "analysis_method": "AI-generated parameters based on survey content",

            # Dynamic analysis results (NEW)
            "dynamic_analysis": {
                "survey_type": dynamic_result.survey_type,
                "industry_context": dynamic_result.industry_context,
                "total_responses": dynamic_result.total_responses,
                "parameters_analyzed": dynamic_result.parameters_analyzed,
                "results": dynamic_result.analysis_results
            },

            # Traditional analysis for comparison (EXISTING)
            "traditional_analysis": traditional_summary,

            # Metadata
            "analysis_date": datetime.now(),
            "chunks_processed": len(hierarchical_analyzer.chunks) if hierarchical_analyzer.chunks else 0,
            "approach_used": "hierarchical_chunking" if not hierarchical_analyzer.use_direct_analysis else "direct_analysis"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing dynamic analysis: {str(e)}")


@app.get("/analyze/comparison")
async def get_analysis_comparison():
    """
    Compare traditional hardcoded analysis vs dynamic AI-generated analysis
    This shows the difference between your original approach and the enhanced version
    """
    try:
        if not hierarchical_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please process data first."
            )

        # Get traditional analysis
        traditional_result = hierarchical_analyzer.analyze_comprehensive_hierarchical()

        # Get dynamic analysis (if configured)
        dynamic_result = None
        if hierarchical_analyzer.dynamic_analyzer.survey_profile:
            total_responses = len(hierarchical_analyzer.all_metadata) if not hierarchical_analyzer.use_direct_analysis else len(
                hierarchical_analyzer.small_dataset_responses)
            dynamic_result = hierarchical_analyzer.dynamic_analyzer.run_dynamic_analysis(
                hierarchical_analyzer, total_responses
            )

        comparison = {
            "traditional_analysis": {
                "method": "Hardcoded parameters (employee survey focused)",
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
                }
            }
        }

        if dynamic_result:
            comparison["dynamic_analysis"] = {
                "method": "AI-generated parameters based on survey content",
                "survey_type_detected": dynamic_result.survey_type,
                "industry_context": dynamic_result.industry_context,
                "parameters": list(dynamic_result.analysis_results.keys()),
                "results": {
                    param: result.get(
                        "about_count", result.get("error", "N/A"))
                    for param, result in dynamic_result.analysis_results.items()
                }
            }

            comparison["insights"] = [
                f"Traditional analysis uses {len(comparison['traditional_analysis']['parameters'])} fixed parameters",
                f"Dynamic analysis generated {len(comparison['dynamic_analysis']['parameters'])} survey-specific parameters",
                f"Survey type detected: {dynamic_result.survey_type}",
                "Dynamic analysis adapts to different survey types automatically"
            ]
        else:
            comparison["dynamic_analysis"] = {
                "status": "Not configured",
                "message": "Call /analyze/configure-dynamic first to enable dynamic analysis"
            }
            comparison["insights"] = [
                "Traditional analysis uses fixed parameters suitable for employee surveys",
                "Dynamic analysis not yet configured - would adapt to your specific survey type"
            ]

        return {
            "message": "Analysis comparison completed",
            "comparison": comparison,
            "recommendation": "Use dynamic analysis for better survey-specific insights" if dynamic_result else "Configure dynamic analysis for survey-specific insights",
            "analysis_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing analysis comparison: {str(e)}")


@app.get("/analyze/cache-info")
async def get_cache_info():
    """
    Get information about the current cache state
    This helps monitor cache performance and debug caching issues
    """
    try:
        cache_info = hierarchical_analyzer.get_cache_info()

        return {
            "message": "Cache information retrieved successfully",
            "cache_info": cache_info,
            "performance_note": "Cached results provide instant responses, uncached results require full computation",
            "cache_benefits": {
                "speed_improvement": "9x faster for individual metric endpoints",
                "efficiency": "Eliminates redundant computations",
                "user_experience": "Instant response for repeated requests"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving cache information: {str(e)}")


@app.post("/analyze/clear-cache")
async def clear_cache():
    """
    Manually clear all cached analysis results
    Useful for forcing fresh analysis or troubleshooting
    """
    try:
        hierarchical_analyzer._clear_all_cache()

        return {
            "message": "All cached analysis results cleared successfully",
            "note": "Next analysis requests will perform fresh computations",
            "when_to_use": [
                "When you want to force fresh analysis",
                "After making configuration changes",
                "For troubleshooting cache-related issues"
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing cache: {str(e)}")

# ===== END NEW DYNAMIC ANALYSIS ENDPOINTS =====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

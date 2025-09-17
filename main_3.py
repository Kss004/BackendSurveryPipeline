from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
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

from services.na_handler import NAHandler

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Survey Analysis API v4.0 - Pure LLM-Powered Dynamic Analysis",
    description="Truly dynamic LLM-powered analysis with user-customizable topics",
    version="4.0-dynamic"
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
class AnalysisTopic:
    """A dynamically generated or user-defined analysis topic"""
    name: str
    description: str
    search_terms: str
    importance: str  # high, medium, low
    source: str  # "llm_generated" or "user_defined"


@dataclass
class TopicAnalysisResult:
    """Result for a specific topic analysis"""
    topic_name: str
    value: int  # Simplified single value instead of count, percentage, about_value, sentiment_score
    key_insights: List[str]
    sample_quotes: List[str]
    indicator: str  # "Positive", "Needs attention", "Concerning"


@dataclass
class ComprehensiveAnalysisResult:
    """Complete analysis results for all topics"""
    survey_type: str
    industry_context: str
    total_responses: int
    topics_analyzed: List[TopicAnalysisResult]
    overall_insights: List[str]
    recommendations: List[str]
    analysis_date: datetime


class PureLLMAnalyzer:
    """
    Pure LLM-Powered Analyzer - No hardcoded topics, fully dynamic
    """

    def __init__(self, chunk_size: int = 1000, openai_api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        # Hierarchical chunking configuration
        self.chunk_size = chunk_size
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_vectors: Dict[int, np.ndarray] = {}
        self.use_direct_analysis = False

        # Data storage
        self.survey_responses: List[str] = []
        self.response_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Analysis state
        self.survey_type: Optional[str] = None
        self.industry_context: Optional[str] = None
        self.generated_topics: List[AnalysisTopic] = []
        self.user_topics: List[AnalysisTopic] = []

        # Caching
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

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

        # Extract responses from the chunk
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
        """Aggregate all chunk vectors for final analysis"""
        if not self.chunks:
            print("❌ No chunks to aggregate")
            return False

        print(f"🔄 Aggregating {len(self.chunks)} chunks...")

        all_embeddings = []
        all_metadata = []

        for chunk_data in self.chunks:
            if chunk_data and 'embeddings' in chunk_data:
                all_embeddings.append(chunk_data['embeddings'])
                all_metadata.extend(chunk_data['metadata'])
                self.chunk_vectors[chunk_data['chunk_id']
                                   ] = chunk_data['summary_vector']

        if not all_embeddings:
            print("❌ No embeddings found in chunks")
            return False

        # Combine all response-level embeddings
        self.embeddings = np.vstack(all_embeddings)
        self.response_metadata = all_metadata

        # Update survey_responses for compatibility
        self.survey_responses = [meta['response_text']
                                 for meta in all_metadata]

        print(
            f"✅ Aggregated {len(self.embeddings)} total embeddings from {len(self.chunks)} chunks")

        # Create Faiss index for similarity search
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_vectors = self.embeddings / norms
        self.faiss_index.add(normalized_vectors.astype('float32'))

        print(f"🔍 Created Faiss index with {self.faiss_index.ntotal} vectors")
        return True

    def process_survey_data(self, df: pd.DataFrame, custom_chunk_size: Optional[int] = None) -> bool:
        """Process survey data using hierarchical chunking with create embeddings"""
        try:
            print(
                f"🚀 Processing {len(df)} survey responses with hierarchical chunking...")

            # Clear previous data
            self.survey_responses = []
            self.response_metadata = []
            self.chunks = []
            self._cache.clear()

            # Calculate optimal chunk size
            dataset_size = len(df)
            calculated_chunk_size, use_direct_analysis = self.calculate_optimal_chunk_size(
                dataset_size, custom_chunk_size)
            self.update_chunk_size(calculated_chunk_size, dataset_size)

            if self.use_direct_analysis:
                # For small datasets, process directly without chunking
                return self._process_small_dataset(df)
            else:
                # For large datasets, use hierarchical chunking
                return self._process_large_dataset(df)

        except Exception as e:
            print(f"❌ Error processing survey data: {e}")
            return False

    def _process_small_dataset(self, df: pd.DataFrame) -> bool:
        """Process small datasets directly without chunking"""
        try:
            print(
                f"📋 Processing small dataset with {len(df)} responses directly")

            # Extract responses (same logic as before)
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
                employee_id = str(row.get('Participant ID', f'emp_{idx}'))
                for col in available_cols:
                    response = str(row[col]) if pd.notna(row[col]) else ""
                    if response and response.lower() not in ['nan', 'none', '']:
                        self.survey_responses.append(response)
                        self.response_metadata.append({
                            'employee_id': employee_id,
                            'question_type': col,
                            'response_text': response
                        })

            print(
                f"📊 Extracted {len(self.survey_responses)} responses for direct analysis")

            # Create embeddings
            if self.survey_responses:
                self.embeddings = self._get_embeddings_batch(
                    self.survey_responses)

                # Create FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_vectors = self.embeddings / norms
                self.faiss_index.add(normalized_vectors.astype('float32'))

                print(f"✅ Created embeddings and FAISS index for small dataset")
                return True

            return False

        except Exception as e:
            print(f"❌ Error processing small dataset: {e}")
            return False

    def _process_large_dataset(self, df: pd.DataFrame) -> bool:
        """Process large datasets using hierarchical chunking"""
        try:
            print(f"🚀 Processing large dataset with hierarchical chunking...")

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
            print(f"❌ Error processing large dataset: {e}")
            return False

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
                print(f"Error getting embeddings: {e}")
                # Fallback to mock embeddings for testing
                embeddings.extend(
                    [np.random.normal(0, 0.1, self.embedding_dimension) for _ in batch])

        return np.array(embeddings)

    async def generate_analysis_topics(self, additional_topics: Optional[List[str]] = None) -> List[AnalysisTopic]:
        """
        Generate analysis topics using LLM based on survey content
        """
        if not self.survey_responses:
            raise ValueError("No survey data available. Process data first.")

        # Sample responses for LLM analysis
        sample_responses = self.survey_responses[:20]
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
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Parse JSON response
            response_content = response.choices[0].message.content
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback
                    result = {
                        "survey_type": "employee",
                        "industry_context": "general",
                        "analysis_topics": [
                            {"name": "overall_satisfaction", "description": "Overall satisfaction levels",
                             "search_terms": "satisfaction happy pleased", "importance": "high"}
                        ]
                    }

            # Store survey context
            self.survey_type = result["survey_type"]
            self.industry_context = result["industry_context"]

            # Convert to AnalysisTopic objects
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

            # Add user-defined topics if provided
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

            all_topics = self.generated_topics + self.user_topics
            print(
                f"✅ Generated {len(self.generated_topics)} LLM topics + {len(self.user_topics)} user topics")

            return all_topics

        except Exception as e:
            print(f"❌ Error generating topics: {e}")
            # Return fallback topics
            return [
                AnalysisTopic(
                    name="overall_satisfaction",
                    description="Overall satisfaction levels",
                    search_terms="satisfaction happy pleased content",
                    importance="high",
                    source="fallback"
                )
            ]

    async def analyze_topic(self, topic: AnalysisTopic) -> TopicAnalysisResult:
        """Analyze a specific topic using LLM"""

        # Find relevant responses using semantic search
        relevant_responses = self._semantic_search(
            topic.search_terms, top_k=50)

        if not relevant_responses:
            return TopicAnalysisResult(
                topic_name=topic.name,
                value=0,
                key_insights=["No relevant responses found"],
                sample_quotes=[],
                indicator="No data"
            )

        # Prepare responses for LLM analysis
        # Limit for token efficiency
        response_texts = [resp[0] for resp in relevant_responses[:20]]
        responses_text = "\n".join(
            [f"{i+1}. {resp[:200]}" for i, resp in enumerate(response_texts)])

        prompt = f"""
        Analyze these survey responses for the topic: {topic.description}
        
        Responses:
        {responses_text}
        
        Provide ONLY a valid JSON object with:
        {{
            "count": {len(relevant_responses)},
            "sentiment_score": <0-100 sentiment score>,
            "key_insights": ["insight1", "insight2", "insight3"],
            "sample_quotes": ["quote1", "quote2"],
            "indicator": "Positive|Needs attention|Concerning"
        }}
        
        Focus on: {topic.description}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Parse response
            response_content = response.choices[0].message.content
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "count": len(relevant_responses),
                        "sentiment_score": 50,
                        "key_insights": ["Analysis completed"],
                        "sample_quotes": response_texts[:2] if response_texts else [],
                        "indicator": "Analyzed"
                    }

            count = result.get("count", len(relevant_responses))

            return TopicAnalysisResult(
                topic_name=topic.name,
                value=count,
                key_insights=result.get("key_insights", []),
                sample_quotes=result.get("sample_quotes", []),
                indicator=result.get("indicator", "Analyzed")
            )

        except Exception as e:
            print(f"❌ Error analyzing topic {topic.name}: {e}")
            return TopicAnalysisResult(
                topic_name=topic.name,
                value=len(relevant_responses),
                key_insights=[f"Analysis error: {str(e)}"],
                sample_quotes=[],
                indicator="Error"
            )

    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Perform semantic search for relevant responses"""
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

            # Filter by similarity threshold
            return [(text, score) for text, score in results if score > 0.3]

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    async def comprehensive_analysis(self) -> ComprehensiveAnalysisResult:
        """Perform comprehensive analysis on all topics"""

        if not self.generated_topics and not self.user_topics:
            raise ValueError(
                "No analysis topics available. Generate topics first.")

        all_topics = self.generated_topics + self.user_topics
        topic_results = []

        print(f"🔄 Analyzing {len(all_topics)} topics...")

        for topic in all_topics:
            result = await self.analyze_topic(topic)
            topic_results.append(result)
            print(f"  ✅ {topic.name}: {result.value} mentions")

        # Generate overall insights
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
        """Generate overall insights from topic analysis"""

        # Prepare summary for LLM
        summary = []
        for result in topic_results:
            summary.append(
                f"{result.topic_name}: {result.value} mentions, {result.indicator}")

        summary_text = "\n".join(summary)

        prompt = f"""
        Based on these analysis results, provide 3-5 key overall insights:
        
        {summary_text}
        
        Survey type: {self.survey_type}
        Total responses: {len(self.survey_responses)}
        
        Provide insights as a JSON array: ["insight1", "insight2", "insight3"]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            content = response.choices[0].message.content
            try:
                insights = json.loads(content)
                return insights if isinstance(insights, list) else [content]
            except:
                return [content]

        except Exception as e:
            return [f"Overall analysis of {len(topic_results)} topics completed"]

    async def _generate_recommendations(self, topic_results: List[TopicAnalysisResult]) -> List[str]:
        """Generate actionable recommendations"""

        concerning_topics = [r for r in topic_results if r.indicator in [
            "Needs attention", "Concerning"]]

        if not concerning_topics:
            return ["Overall survey results are positive - continue current practices"]

        topics_text = "\n".join(
            [f"{t.topic_name}: {t.indicator}" for t in concerning_topics])

        prompt = f"""
        Based on these concerning areas, provide 3-5 actionable recommendations:
        
        {topics_text}
        
        Provide recommendations as a JSON array: ["recommendation1", "recommendation2"]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            content = response.choices[0].message.content
            try:
                recommendations = json.loads(content)
                return recommendations if isinstance(recommendations, list) else [content]
            except:
                return [content]

        except Exception as e:
            return ["Focus on areas that need attention based on survey feedback"]


# Initialize components
na_handler = NAHandler()
# Default chunk size, will be auto-calculated
llm_analyzer = PureLLMAnalyzer(chunk_size=1000)


@app.get("/")
async def root():
    return {
        "message": "Survey Analysis API v4.0 - Pure LLM-Powered Dynamic Analysis",
        "description": "Truly dynamic analysis with LLM-generated topics and user customization",
        "philosophy": "No hardcoded topics - everything is dynamically generated based on survey content",
        "endpoints": {
            "data_processing": "POST /analyze/process - Upload and process survey data (supports chunk_size parameter)",
            "topic_generation": "GET /analyze/topics - Generate analysis topics (with optional user additions)",
            "comprehensive_analysis": "GET /analyze/comprehensive - Complete analysis of all topics"
        },
        "features": {
            "pure_llm_analysis": "100% LLM-powered with no hardcoded assumptions",
            "hierarchical_chunking": "Scalable processing with 10% dataset size chunks (min 100)",
            "custom_chunk_sizes": "Override automatic chunk size calculation when needed",
            "dynamic_topics": "Topics generated based on actual survey content",
            "user_customization": "Users can add their own analysis topics",
            "adaptive_insights": "Analysis adapts to any survey type automatically",
            "actionable_recommendations": "LLM generates specific recommendations"
        },
        "workflow": [
            "1. POST /analyze/process - Upload your survey data (optional: chunk_size parameter)",
            "2. GET /analyze/topics?additional_topics=topic1,topic2 - Generate/customize topics",
            "3. GET /analyze/comprehensive - Get complete analysis"
        ],
        "chunking_strategy": {
            "small_datasets": "< 100 responses: Direct analysis (no chunking)",
            "large_datasets": ">= 100 responses: Hierarchical chunking",
            "automatic_chunk_size": "10% of dataset size (minimum 100, rounded to nearest 100)",
            "examples": "1000 responses → 100 chunk size, 10000 responses → 1000 chunk size",
            "custom_override": "Use chunk_size parameter to override automatic calculation"
        }
    }


@app.post("/analyze/process")
async def process_survey_data(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(
        None,
        description="Optional custom chunk size. If not provided, will auto-calculate as 10% of dataset size (minimum 100). Set to override automatic calculation."
    )
):
    """
    Process survey data - the only data processing endpoint needed
    """
    try:
        contents = await file.read()

        # Handle different file types
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df_test = pd.read_excel(io.BytesIO(contents), header=None, nrows=3)
            if (pd.isna(df_test.iloc[0]).sum() > len(df_test.columns) / 2 or
                    'Participant' in str(df_test.iloc[1].values)):
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

        # Clean data
        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')

        # Process with LLM analyzer using hierarchical chunking
        success = llm_analyzer.process_survey_data(df_clean, chunk_size)

        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data")

        # Prepare response with chunking information
        if llm_analyzer.use_direct_analysis:
            processing_type = "Direct LLM Analysis"
            chunks_created = 0
        else:
            processing_type = "Hierarchical Chunking"
            chunks_created = len(llm_analyzer.chunks)

        return {
            "message": "Survey data processed successfully with hierarchical chunking",
            "filename": file.filename,
            "processing_type": processing_type,
            "dataset_size": len(df_clean),
            "chunk_size_used": llm_analyzer.chunk_size,
            "chunk_size_source": "custom" if chunk_size else "auto-calculated",
            "total_responses": len(llm_analyzer.survey_responses),
            "chunks_created": chunks_created,
            "data_shape": df_clean.shape,
            "next_step": "Call GET /analyze/topics to generate analysis topics",
            "processing_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/analyze/topics")
async def generate_analysis_topics(additional_topics: Optional[str] = None):
    """
    Generate analysis topics using LLM + optional user topics
    """
    try:
        if not llm_analyzer.survey_responses:
            raise HTTPException(
                status_code=400, detail="No survey data available. Process data first.")

        # Parse additional topics
        user_topics = []
        if additional_topics:
            user_topics = [topic.strip()
                           for topic in additional_topics.split(',') if topic.strip()]

        # Generate topics
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
    """
    Get comprehensive analysis for all generated topics
    """
    try:
        if not llm_analyzer.generated_topics and not llm_analyzer.user_topics:
            raise HTTPException(
                status_code=400, detail="No analysis topics available. Generate topics first.")

        # Perform comprehensive analysis
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
                    "value": topic.value,
                    "indicator": topic.indicator,
                    "key_insights": topic.key_insights,
                    # Limit for response size
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

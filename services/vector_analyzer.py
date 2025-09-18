"""
Vector-Powered Survey Analysis using OpenAI Embeddings + Faiss

This module provides advanced semantic analysis of employee survey responses using
OpenAI's text-embedding-3-small model and Faiss for efficient vector similarity search.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import OpenAI


@dataclass
class MetricWithAnalysis:
    """A metric that includes range, percentage, and analysis"""
    range_estimate: str  # e.g., "220-240"
    analysis: str  # e.g., "Most frequent positive keyword - Positive indicator"
    indicator: str  # e.g., "Positive indicator", "Needs attention"
    percentage: Optional[float] = None  # e.g., 45.2 (when applicable)


@dataclass
class VectorAnalysisResult:
    """Container for vector analysis results with ranges and descriptive analysis"""
    overall_positive_sentiment: MetricWithAnalysis  # e.g., 53% - Above average satisfaction
    # e.g., 220-240 - Most frequent positive keyword
    team_support_mentions: MetricWithAnalysis
    # e.g., 162 - High demand for recognition
    recognition_requests: MetricWithAnalysis
    promotion_concerns: MetricWithAnalysis  # e.g., 110-115 - Top negative keyword
    # e.g., 29% - Highly satisfied employees
    strong_positive_percent: MetricWithAnalysis
    learning_mentions: MetricWithAnalysis  # e.g., 170-180 - Training highly valued
    politics_concerns: MetricWithAnalysis  # e.g., 60-76 - Office politics issues
    # e.g., 32% - Top organizational strength
    team_culture_strength: MetricWithAnalysis
    strong_negative_percent: MetricWithAnalysis  # e.g., 8% - Highly dissatisfied
    themes: List[Dict[str, Any]]
    cohort_analysis: List[Dict[str, Any]]
    insights: List[str]


@dataclass
class EmbeddingMetadata:
    """Metadata for each embedding"""
    employee_id: str
    question_type: str
    response_text: str
    gender: str
    tenure: str
    response_index: int


class VectorAnalyzer:
    """
    Advanced survey analysis using OpenAI embeddings and Faiss vector search
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        # Faiss index and metadata storage
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[EmbeddingMetadata] = []
        self.embeddings_cache: Optional[np.ndarray] = None

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

    def _initialize_sentiment_vectors(self):
        """Initialize reference vectors for sentiment analysis"""
        try:
            positive_embeddings = []
            negative_embeddings = []

            # Get embeddings for positive examples
            for text in self.sentiment_examples['positive']:
                embedding = self._get_embedding(text)
                if embedding is not None:
                    positive_embeddings.append(embedding)

            # Get embeddings for negative examples
            for text in self.sentiment_examples['negative']:
                embedding = self._get_embedding(text)
                if embedding is not None:
                    negative_embeddings.append(embedding)

            if positive_embeddings and negative_embeddings:
                self.sentiment_vectors = {
                    'positive': np.mean(positive_embeddings, axis=0),
                    'negative': np.mean(negative_embeddings, axis=0)
                }
                print("✅ Sentiment reference vectors initialized")
            else:
                print("⚠️ Failed to initialize sentiment vectors")

        except Exception as e:
            print(f"⚠️ Error initializing sentiment vectors: {e}")
            self.sentiment_vectors = None

    def has_vectors(self) -> bool:
        """Check if vectors are available for analysis"""
        return (self.embeddings_cache is not None and
                len(self.embeddings_cache) > 0 and
                len(self.metadata) > 0)

    def get_vectors_info(self) -> Dict[str, Any]:
        """Get information about stored vectors"""
        if not self.has_vectors():
            return {"vectors_available": False}

        return {
            "vectors_available": True,
            "total_responses": len(self.metadata),
            "vector_dimension": self.embedding_dimension,
            "embeddings_shape": self.embeddings_cache.shape,
            "unique_employees": len(set(m.employee_id for m in self.metadata)),
            "genders": list(set(m.gender for m in self.metadata)),
            "tenures": list(set(m.tenure for m in self.metadata))
        }

    def _get_embedding(self, text: str, retries: int = 3) -> Optional[np.ndarray]:
        """Get embedding for a single text using OpenAI API"""
        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text.replace("\n", " ").strip()
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                print(f"⚠️ Embedding attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    return None
        return None

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=[text.replace("\n", " ").strip() for text in batch]
                )

                batch_embeddings = [np.array(item.embedding)
                                    for item in response.data]
                embeddings.extend(batch_embeddings)

                print(
                    f"✅ Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            except Exception as e:
                print(f"⚠️ Batch embedding failed: {e}")
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch))

        return embeddings

    def process_survey_data_from_list(self, survey_data: List[Dict[str, str]]) -> bool:
        """Process survey data from a list of dictionaries and create vector embeddings"""
        print("🔄 Processing survey data from list for vector analysis...")

        # Clear existing data
        self.metadata = []
        responses_data = []

        # Extract responses and metadata
        for idx, entry in enumerate(survey_data):
            employee_id = entry.get('employee_id', f'EMP{idx+1:03d}')
            gender = entry.get('gender', 'Unknown')
            tenure = entry.get('tenure', 'Unknown')
            response_text = entry.get('response', '')

            if not response_text or response_text.strip() == '':
                continue

            # Store response data
            responses_data.append(response_text)

            # Store metadata
            metadata = EmbeddingMetadata(
                employee_id=employee_id,
                question_type='survey_response',
                response_text=response_text,
                gender=gender,
                tenure=tenure,
                response_index=len(self.metadata)
            )
            self.metadata.append(metadata)

        if not responses_data:
            print("❌ No valid responses found in survey data")
            return False

        print(f"📊 Processing {len(responses_data)} responses...")

        # Create embeddings for all responses
        try:
            embeddings = self._get_embeddings_batch(responses_data)

            # Store embeddings in cache
            self.embeddings_cache = np.array(embeddings)

            # Create Faiss index
            dimension = embeddings[0].shape[0]
            # Inner product for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)

            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / \
                np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(normalized_embeddings.astype('float32'))

            # Create sentiment reference vectors
            self._create_sentiment_vectors()

            print(f"✅ Successfully processed {len(responses_data)} responses")
            print(f"📐 Embedding dimension: {dimension}")
            print(f"🔍 Faiss index created with {self.index.ntotal} vectors")

            return True

        except Exception as e:
            print(f"❌ Error processing survey data: {e}")
            return False

    def process_survey_data(self, df: pd.DataFrame) -> bool:
        """Process survey data and create vector embeddings"""
        print("🔄 Processing survey data for vector analysis...")

        # Clear existing data
        self.metadata = []
        responses_data = []

        # Extract all text responses with metadata
        demographic_cols = ['Participant',
                            'GENDER', 'TENURE IN JAQ', 'Department']

        for idx, row in df.iterrows():
            employee_id = row.get('Participant', f'Employee_{idx}')
            gender = row.get('GENDER', 'Unknown')
            tenure = row.get('TENURE IN JAQ', 'Unknown')

            for col in df.columns:
                if col not in demographic_cols:
                    response_text = row[col]

                    # Skip invalid responses
                    if (pd.isna(response_text) or
                            response_text in ['No response provided', 'nan', 'NaN', '']):
                        continue

                    # Store response data
                    responses_data.append(response_text)

                    # Store metadata
                    metadata = EmbeddingMetadata(
                        employee_id=employee_id,
                        question_type=col,
                        response_text=response_text,
                        gender=gender,
                        tenure=tenure,
                        response_index=len(responses_data) - 1
                    )
                    self.metadata.append(metadata)

        print(
            f"📊 Found {len(responses_data)} valid responses from {len(df)} employees")

        if not responses_data:
            print("❌ No valid responses found")
            return False

        # Get embeddings for all responses
        print("🔄 Generating embeddings...")
        embeddings = self._get_embeddings_batch(responses_data)

        # Filter out failed embeddings
        valid_embeddings = []
        valid_metadata = []

        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_metadata.append(self.metadata[i])

        if not valid_embeddings:
            print("❌ No valid embeddings generated")
            return False

        # Update metadata to only include valid ones
        self.metadata = valid_metadata
        self.embeddings_cache = np.array(valid_embeddings)

        # Create Faiss index
        self._create_faiss_index()

        print(f"✅ Successfully processed {len(valid_embeddings)} embeddings")
        return True

    def _create_faiss_index(self):
        """Create and populate Faiss index for fast similarity search"""
        if self.embeddings_cache is None:
            return

        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings_cache / np.linalg.norm(
            self.embeddings_cache, axis=1, keepdims=True
        )

        # Create Faiss index (Inner Product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(embeddings_normalized.astype('float32'))

        print(f"✅ Faiss index created with {self.index.ntotal} vectors")

    def _analyze_with_openai(self, prompt: str, response_texts: List[str]) -> Dict[str, Any]:
        """Use OpenAI API to analyze survey responses with custom prompts"""
        try:
            # Combine all responses for analysis
            combined_text = "\n".join(
                [f"Response {i+1}: {text}" for i, text in enumerate(response_texts)])

            # Create the full prompt
            full_prompt = f"""
{prompt}

Survey Responses to Analyze:
{combined_text}

Please provide your analysis in JSON format.
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst. Provide precise, data-driven analysis in JSON format."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1  # Low temperature for consistent results
            )

            # Parse the response
            analysis_text = response.choices[0].message.content

            # Try to extract JSON from the response
            try:
                import json
                import re
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # Fallback: parse structured text
                    return {"analysis": analysis_text, "raw_response": True}
            except:
                return {"analysis": analysis_text, "raw_response": True}

        except Exception as e:
            print(f"⚠️ OpenAI analysis error: {e}")
            return {"error": str(e)}

    def _analyze_theme_percentages(self, all_responses: List[str]) -> Dict[str, float]:
        """Analyze what percentage of responses mention each theme using OpenAI"""

        prompt = """
Analyze the following employee survey responses and determine what percentage mention each of these themes:

1. Team Support (collaboration, help, assistance, teamwork, mentorship)
2. Recognition (appreciation, acknowledgment, reward, praise, credit)
3. Promotion/Career Growth (advancement, opportunities, career development, growth)
4. Learning/Training (education, skill development, courses, training, learning)
5. Office Politics (favoritism, bias, unfair treatment, cliques, politics)

For each theme, provide:
- The percentage of responses that mention this theme (round to nearest 1%)
- Whether it's mentioned positively or negatively

Return your analysis in this JSON format:
{
    "team_support_percent": 20,
    "recognition_percent": 15,
    "promotion_percent": 25,
    "learning_percent": 30,
    "politics_percent": 12,
    "analysis_notes": "Brief summary of findings"
}
"""

        result = self._analyze_with_openai(prompt, all_responses)

        # Extract percentages with fallbacks
        return {
            "team_support_percent": result.get("team_support_percent", 0),
            "recognition_percent": result.get("recognition_percent", 0),
            "promotion_percent": result.get("promotion_percent", 0),
            "learning_percent": result.get("learning_percent", 0),
            "politics_percent": result.get("politics_percent", 0),
            "analysis_notes": result.get("analysis_notes", "Analysis completed")
        }

    def _analyze_sentiment_distribution(self, all_responses: List[str]) -> Dict[str, float]:
        """Analyze sentiment distribution using OpenAI"""

        prompt = """
Analyze the sentiment of these employee survey responses and provide percentages for:

1. Overall Positive Sentiment: What percentage of responses show positive sentiment?
2. Strong Positive: What percentage show very positive/enthusiastic sentiment?
3. Strong Negative: What percentage show very negative/dissatisfied sentiment?
4. Team Culture Strength: What percentage indicate strong, positive team culture?

Return your analysis in this JSON format:
{
    "overall_positive_percent": 55,
    "strong_positive_percent": 25,
    "strong_negative_percent": 8,
    "team_culture_strength_percent": 35,
    "sentiment_summary": "Brief summary of overall sentiment"
}
"""

        result = self._analyze_with_openai(prompt, all_responses)

        return {
            "overall_positive_percent": result.get("overall_positive_percent", 50),
            "strong_positive_percent": result.get("strong_positive_percent", 20),
            "strong_negative_percent": result.get("strong_negative_percent", 10),
            "team_culture_strength_percent": result.get("team_culture_strength_percent", 30),
            "sentiment_summary": result.get("sentiment_summary", "Analysis completed")
        }

    def _analyze_themes_with_openai(self, all_responses: List[str]) -> List[Dict[str, Any]]:
        """Discover and analyze themes using OpenAI"""

        prompt = """
Analyze these employee survey responses and identify the top 5-7 themes or concerns.

For each theme, provide:
1. Theme name/title
2. What percentage of responses relate to this theme
3. Average sentiment for this theme (positive/neutral/negative)
4. Key insights or patterns
5. 2-3 representative quotes

Return your analysis in this JSON format:
{
    "themes": [
        {
            "theme_name": "Work-Life Balance",
            "percentage": 25,
            "sentiment": "mixed",
            "insights": "Employees want more flexibility",
            "representative_quotes": ["quote1", "quote2"]
        }
    ]
}
"""

        result = self._analyze_with_openai(prompt, all_responses)

        if "themes" in result:
            return result["themes"]
        else:
            # Fallback theme structure
            return [
                {
                    "theme_name": "General Feedback",
                    "percentage": 100,
                    "sentiment": "mixed",
                    "insights": "Various employee feedback collected",
                    "representative_quotes": all_responses[:2] if all_responses else []
                }
            ]

    def _analyze_cohorts_with_openai(self, cohort_data: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Analyze different employee cohorts using OpenAI"""

        cohort_results = []

        for cohort_name, responses in cohort_data.items():
            if len(responses) < 2:
                continue

            prompt = f"""
Analyze the survey responses from {cohort_name} employees and provide:

1. Overall sentiment percentage for this cohort
2. Top 3 themes/concerns for this cohort
3. How this cohort compares to typical employee feedback
4. Key insights specific to this demographic

Return analysis in this JSON format:
{{
    "cohort_name": "{cohort_name}",
    "sentiment_percent": 65,
    "top_themes": ["theme1", "theme2", "theme3"],
    "key_insights": "Summary of findings for this cohort"
}}
"""

            result = self._analyze_with_openai(prompt, responses)

            cohort_result = {
                "cohort_name": cohort_name,
                "employee_count": len(responses),
                "avg_sentiment": result.get("sentiment_percent", 50),
                "common_themes": result.get("top_themes", ["General feedback"]),
                "insights": result.get("key_insights", f"Analysis for {cohort_name} completed")
            }

            cohort_results.append(cohort_result)

        return sorted(cohort_results, key=lambda x: x['employee_count'], reverse=True)

    def _generate_openai_insights(self, responses: List[str], themes: List[str], sentiment_analysis: Dict) -> List[str]:
        """Generate comprehensive insights using OpenAI"""
        try:
            # Prepare summary data for insight generation
            response_sample = responses[:10]  # Sample for context

            prompt = f"""Based on this employee survey analysis, generate 3-5 key actionable insights:

SURVEY DATA SUMMARY:
- Total responses analyzed: {len(responses)}
- Overall positive sentiment: {sentiment_analysis.get('overall_positive_percent', 0)}%
- Strong positive responses: {sentiment_analysis.get('strong_positive_percent', 0)}%
- Strong negative responses: {sentiment_analysis.get('strong_negative_percent', 0)}%
- Team culture strength: {sentiment_analysis.get('team_culture_strength_percent', 0)}%

KEY THEMES DISCOVERED:
{chr(10).join(f"- {theme}" for theme in themes[:5])}

SAMPLE RESPONSES:
{chr(10).join(f"- {response[:100]}..." for response in response_sample)}

Generate 3-5 specific, actionable insights that would be valuable for HR and management. Focus on:
1. What's working well that should be maintained
2. Key areas for improvement with specific recommendations
3. Any concerning patterns that need immediate attention
4. Opportunities for employee engagement improvements

Format as a list of clear, actionable insights."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )

            # Parse insights from response
            content = response.choices[0].message.content.strip()
            insights = [
                line.strip().lstrip('- ').lstrip('• ').lstrip('* ')
                for line in content.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

            # Filter to get meaningful insights
            meaningful_insights = [
                insight for insight in insights
                if len(insight) > 20 and not insight.lower().startswith('here are')
            ]

            return meaningful_insights[:5]  # Return top 5 insights

        except Exception as e:
            print(f"Error generating insights: {e}")
            return [
                f"Overall sentiment shows {sentiment_analysis.get('overall_positive_percent', 0)}% positive responses",
                f"Key themes include: {', '.join(themes[:3])}",
                "Further analysis recommended for detailed insights"
            ]

    def _generate_vector_insights_with_openai(self, vector_summary: Dict) -> List[str]:
        """Generate insights using OpenAI based on vector analysis results"""
        try:
            prompt = f"""Based on this comprehensive vector-based employee survey analysis, generate 4-6 key actionable insights:

VECTOR ANALYSIS SUMMARY:
- Total responses analyzed: {vector_summary['total_responses']}
- Analysis method: Vector embeddings + similarity search

SENTIMENT METRICS (Vector-derived):
- Overall positive sentiment: {vector_summary['sentiment_metrics']['overall_positive']}%
- Strong positive responses: {vector_summary['sentiment_metrics']['strong_positive']}%
- Strong negative responses: {vector_summary['sentiment_metrics']['strong_negative']}%
- Team culture strength: {vector_summary['sentiment_metrics']['team_culture_strength']}%

THEME ANALYSIS (Vector semantic search):
- Team support mentions: {vector_summary['theme_percentages']['team_support']}%
- Recognition requests: {vector_summary['theme_percentages']['recognition']}%
- Promotion concerns: {vector_summary['theme_percentages']['promotion']}%
- Learning mentions: {vector_summary['theme_percentages']['learning']}%
- Politics concerns: {vector_summary['theme_percentages']['politics']}%

DISCOVERED THEMES (Vector clustering):
{chr(10).join(f"- {theme}" for theme in vector_summary['discovered_themes'][:5])}

COHORT INSIGHTS:
{chr(10).join(f"- {cohort['cohort_name']}: {cohort['sentiment_score']}% positive" for cohort in vector_summary['cohort_insights'][:5])}

Generate 4-6 specific, actionable insights for HR and management based on this vector analysis. Focus on:
1. Patterns identified through semantic similarity
2. What the clustering analysis reveals about employee concerns
3. Cohort-specific recommendations
4. Data-driven action items based on the vector metrics
5. Areas where the semantic analysis shows strong signals

Format as clear, actionable insights."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )

            # Parse insights from response
            content = response.choices[0].message.content.strip()
            insights = [
                line.strip().lstrip('- ').lstrip('• ').lstrip('* ')
                for line in content.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

            # Filter to get meaningful insights
            meaningful_insights = [
                insight for insight in insights
                if len(insight) > 20 and not insight.lower().startswith('here are')
            ]

            return meaningful_insights[:6]  # Return top 6 insights

        except Exception as e:
            print(f"Error generating vector insights: {e}")
            return [
                f"Vector analysis shows {vector_summary['sentiment_metrics']['overall_positive']}% overall positive sentiment",
                f"Semantic search identified {vector_summary['theme_percentages']['team_support']}% team support mentions",
                f"Clustering analysis discovered {len(vector_summary['discovered_themes'])} key themes",
                "Vector-based analysis suggests further investigation into cohort differences"
            ]

    def _calculate_sentiment_score(self, embedding: np.ndarray) -> float:
        if self.sentiment_vectors is None:
            return 0.5  # Neutral fallback

        # Calculate similarities to positive and negative references
        pos_similarity = cosine_similarity(
            embedding.reshape(1, -1),
            self.sentiment_vectors['positive'].reshape(1, -1)
        )[0][0]

        neg_similarity = cosine_similarity(
            embedding.reshape(1, -1),
            self.sentiment_vectors['negative'].reshape(1, -1)
        )[0][0]

        # Convert similarities to 0-1 scale (sentiment score)
        # Higher positive similarity = higher sentiment score
        sentiment_score = (pos_similarity + 1) / \
            ((pos_similarity + 1) + (neg_similarity + 1))

        return max(0.0, min(1.0, sentiment_score))  # Clamp to [0, 1]

    def _semantic_search(self, query_text: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Perform semantic search for responses similar to query"""
        if self.index is None:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query_text)
        if query_embedding is None:
            return []

        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        # Search in Faiss index
        similarities, indices = self.index.search(
            query_normalized.reshape(1, -1).astype('float32'),
            min(top_k, self.index.ntotal)
        )

        return list(zip(indices[0], similarities[0]))

    def _discover_themes_clustering(self, num_clusters: int = 8) -> List[Dict[str, Any]]:
        """Discover themes using K-means clustering on embeddings"""
        if self.embeddings_cache is None or len(self.embeddings_cache) < num_clusters:
            return []

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings_cache)

        themes = []

        for cluster_id in range(num_clusters):
            # Get responses in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_responses = [
                self.metadata[i].response_text for i in cluster_indices]

            if len(cluster_responses) < 2:  # Skip small clusters
                continue

            # Calculate average sentiment for this cluster
            cluster_embeddings = self.embeddings_cache[cluster_indices]
            sentiments = [self._calculate_sentiment_score(
                emb) for emb in cluster_embeddings]
            avg_sentiment = np.mean(sentiments)

            # Get representative responses (closest to cluster center)
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            representative_text = self.metadata[closest_idx].response_text

            theme = {
                'theme_id': f'cluster_{cluster_id}',
                'response_count': len(cluster_responses),
                'avg_sentiment': round(avg_sentiment, 3),
                'representative_text': representative_text,
                'sample_responses': cluster_responses[:3]  # Top 3 examples
            }
            themes.append(theme)

        # Sort by response count (most common themes first)
        themes.sort(key=lambda x: x['response_count'], reverse=True)
        return themes

    def analyze_comprehensive(self) -> VectorAnalysisResult:
        """Perform comprehensive vector + OpenAI analysis with percentage outputs"""
        if self.embeddings_cache is None:
            raise ValueError(
                "No embeddings available. Run process_survey_data first.")

        print("🔄 Performing hybrid vector + OpenAI analysis...")
        print(
            f"📊 Processing {len(self.embeddings_cache)} vectorized responses...")

        # 1. Vector-based sentiment analysis using embeddings
        print("😊 Analyzing sentiment using vector similarities...")
        all_sentiments = [
            self._calculate_sentiment_score(emb) for emb in self.embeddings_cache
        ]
        overall_positive_sentiment = round(np.mean(all_sentiments) * 100, 1)

        # Strong positive/negative percentages
        strong_positive_count = sum(1 for s in all_sentiments if s > 0.8)
        strong_negative_count = sum(1 for s in all_sentiments if s < 0.2)
        strong_positive_percent = round(
            (strong_positive_count / len(all_sentiments)) * 100, 1)
        strong_negative_percent = round(
            (strong_negative_count / len(all_sentiments)) * 100, 1)

        # 2. Vector-based semantic searches for themes
        print("🔍 Performing vector semantic searches for themes...")

        # Team support analysis
        team_support_results = self._semantic_search(
            "team support collaboration help assistance mentorship guidance", 50
        )
        # Lowered from 0.7 to 0.4
        team_support_mentions = len(
            [r for r in team_support_results if r[1] > 0.4])
        team_support_mentions_percent = round(
            (team_support_mentions / len(self.embeddings_cache)) * 100, 1)
        print(
            f"🔍 Team support: Found {team_support_mentions} matches (top scores: {[round(r[1], 3) for r in team_support_results[:5]]})")

        # Recognition analysis
        recognition_results = self._semantic_search(
            "recognition appreciation reward praise acknowledgment credit", 50
        )
        # Lowered from 0.7 to 0.4
        recognition_requests = len(
            [r for r in recognition_results if r[1] > 0.4])
        recognition_requests_percent = round(
            (recognition_requests / len(self.embeddings_cache)) * 100, 1)
        print(
            f"🔍 Recognition: Found {recognition_requests} matches (top scores: {[round(r[1], 3) for r in recognition_results[:5]]})")

        # Promotion analysis
        promotion_results = self._semantic_search(
            "promotion career advancement growth opportunity progress", 50
        )
        # Lowered from 0.7 to 0.4
        promotion_concerns = len([r for r in promotion_results if r[1] > 0.4])
        promotion_concerns_percent = round(
            (promotion_concerns / len(self.embeddings_cache)) * 100, 1)

        # Learning analysis
        learning_results = self._semantic_search(
            "learning training development education skill course workshop", 50
        )
        # Lowered from 0.7 to 0.4
        learning_mentions = len([r for r in learning_results if r[1] > 0.4])
        learning_mentions_percent = round(
            (learning_mentions / len(self.embeddings_cache)) * 100, 1)

        # Politics analysis
        politics_results = self._semantic_search(
            "politics favoritism bias unfair clique gossip drama conflict", 50
        )
        # Lowered from 0.7 to 0.4
        politics_concerns = len([r for r in politics_results if r[1] > 0.4])
        politics_concerns_percent = round(
            (politics_concerns / len(self.embeddings_cache)) * 100, 1)

        # Team culture strength
        culture_results = self._semantic_search(
            "culture values collaboration belonging trust teamwork unity", 50
        )
        culture_positive_count = sum(1 for idx, sim in culture_results
                                     # Lowered from 0.7 to 0.4
                                     if sim > 0.4 and self._calculate_sentiment_score(self.embeddings_cache[idx]) > 0.6)
        team_culture_strength = round(
            (culture_positive_count / len(self.embeddings_cache)) * 100, 1)

        # 3. Vector-based theme discovery through clustering
        print("🎯 Discovering themes through vector clustering...")
        themes = self._discover_themes_clustering()

        # 4. Vector-based cohort analysis
        print("👥 Performing vector-based cohort analysis...")
        cohort_analysis = self._analyze_cohorts()

        # 5. Use OpenAI for high-level insights and interpretation
        print("💡 Generating OpenAI insights from vector analysis...")

        # Prepare vector analysis summary for OpenAI
        vector_summary = {
            "total_responses": len(self.embeddings_cache),
            "sentiment_metrics": {
                "overall_positive": overall_positive_sentiment,
                "strong_positive": strong_positive_percent,
                "strong_negative": strong_negative_percent,
                "team_culture_strength": team_culture_strength
            },
            "theme_percentages": {
                "team_support": team_support_mentions_percent,
                "recognition": recognition_requests_percent,
                "promotion": promotion_concerns_percent,
                "learning": learning_mentions_percent,
                "politics": politics_concerns_percent
            },
            "discovered_themes": themes,
            "cohort_insights": cohort_analysis
        }

        insights = self._generate_vector_insights_with_openai(vector_summary)

        # Create result with vector-derived metrics using ranges and analysis
        result = VectorAnalysisResult(
            overall_positive_sentiment=self._create_metric_with_analysis(
                int(overall_positive_sentiment), len(self.embeddings_cache),
                "overall_positive_sentiment", is_percentage=True
            ),
            team_support_mentions=self._create_metric_with_analysis(
                team_support_mentions, len(
                    self.embeddings_cache), "team_support_mentions"
            ),
            recognition_requests=self._create_metric_with_analysis(
                recognition_requests, len(
                    self.embeddings_cache), "recognition_requests"
            ),
            promotion_concerns=self._create_metric_with_analysis(
                promotion_concerns, len(
                    self.embeddings_cache), "promotion_concerns"
            ),
            strong_positive_percent=self._create_metric_with_analysis(
                int(strong_positive_percent), len(self.embeddings_cache),
                "strong_positive_percent", is_percentage=True
            ),
            learning_mentions=self._create_metric_with_analysis(
                learning_mentions, len(
                    self.embeddings_cache), "learning_mentions"
            ),
            politics_concerns=self._create_metric_with_analysis(
                politics_concerns, len(
                    self.embeddings_cache), "politics_concerns"
            ),
            team_culture_strength=self._create_metric_with_analysis(
                int(team_culture_strength), len(self.embeddings_cache),
                "team_culture_strength", is_percentage=True
            ),
            strong_negative_percent=self._create_metric_with_analysis(
                int(strong_negative_percent), len(self.embeddings_cache),
                "strong_negative_percent", is_percentage=True
            ),
            themes=themes,
            cohort_analysis=cohort_analysis,
            insights=insights
        )

        print("✅ Hybrid vector + OpenAI analysis completed")
        return result

    def _analyze_cohorts(self) -> List[Dict[str, Any]]:
        """Analyze different employee cohorts using vector analysis"""
        cohorts = {}

        # Group by gender and tenure
        for i, metadata in enumerate(self.metadata):
            gender_key = f"Gender: {metadata.gender}"
            tenure_key = f"Tenure: {metadata.tenure}"

            for key in [gender_key, tenure_key]:
                if key not in cohorts:
                    cohorts[key] = {
                        'embeddings': [],
                        'metadata': []
                    }
                cohorts[key]['embeddings'].append(self.embeddings_cache[i])
                cohorts[key]['metadata'].append(metadata)

        cohort_results = []

        for cohort_name, cohort_data in cohorts.items():
            if len(cohort_data['embeddings']) < 2:  # Skip small cohorts
                continue

            embeddings = np.array(cohort_data['embeddings'])

            # Calculate cohort sentiment
            sentiments = [self._calculate_sentiment_score(
                emb) for emb in embeddings]
            avg_sentiment = round(np.mean(sentiments), 3)

            # Find top themes for this cohort using mini-clustering
            if len(embeddings) >= 3:
                n_clusters = min(3, len(embeddings) // 2)
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)

                themes = []
                for cluster_id in range(n_clusters):
                    cluster_mask = labels == cluster_id
                    cluster_responses = [
                        cohort_data['metadata'][j].response_text
                        for j in range(len(cohort_data['metadata']))
                        if cluster_mask[j]
                    ]
                    if len(cluster_responses) > 0:
                        # Get most representative response
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        cluster_embeddings = embeddings[cluster_mask]
                        distances = np.linalg.norm(
                            cluster_embeddings - cluster_center, axis=1)
                        closest_idx = np.where(cluster_mask)[
                            0][np.argmin(distances)]
                        representative = cohort_data['metadata'][closest_idx].response_text

                        themes.append(f"Theme: {representative[:50]}...")
            else:
                themes = ["Insufficient data for theme analysis"]

            cohort_result = {
                'cohort_name': cohort_name,
                'employee_count': len(cohort_data['embeddings']),
                'avg_sentiment': avg_sentiment,
                'common_themes': themes[:3]  # Top 3 themes
            }
            cohort_results.append(cohort_result)

        return sorted(cohort_results, key=lambda x: x['employee_count'], reverse=True)

    def _create_metric_with_analysis(self, exact_count: int, total_responses: int,
                                     metric_type: str, is_percentage: bool = False) -> MetricWithAnalysis:
        """Create a MetricWithAnalysis object with accurate ranges and descriptive analysis"""

        if is_percentage:
            # For percentage metrics, show exact percentage with analysis
            percentage = exact_count
            range_estimate = f"{percentage}%"
        else:
            # For count metrics, create accurate ranges
            percentage = round((exact_count / total_responses) * 100, 1)

            # Create realistic ranges based on dataset size
            if total_responses < 50:
                # 8% range for small datasets
                range_width = max(2, int(total_responses * 0.08))
            elif total_responses < 200:
                # 5% range for medium datasets
                range_width = max(5, int(total_responses * 0.05))
            else:
                # 3% range for large datasets
                range_width = max(10, int(total_responses * 0.03))

            range_start = max(0, exact_count - range_width // 2)
            range_end = exact_count + range_width // 2

            if range_start == range_end:
                range_estimate = str(exact_count)
            else:
                range_estimate = f"{range_start}-{range_end}"

        # Generate analysis and indicator based on metric type and values
        analysis_map = {
            "overall_positive_sentiment": {
                "analysis": f"{'Above average' if exact_count > 50 else 'Below average'} satisfaction",
                "indicator": "Positive indicator" if exact_count > 50 else "Needs attention"
            },
            "team_support_mentions": {
                "analysis": f"{'High' if percentage > 15 else 'Moderate' if percentage > 8 else 'Low'} team support frequency",
                "indicator": "Positive indicator" if percentage > 10 else "Monitor closely"
            },
            "recognition_requests": {
                "analysis": f"{'High' if percentage > 12 else 'Moderate' if percentage > 6 else 'Low'} demand for recognition",
                "indicator": "Needs attention" if percentage > 10 else "Acceptable level"
            },
            "promotion_concerns": {
                "analysis": f"{'High' if percentage > 15 else 'Moderate' if percentage > 8 else 'Low'} career advancement concerns",
                "indicator": "Needs attention" if percentage > 12 else "Acceptable level"
            },
            "strong_positive_percent": {
                "analysis": f"{'Excellent' if exact_count > 30 else 'Good' if exact_count > 20 else 'Average'} employee satisfaction",
                "indicator": "Positive indicator" if exact_count > 20 else "Monitor closely"
            },
            "learning_mentions": {
                "analysis": f"{'High' if percentage > 18 else 'Moderate' if percentage > 10 else 'Low'} training engagement",
                "indicator": "Positive indicator" if percentage > 12 else "Monitor closely"
            },
            "politics_concerns": {
                "analysis": f"{'High' if percentage > 12 else 'Moderate' if percentage > 6 else 'Low'} office politics issues",
                "indicator": "Needs attention" if percentage > 10 else "Acceptable level"
            },
            "team_culture_strength": {
                "analysis": f"{'Strong' if exact_count > 30 else 'Moderate' if exact_count > 20 else 'Weak'} organizational culture",
                "indicator": "Positive indicator" if exact_count > 25 else "Needs attention"
            },
            "strong_negative_percent": {
                "analysis": f"{'High' if exact_count > 15 else 'Moderate' if exact_count > 8 else 'Low'} employee dissatisfaction",
                "indicator": "Needs attention" if exact_count > 10 else "Acceptable level"
            }
        }

        metric_analysis = analysis_map.get(metric_type, {
            "analysis": f"{percentage}% of responses",
            "indicator": "Monitor closely"
        })

        return MetricWithAnalysis(
            range_estimate=range_estimate,
            analysis=metric_analysis["analysis"],
            indicator=metric_analysis["indicator"],
            percentage=percentage if not is_percentage else None
        )

    def _generate_insights(self, sentiments: List[float], themes: List[Dict]) -> Dict[str, Any]:
        """Generate actionable insights from the analysis"""
        insights = {
            'sentiment_distribution': {
                'very_positive': sum(1 for s in sentiments if s > 0.8),
                'positive': sum(1 for s in sentiments if 0.6 < s <= 0.8),
                'neutral': sum(1 for s in sentiments if 0.4 <= s <= 0.6),
                'negative': sum(1 for s in sentiments if 0.2 <= s < 0.4),
                'very_negative': sum(1 for s in sentiments if s < 0.2)
            },
            'top_themes': themes[:5],  # Top 5 themes
            'risk_indicators': [],
            'recommendations': []
        }

        # Risk indicators
        if np.mean(sentiments) < 0.4:
            insights['risk_indicators'].append(
                "Overall low satisfaction detected")

        if sum(1 for s in sentiments if s < 0.2) / len(sentiments) > 0.15:
            insights['risk_indicators'].append(
                "High percentage of very negative responses")

        # Recommendations
        if any('recognition' in theme.get('representative_text', '').lower()
               for theme in themes[:3]):
            insights['recommendations'].append("Implement recognition program")

        if any('training' in theme.get('representative_text', '').lower()
               for theme in themes[:3]):
            insights['recommendations'].append(
                "Expand learning and development opportunities")

        return insights

    def save_analysis(self, file_path: str, analysis_result: VectorAnalysisResult):
        """Save analysis results to file"""
        data = {
            'analysis_result': analysis_result.__dict__,
            'metadata_count': len(self.metadata),
            'embeddings_shape': self.embeddings_cache.shape if self.embeddings_cache is not None else None
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"✅ Analysis saved to {file_path}")

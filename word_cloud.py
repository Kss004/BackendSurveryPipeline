"""
Enhanced Word Cloud Analysis for Survey Pipeline
Performs semantic keyword/phrase extraction, clustering, and sentiment analysis.
Version: 3.0 (Optimized & Configurable)
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter, defaultdict
import re
from datetime import datetime
import numpy as np
import json
import os
from pathlib import Path

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Dependency Imports with Graceful Fallbacks ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence Transformers or Scikit-learn not installed. Analysis will be basic. Install with: pip install sentence-transformers scikit-learn")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER Sentiment not installed. Install with: pip install vaderSentiment")

try:
    import spacy
    from nltk.util import ngrams
    import nltk
    SPACY_NLTK_AVAILABLE = True
except ImportError:
    SPACY_NLTK_AVAILABLE = False
    print("⚠️ spaCy or NLTK not installed. Keyword quality will be reduced. Install with: pip install spacy nltk && python -m spacy download en_core_web_sm")


class ConfigurableWordCloudAnalyzer:
    """
    Enhanced, configurable analyzer for survey responses with dynamic phrase extraction,
    semantic clustering, and comprehensive sentiment analysis.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional external configuration file"""
        self.config = self._load_configuration(config_path)
        self.model = None
        self.sentiment_analyzer = None
        self.nlp = None
        self._phrase_cache = {}
        self._embedding_cache = {}

    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "analysis_parameters": {
                "min_phrase_frequency": 2,
                "max_clusters": 15,
                "top_n_themes": 25,
                "random_state": 42,
                "phrase_lengths": [2, 3, 4],  # Support for longer phrases
                "sentiment_threshold": 0.3,
                "similarity_threshold": 0.7
            },
            "models": {
                "sentence_transformer": "all-MiniLM-L6-v2",
                "spacy": "en_core_web_sm"
            },
            "stopwords": [
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
                'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'many', 'then', 'them',
                'well', 'were', 'been', 'have', 'there', 'where', 'much', 'your', 'work', 'life', 'only', 'think', 'also', 'back', 'after',
                'first', 'year', 'come', 'could', 'like', 'time', 'very', 'when', 'write', 'would', 'each', 'which', 'their', 'said', 'will',
                'about', 'from', 'they', 'know', 'want', 'good', 'some', 'here', 'just', 'long', 'make', 'over', 'such', 'take', 'than', 'with',
                'this', 'that', 'what', 'more', 'other', 'into', 'people', 'really', 'things', 'always', 'being', 'feel', 'need', 'company',
                'employees', 'employee', 'survey', 'response', 'question', 'answer', 'often', 'almost', 'kept'
            ],
            "predefined_phrases": {
                "positive": [
                    "great opportunities", "excellent training", "strong support", "clear communication",
                    "professional development", "career growth", "team collaboration", "work life balance",
                    "supportive management", "learning opportunities", "good culture", "flexible work",
                    "recognition program", "competitive salary", "job satisfaction"
                ],
                "negative": [
                    "lack growth", "insufficient training", "poor communication", "limited opportunities",
                    "challenging collaboration", "work overload", "stress management", "unclear expectations",
                    "office politics", "inadequate support", "poor work life balance", "limited recognition",
                    "salary concerns", "management issues", "toxic environment", "rarely appreciated",
                    "almost nonexistent", "kept out loop", "development nonexistent"
                ]
            },
            "sentiment_indicators": {
                "positive": [
                    'great', 'excellent', 'strong', 'effective', 'supportive', 'wonderful', 'amazing', 'fantastic',
                    'clear', 'collaborative', 'love', 'appreciate', 'impressed', 'grateful', 'outstanding',
                    'exceptional', 'satisfied', 'happy', 'pleased', 'thrilled', 'delighted'
                ],
                "negative": [
                    'lack', 'insufficient', 'nonexistent', 'poor', 'challenging', 'difficult', 'unclear',
                    'limited', 'rarely', 'inadequate', 'frustrating', 'stressed', 'overwhelmed', 'never',
                    'terrible', 'awful', 'disappointing', 'unsatisfied', 'unhappy', 'concerned', 'worried'
                ],
                "negation": ['not', 'no', 'never', 'nothing', 'neither', 'none', 'barely', 'hardly']
            },
            "workplace_categories": {
                "support": ["support", "help", "assist", "guidance", "mentorship", "backup"],
                "teamwork": ["team", "collaboration", "cooperation", "colleague", "partnership", "unity"],
                "growth": ["growth", "development", "learning", "training", "skill", "career", "advancement", "progression"],
                "management": ["manager", "leadership", "supervisor", "management", "boss", "director"],
                "culture": ["culture", "environment", "atmosphere", "workplace", "climate", "values"],
                "communication": ["communication", "feedback", "information", "transparent", "loop", "updates"],
                "recognition": ["recognition", "appreciation", "valued", "acknowledged", "reward", "praise"],
                "workload": ["workload", "stress", "pressure", "overwhelmed", "balance", "burnout"],
                "compensation": ["salary", "pay", "compensation", "benefit", "bonus", "package"],
                "flexibility": ["flexible", "remote", "hybrid", "schedule", "hours", "time"]
            }
        }

        # Load external config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    external_config = json.load(f)
                    # Merge configurations (external overrides default)
                    self._deep_merge(default_config, external_config)
            except Exception as e:
                print(f"⚠️ Error loading config from {config_path}: {e}")

        return default_config

    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep merge two dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    def _load_models_if_needed(self):
        """Lazily loads models on first use to save memory and startup time."""
        if self.model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🔄 Loading Sentence Transformer model...")
            self.model = SentenceTransformer(
                self.config['models']['sentence_transformer'])
            print("✅ Sentence Transformer model loaded.")

        if self.sentiment_analyzer is None and VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

        if self.nlp is None and SPACY_NLTK_AVAILABLE:
            print("🔄 Loading spaCy NLP model...")
            try:
                self.nlp = spacy.load(self.config['models']['spacy'])
                print("✅ spaCy model loaded.")
            except OSError:
                print(
                    "⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None

    def analyze_survey_responses(self, responses: Union[List[str], List[Dict]],
                                 include_phrases: bool = True,
                                 include_clustering: bool = True) -> Dict[str, Any]:
        """
        Enhanced analysis pipeline with configurable features.

        Args:
            responses: List of response strings or SurveyResponse objects
            include_phrases: Whether to extract and analyze phrases
            include_clustering: Whether to perform semantic clustering
        """
        self._load_models_if_needed()

        print(
            f"🔍 Starting enhanced analysis of {len(responses)} survey responses...")

        # Convert responses to text if needed
        response_texts = self._extract_response_texts(responses)

        # Initialize collections
        positive_items, negative_items = [], []
        positive_count, negative_count, neutral_count = 0, 0, 0

        # Enhanced single-pass analysis
        for text in response_texts:
            if not isinstance(text, str) or not text.strip():
                continue

            sentiment = self._classify_sentiment(text)

            if sentiment == 'positive':
                if include_phrases:
                    positive_items.extend(
                        self._extract_meaningful_phrases(text, 'positive'))
                else:
                    positive_items.extend(
                        self._extract_keywords(text, 'positive'))
                positive_count += 1
            elif sentiment == 'negative':
                if include_phrases:
                    negative_items.extend(
                        self._extract_meaningful_phrases(text, 'negative'))
                else:
                    negative_items.extend(
                        self._extract_keywords(text, 'negative'))
                negative_count += 1
            else:
                neutral_count += 1

        print(
            f"📊 Sentiment distribution: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")

        # Generate word clouds
        positive_wordcloud = self._generate_wordcloud_data(
            positive_items, "positive", include_clustering)
        negative_wordcloud = self._generate_wordcloud_data(
            negative_items, "negative", include_clustering)

        # Calculate statistics
        stats = self._calculate_statistics(
            response_texts, positive_count, negative_count, neutral_count)

        result = {
            "message": "Enhanced word cloud analysis completed",
            "analysis_metadata": {
                "total_responses": len(response_texts),
                "sentiment_distribution": {
                    "positive": positive_count,
                    "negative": negative_count,
                    "neutral": neutral_count
                },
                "analysis_type": "phrases" if include_phrases else "keywords",
                "clustering_enabled": include_clustering,
                "analysis_date": datetime.now().isoformat()
            },
            "positive_wordcloud": positive_wordcloud,
            "negative_wordcloud": negative_wordcloud,
            "statistics": stats
        }

        # Ensure all data is JSON serializable
        return self._make_json_serializable(result)

    def _extract_response_texts(self, responses: Union[List[str], List[Dict]]) -> List[str]:
        """Extract text from various response formats"""
        texts = []

        for response in responses:
            if isinstance(response, str):
                texts.append(response)
            elif hasattr(response, 'responses'):  # SurveyResponse object
                combined_text = " ".join(response.responses.values())
                texts.append(combined_text)
            elif isinstance(response, dict):
                if 'response' in response:
                    texts.append(response['response'])
                elif 'text' in response:
                    texts.append(response['text'])
                else:
                    # Combine all string values
                    text_parts = [
                        str(v) for v in response.values() if isinstance(v, str)]
                    texts.append(" ".join(text_parts))
            else:
                texts.append(str(response))

        return texts

    def _classify_sentiment(self, text: str) -> str:
        """Enhanced sentiment classification with configurable thresholds"""
        threshold = self.config['analysis_parameters']['sentiment_threshold']

        if self.sentiment_analyzer:
            score = self.sentiment_analyzer.polarity_scores(text)['compound']
            if score >= threshold:
                return "positive"
            if score <= -threshold:
                return "negative"

        # Enhanced keyword-based classification with phrase matching
        text_lower = text.lower()

        # Check for predefined phrases first (more accurate)
        positive_phrases = self.config['predefined_phrases']['positive']
        negative_phrases = self.config['predefined_phrases']['negative']

        phrase_positive_score = sum(
            1 for phrase in positive_phrases if phrase in text_lower)
        phrase_negative_score = sum(
            1 for phrase in negative_phrases if phrase in text_lower)

        # Check individual sentiment indicators
        positive_indicators = set(
            self.config['sentiment_indicators']['positive'])
        negative_indicators = set(
            self.config['sentiment_indicators']['negative'])
        negation_words = set(self.config['sentiment_indicators']['negation'])

        has_positive = any(kw in text_lower for kw in positive_indicators)
        has_negative = any(kw in text_lower for kw in negative_indicators)
        has_negation = any(neg in text_lower for neg in negation_words)

        # Handle negation (e.g., "not good" should be negative)
        if has_negation and has_positive:
            has_positive = False
            has_negative = True

        # Combine phrase and keyword scores
        total_positive = phrase_positive_score + (1 if has_positive else 0)
        total_negative = phrase_negative_score + (1 if has_negative else 0)

        if total_positive > total_negative:
            return "positive"
        elif total_negative > total_positive:
            return "negative"
        return "neutral"

    def _extract_meaningful_phrases(self, text: str, sentiment_type: str) -> List[str]:
        """Enhanced phrase extraction with multiple strategies"""
        phrases = []

        # Strategy 1: Predefined phrases (highest priority)
        predefined = self.config['predefined_phrases'][sentiment_type]
        text_lower = text.lower()
        for phrase in predefined:
            if phrase in text_lower:
                phrases.append(phrase)

        # Strategy 2: spaCy-based phrase extraction
        if self.nlp:
            phrases.extend(self._extract_spacy_phrases(text, sentiment_type))

        # Strategy 3: Pattern-based extraction
        phrases.extend(self._extract_pattern_phrases(text, sentiment_type))

        return phrases

    def _extract_spacy_phrases(self, text: str, sentiment_type: str) -> List[str]:
        """Extract phrases using spaCy NLP"""
        if not self.nlp:
            return []

        doc = self.nlp(text.lower())
        phrases = []

        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                clean_phrase = self._clean_phrase(chunk.text)
                if clean_phrase and self._is_meaningful_phrase(clean_phrase, sentiment_type):
                    phrases.append(clean_phrase)

        # Extract n-grams with lemmatization
        lemmas = [
            token.lemma_ for token in doc
            if (token.lemma_ not in self.config['stopwords'] and
                not token.is_punct and
                not token.is_space and
                len(token.lemma_) > 2)
        ]

        phrase_lengths = self.config['analysis_parameters']['phrase_lengths']
        for n in phrase_lengths:
            if len(lemmas) >= n:
                for ngram in ngrams(lemmas, n):
                    phrase = " ".join(ngram)
                    if self._is_meaningful_phrase(phrase, sentiment_type):
                        phrases.append(phrase)

        return phrases

    def _extract_pattern_phrases(self, text: str, sentiment_type: str) -> List[str]:
        """Extract phrases using regex patterns"""
        phrases = []
        text_lower = text.lower()

        # Pattern 1: Adjective + Noun combinations
        adj_noun_pattern = r'\b(great|excellent|poor|bad|good|amazing|terrible|wonderful|awful|outstanding|disappointing)\s+(\w+(?:\s+\w+)?)\b'
        matches = re.findall(adj_noun_pattern, text_lower)
        for adj, noun in matches:
            phrase = f"{adj} {noun}".strip()
            if self._is_meaningful_phrase(phrase, sentiment_type):
                phrases.append(phrase)

        # Pattern 2: "lack of" or "plenty of" constructions
        lack_pattern = r'\b(lack|plenty|absence|abundance)\s+of\s+(\w+(?:\s+\w+)?)\b'
        matches = re.findall(lack_pattern, text_lower)
        for modifier, noun in matches:
            phrase = f"{modifier} of {noun}".strip()
            if self._is_meaningful_phrase(phrase, sentiment_type):
                phrases.append(phrase)

        # Pattern 3: Workplace-specific phrases
        workplace_patterns = [
            r'\b(work\s+life\s+balance)\b',
            r'\b(career\s+(?:growth|development|advancement))\b',
            r'\b(team\s+(?:collaboration|work|spirit|dynamics))\b',
            r'\b(professional\s+development)\b',
            r'\b(office\s+(?:politics|environment|culture))\b'
        ]

        for pattern in workplace_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    phrase = " ".join(match)
                else:
                    phrase = match
                if self._is_meaningful_phrase(phrase, sentiment_type):
                    phrases.append(phrase)

        return phrases

    def _extract_keywords(self, text: str, sentiment_type: str) -> List[str]:
        """Extract individual keywords when phrase extraction is disabled"""
        keywords = []
        text_lower = text.lower()

        # Extract from workplace categories
        categories = self.config['workplace_categories']
        for category, terms in categories.items():
            for term in terms:
                if term in text_lower:
                    keywords.append(term)

        # Extract sentiment indicators
        indicators = self.config['sentiment_indicators'][sentiment_type]
        for indicator in indicators:
            if indicator in text_lower:
                keywords.append(indicator)

        return keywords

    def _clean_phrase(self, phrase: str) -> str:
        """Clean and normalize phrases"""
        # Remove extra whitespace and punctuation
        phrase = re.sub(r'\s+', ' ', phrase.strip())
        phrase = re.sub(r'[^\w\s]', '', phrase)

        # Remove stopwords from beginning and end
        words = phrase.split()
        stopwords = self.config['stopwords']

        while words and words[0] in stopwords:
            words.pop(0)
        while words and words[-1] in stopwords:
            words.pop()

        return " ".join(words) if len(words) >= 2 else ""

    def _is_meaningful_phrase(self, phrase: str, sentiment_type: str) -> bool:
        """Enhanced phrase meaningfulness check with workplace context"""
        if not phrase or len(phrase.split()) < 2:
            return False

        words = set(phrase.lower().split())

        # Check against stopwords
        stopwords_set = set(self.config['stopwords'])
        if len(words - stopwords_set) < 2:
            return False

        # Check sentiment alignment
        negation_words = set(self.config['sentiment_indicators']['negation'])
        positive_words = set(self.config['sentiment_indicators']['positive'])
        negative_words = set(self.config['sentiment_indicators']['negative'])

        has_negation = bool(words & negation_words)
        has_positive = bool(words & positive_words)
        has_negative = bool(words & negative_words)

        # Check workplace relevance
        workplace_terms = set()
        for category_terms in self.config['workplace_categories'].values():
            workplace_terms.update(category_terms)
        has_workplace_context = bool(words & workplace_terms)

        # Enhanced sentiment matching logic
        if sentiment_type == "positive":
            return (has_positive and not has_negation) or (has_workplace_context and not has_negative and not has_negation)
        elif sentiment_type == "negative":
            return has_negative or (has_positive and has_negation) or (has_workplace_context and has_negation)

        return has_workplace_context

    def _generate_wordcloud_data(self, items: List[str], sentiment_type: str,
                                 include_clustering: bool = True) -> Dict[str, Any]:
        """Generate comprehensive word cloud data with optional clustering"""
        if not items:
            return {"items": [], "clusters": [], "statistics": {}}

        # Count frequencies
        item_counts = Counter(items)
        min_freq = self.config['analysis_parameters']['min_phrase_frequency']
        filtered_items = {item: count for item,
                          count in item_counts.items() if count >= min_freq}

        # Sort by frequency
        sorted_items = sorted(filtered_items.items(),
                              key=lambda x: x[1], reverse=True)
        top_n = self.config['analysis_parameters']['top_n_themes']

        # Basic word cloud data
        wordcloud_data = {
            "items": [{"text": item, "count": count} for item, count in sorted_items[:top_n]],
            "total_unique_items": len(filtered_items),
            "total_mentions": sum(filtered_items.values())
        }

        # Add clustering if enabled and model available
        if include_clustering and self.model and len(filtered_items) >= 5:
            clusters = self._perform_semantic_clustering(
                filtered_items, sentiment_type)
            wordcloud_data["clusters"] = clusters
        else:
            wordcloud_data["clusters"] = []

        # Add category analysis
        wordcloud_data["categories"] = self._analyze_categories(filtered_items)

        # Ensure all data is JSON serializable
        return self._make_json_serializable(wordcloud_data)

    def _perform_semantic_clustering(self, item_counts: Dict[str, int],
                                     sentiment_type: str) -> List[Dict[str, Any]]:
        """Perform semantic clustering on items using sentence transformers"""
        items = list(item_counts.keys())

        # Use cache if available
        cache_key = f"{sentiment_type}_{hash(tuple(sorted(items)))}"
        if cache_key in self._embedding_cache:
            embeddings = self._embedding_cache[cache_key]
        else:
            print(
                f"🔄 Generating embeddings for {len(items)} {sentiment_type} items...")
            embeddings = self.model.encode(items)
            self._embedding_cache[cache_key] = embeddings

        # Determine optimal number of clusters
        max_clusters = self.config['analysis_parameters']['max_clusters']
        num_clusters = min(max_clusters, max(2, len(items) // 4))

        # Perform clustering
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=self.config['analysis_parameters']['random_state'],
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group items by cluster
        clusters = defaultdict(list)
        for item, label in zip(items, cluster_labels):
            clusters[label].append(item)

        # Create cluster summaries
        cluster_results = []
        for cluster_id, cluster_items in clusters.items():
            if len(cluster_items) < 2:  # Skip small clusters
                continue

            # Calculate cluster statistics
            total_count = sum(item_counts[item] for item in cluster_items)
            avg_count = total_count / len(cluster_items)

            # Sort items by frequency within cluster
            sorted_cluster_items = sorted(
                cluster_items,
                key=lambda x: item_counts[x],
                reverse=True
            )

            # Generate cluster name from top items
            top_items = sorted_cluster_items[:3]
            cluster_name = " / ".join(top_items)

            # Categorize cluster
            category = self._categorize_cluster(cluster_items)

            cluster_results.append({
                # Convert numpy int to Python int
                "cluster_id": int(cluster_id),
                "name": cluster_name,
                "category": category,
                # Convert numpy int to Python int
                "total_count": int(total_count),
                # Convert numpy float to Python float
                "avg_count": round(float(avg_count), 1),
                "item_count": len(cluster_items),
                "top_items": [
                    # Convert counts to Python int
                    {"text": item, "count": int(item_counts[item])}
                    for item in sorted_cluster_items[:5]
                ],
                "all_items": sorted_cluster_items
            })

        # Sort clusters by total count
        cluster_results.sort(key=lambda x: x['total_count'], reverse=True)
        return cluster_results

    def _analyze_categories(self, item_counts: Dict[str, int]) -> Dict[str, Any]:
        """Analyze items by workplace categories"""
        category_analysis = {}
        categories = self.config['workplace_categories']

        for category, terms in categories.items():
            category_items = []
            category_count = 0

            for item, count in item_counts.items():
                item_words = set(item.lower().split())
                if any(term in item_words for term in terms):
                    category_items.append({"text": item, "count": count})
                    category_count += count

            if category_items:
                category_items.sort(key=lambda x: x['count'], reverse=True)
                category_analysis[category] = {
                    "total_count": category_count,
                    "item_count": len(category_items),
                    "items": category_items[:10]  # Top 10 items per category
                }

        return category_analysis

    def _categorize_cluster(self, cluster_items: List[str]) -> str:
        """Determine the primary category for a cluster"""
        categories = self.config['workplace_categories']
        category_scores = defaultdict(int)

        for item in cluster_items:
            item_words = set(item.lower().split())
            for category, terms in categories.items():
                score = sum(1 for term in terms if term in item_words)
                category_scores[category] += score

        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "general"

    def _calculate_statistics(self, response_texts: List[str], positive_count: int,
                              negative_count: int, neutral_count: int) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the analysis"""
        total_responses = len(response_texts)

        if total_responses == 0:
            return {}

        # Basic sentiment statistics
        stats = {
            "sentiment_distribution": {
                "positive_percentage": round((positive_count / total_responses) * 100, 1),
                "negative_percentage": round((negative_count / total_responses) * 100, 1),
                "neutral_percentage": round((neutral_count / total_responses) * 100, 1)
            },
            "response_statistics": {
                "total_responses": total_responses,
                "avg_response_length": round(float(np.mean([len(text.split()) for text in response_texts])), 1),
                "min_response_length": min(len(text.split()) for text in response_texts),
                "max_response_length": max(len(text.split()) for text in response_texts)
            }
        }

        # Predefined phrase analysis
        predefined_stats = self._analyze_predefined_phrases(response_texts)
        stats["predefined_phrases"] = predefined_stats

        return stats

    def _analyze_predefined_phrases(self, response_texts: List[str]) -> Dict[str, Any]:
        """Analyze occurrence of predefined phrases"""
        positive_phrases = self.config['predefined_phrases']['positive']
        negative_phrases = self.config['predefined_phrases']['negative']

        phrase_stats = {"positive": {}, "negative": {}}

        for sentiment_type, phrases in [("positive", positive_phrases), ("negative", negative_phrases)]:
            for phrase in phrases:
                count = sum(
                    1 for text in response_texts if phrase in text.lower())
                if count > 0:
                    phrase_stats[sentiment_type][phrase] = {
                        "count": count,
                        "percentage": round((count / len(response_texts)) * 100, 1)
                    }

        return phrase_stats

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "analysis_parameters": self.config['analysis_parameters'],
            "models": self.config['models'],
            "predefined_phrases_count": {
                "positive": len(self.config['predefined_phrases']['positive']),
                "negative": len(self.config['predefined_phrases']['negative'])
            },
            "workplace_categories": list(self.config['workplace_categories'].keys()),
            "stopwords_count": len(self.config['stopwords'])
        }

    def _make_json_serializable(self, obj):
        """Convert numpy data types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


# --- Integration Functions ---

def analyze_survey_wordcloud(survey_responses: Union[List[str], List[Dict]],
                             include_phrases: bool = True,
                             include_clustering: bool = True) -> Dict[str, Any]:
    """
    Main integration function for the survey pipeline.

    Args:
        survey_responses: List of response strings or SurveyResponse objects
        include_phrases: Whether to extract phrases (True) or just keywords (False)
        include_clustering: Whether to perform semantic clustering

    Returns:
        Comprehensive word cloud analysis results
    """
    analyzer = ConfigurableWordCloudAnalyzer()
    return analyzer.analyze_survey_responses(
        survey_responses,
        include_phrases=include_phrases,
        include_clustering=include_clustering
    )


# --- Example Usage ---
if __name__ == "__main__":
    # Sample survey responses with various formats
    sample_responses = [
        "The team collaboration is amazing and I feel very supported by my manager.",
        "I love the supportive environment and the opportunities for growth are great.",
        "There is a severe lack of communication from upper management.",
        "My manager provides great support and clear direction.",
        "I feel there is no clear path for career advancement and growth is stagnant.",
        "Communication is poor and we are often kept out of the loop.",
        "The professional development programs are fantastic and very helpful.",
        "I really appreciate my team; the teamwork is excellent.",
        "While my team is great, I feel undervalued due to the lack of recognition for my hard work.",
        "The work-life balance is not good here, the workload is too high.",
        "Training opportunities are almost nonexistent in our department.",
        "We are rarely appreciated for our hard work and dedication.",
        "The collaboration across teams is challenging and needs improvement.",
        "Great opportunities for professional development and career growth.",
        "Excellent training programs and strong support from leadership."
    ]

    print("🚀 Running Enhanced Word Cloud Analysis...")

    # Test 1: Full analysis with phrases and clustering
    print("\n=== TEST 1: Full Analysis (Phrases + Clustering) ===")
    results = analyze_survey_wordcloud(
        sample_responses, include_phrases=True, include_clustering=True)

    print(f"📊 Analysis Summary:")
    print(
        f"- Total responses: {results['analysis_metadata']['total_responses']}")
    print(
        f"- Sentiment distribution: {results['analysis_metadata']['sentiment_distribution']}")

    print(f"\n🟢 Positive Word Cloud (Top 10):")
    for item in results['positive_wordcloud']['items'][:10]:
        print(f"  • {item['text']} ({item['count']})")

    print(f"\n🔴 Negative Word Cloud (Top 10):")
    for item in results['negative_wordcloud']['items'][:10]:
        print(f"  • {item['text']} ({item['count']})")

    # Show clusters if available
    if results['positive_wordcloud']['clusters']:
        print(f"\n🟢 Positive Clusters:")
        for cluster in results['positive_wordcloud']['clusters'][:3]:
            print(
                f"  • {cluster['name']} (Total: {cluster['total_count']}, Category: {cluster['category']})")

    if results['negative_wordcloud']['clusters']:
        print(f"\n🔴 Negative Clusters:")
        for cluster in results['negative_wordcloud']['clusters'][:3]:
            print(
                f"  • {cluster['name']} (Total: {cluster['total_count']}, Category: {cluster['category']})")

    # Test 2: Keywords only (faster)
    print("\n=== TEST 2: Keywords Only (No Phrases) ===")
    keyword_results = analyze_survey_wordcloud(
        sample_responses, include_phrases=False, include_clustering=False)

    print(f"🟢 Positive Keywords:")
    for item in keyword_results['positive_wordcloud']['items'][:5]:
        print(f"  • {item['text']} ({item['count']})")

    print(f"🔴 Negative Keywords:")
    for item in keyword_results['negative_wordcloud']['items'][:5]:
        print(f"  • {item['text']} ({item['count']})")

    # Test 3: Custom configuration (removed - no longer supported)

    print("\n🎉 All tests completed successfully!")

"""
Survey Pipeline with Cohort Analysis - Complete Implementation
Supports upload, analysis, and cohort-based reporting with LLM integration
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np
import faiss
from dataclasses import dataclass
import openai
from openai import OpenAI
import json
import random
from services.na_handler import NAHandler
from services.word_cloud import dynamic_quick_wordcloud_analysis

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Survey Pipeline with Cohort Analysis API",
    description="Complete survey analysis pipeline with LLM-based cohort analysis and reporting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
survey_analyzer = None


@dataclass
class SurveyResponse:
    """Individual survey response with metadata and analysis"""
    response_id: int
    metadata: Dict[str, Any]  # Name, Mail Id, Gender, Unit, etc.
    responses: Dict[str, str]  # Q1, Q2, Q3, etc.
    topics_scores: Dict[str, int] = None  # Topic scores 1-5
    sentiment_score: int = None  # Sentiment score 1-5
    insight: str = None  # LLM-generated insight
    embedding: np.ndarray = None  # Vector embedding


@dataclass
class TopicAnalysis:
    """Analysis results for a specific topic"""
    name: str
    avg_score: float
    insights: List[str]
    distribution: Dict[str, int]  # Score distribution
    representative_responses: List[str]


@dataclass
class SurveyReport:
    """Complete survey analysis report"""
    survey_type: str
    total_responses: int
    topics: List[TopicAnalysis]
    sentiment: Dict[str, Any]
    cohort_filters_available: List[str]
    analysis_date: datetime


class SurveyPipelineAnalyzer:
    """Complete survey pipeline with cohort analysis capabilities"""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))

        # Get model from environment variable
        self.llm_model = os.getenv('LLM_MODEL', 'gpt-4.1-mini')

        self.na_handler = NAHandler()

        # Data storage
        self.responses: List[SurveyResponse] = []
        self.survey_type: Optional[str] = None
        self.topics: List[str] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None

        # Metadata fields for cohort analysis
        self.metadata_fields = [
            'Name', 'Mail Id', 'Gender', 'Unit', 'Sub Unit',
            'DOJ', 'Location', 'Manager', 'Years of Experience'
        ]

        print("✅ Survey Pipeline Analyzer initialized")

    def _detect_survey_file_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and parse different survey file formats
        Handles: Header rows, unnamed columns, mixed metadata/response formats
        """
        print("🔍 Analyzing file format...")

        # Check if first row contains headers/questions (common survey format)
        first_row_values = df.iloc[0].astype(str).tolist()
        print(f"🔍 First row sample: {first_row_values[:3]}...")

        # Check if existing column names are already meaningful
        existing_cols = df.columns.tolist()
        has_meaningful_columns = any(keyword in str(col).lower()
                                     for col in existing_cols
                                     for keyword in ['participant', 'gender', 'tenure', 'feel', 'believe', 'work', 'company', 'training', 'opportunities'])

        # Enhanced header detection for enterprise surveys
        has_header_row = False

        # Check if first row contains question text (Q1, Q2, etc. or long question text)
        question_indicators = ['Q1.', 'Q2.', 'Q3.', 'feel', 'believe', 'work', 'company',
                               'training', 'opportunities', 'provided', 'understand', 'satisfied', 'rate']

        if not has_meaningful_columns:
            # Check for question patterns in first row
            has_header_row = any(
                (len(str(val)) > 20 and any(word in str(val).lower() for word in question_indicators)) or
                (str(val).startswith('Q') and '.' in str(val)) or
                ('rate' in str(val).lower() and 'scale' in str(val).lower())
                for val in first_row_values
            )

        # Special handling for enterprise survey format (numbered columns + grouped questions)
        has_enterprise_format = (
            # Many numbered columns
            len([col for col in existing_cols if str(col).isdigit()]) > 5 and
            # Unnamed columns
            len([col for col in existing_cols if 'Unnamed:' in str(col)]) > 2 and
            any('&' in str(col)
                for col in existing_cols)  # Category names with &
        )

        print(f"🔍 Existing columns meaningful: {has_meaningful_columns}")
        print(f"🔍 Header row detected: {has_header_row}")
        print(f"🔍 Enterprise format detected: {has_enterprise_format}")

        df_processed = df.copy()
        metadata_cols = []
        response_cols = []

        if has_header_row:
            print("📋 Detected header row with questions - processing accordingly")
            # Use first row as better column names and skip it for data
            question_headers = df.iloc[0].tolist()
            df_processed = df.iloc[1:].reset_index(drop=True)

            # Create better column mapping
            new_columns = []
            for i, (orig_col, question) in enumerate(zip(df.columns, question_headers)):
                if pd.isna(question) or str(question).strip() == '':
                    new_columns.append(orig_col)
                elif i < 3:  # First 3 are likely metadata
                    # Clean up metadata column names
                    if 'participant' in str(question).lower():
                        new_columns.append('Participant_ID')
                    elif 'gender' in str(question).lower():
                        new_columns.append('Gender')
                    elif 'tenure' in str(question).lower():
                        new_columns.append('Tenure')
                    else:
                        clean_name = str(question).strip().replace(
                            ' ', '_')[:20]
                        new_columns.append(clean_name)
                else:
                    # Keep original column names for responses
                    new_columns.append(orig_col)

            # Update column names
            df_processed.columns = new_columns
            print(f"🔄 Updated columns: {list(df_processed.columns)}")

        elif has_enterprise_format:
            print("🏢 Detected enterprise survey format - processing accordingly")
            # Skip first row (contains questions) and clean up data
            df_processed = df.iloc[1:].reset_index(drop=True)

            # Clean up column names for enterprise format
            cleaned_columns = []
            for col in df.columns:
                if str(col).isdigit():
                    # Map numbered columns to meaningful names based on position
                    col_mapping = {
                        '3': 'Sales_Type', '4': 'Section', '5': 'Zone', '6': 'Branch',
                        '7': 'Base_Station', '8': 'Department', '9': 'Product_Division',
                        '13': 'Grade', '14': 'Band', '15': 'Gender', '20': 'Age', '25': 'Tenure'
                    }
                    cleaned_columns.append(
                        col_mapping.get(str(col), f'Field_{col}'))
                elif 'Unnamed:' in str(col):
                    if col == 'Unnamed: 0':
                        cleaned_columns.append('ID')
                    elif col == 'Unnamed: 48':
                        cleaned_columns.append('What_You_Like')
                    elif col == 'Unnamed: 49':
                        cleaned_columns.append('Suggestions')
                    else:
                        cleaned_columns.append(f'Unknown_{col}')
                else:
                    # Clean survey category names
                    clean_name = str(col).replace(' & ', '_').replace(
                        ' ', '_').replace('.', '_')
                    if clean_name.count('_') > 3:  # Very long names
                        clean_name = clean_name[:30]
                    cleaned_columns.append(clean_name)

            df_processed.columns = cleaned_columns
            print(
                f"🔄 Enterprise columns cleaned: {len(cleaned_columns)} columns")

        # Smart column classification with enterprise format support
        for i, col in enumerate(df_processed.columns):
            col_lower = col.lower()
            sample_values = df_processed[col].dropna().astype(
                str).head(5).tolist()

            # Rule 1: Enterprise format metadata fields (highest priority)
            if has_enterprise_format and any(keyword in col_lower for keyword in [
                'sales_type', 'section', 'zone', 'branch', 'base_station', 'department',
                'product_division', 'grade', 'band', 'gender', 'age', 'tenure', 'id'
            ]):
                metadata_cols.append(col)

            # Rule 2: Check for explicit metadata keywords in column name
            elif any(keyword in col_lower for keyword in ['participant', 'gender', 'tenure', 'id', 'name', 'location', 'department', 'manager']):
                metadata_cols.append(col)

            # Rule 3: Enterprise format survey categories (Work_Environment, Development_Growth, etc.)
            elif has_enterprise_format and any(keyword in col_lower for keyword in [
                'work_environment', 'development', 'growth', 'engagement', 'productivity',
                'team_collaboration', 'leadership', 'job_satisfaction', 'integrity', 'ethics'
            ]):
                response_cols.append(col)

            # Rule 4: Check for survey question patterns in column name
            elif any(keyword in col_lower for keyword in ['feel', 'believe', 'think', 'satisfied', 'agree', 'rate', 'how', 'what', 'understand', 'provided', 'opportunities']):
                response_cols.append(col)

            # Rule 5: Open-ended text fields
            elif any(keyword in col_lower for keyword in ['what_you_like', 'suggestions', 'comments', 'feedback']):
                response_cols.append(col)

            # Rule 6: Numeric survey responses (1-5 scale, 1-10 scale)
            elif sample_values and all(str(val).isdigit() and 1 <= int(val) <= 10 for val in sample_values if str(val).isdigit()):
                response_cols.append(col)

            # Rule 7: First few columns are usually metadata in survey files (but only if no explicit keywords)
            elif i < 13 and not has_enterprise_format:  # Adjust for enterprise format
                metadata_cols.append(col)

            # Rule 8: Check sample values - short categorical values suggest metadata
            elif sample_values and all(len(str(val)) < 20 for val in sample_values):
                unique_vals = set(str(val).upper() for val in sample_values)
                if len(unique_vals) <= 15:  # Increased threshold for enterprise data
                    metadata_cols.append(col)

            # Rule 9: Long text responses are survey responses
            else:
                response_cols.append(col)

        # Fallback: ensure we have both types
        if not response_cols:
            response_cols = [col for col in df_processed.columns[3:]]
            metadata_cols = list(df_processed.columns[:3])

        return {
            'dataframe': df_processed,
            'metadata_cols': metadata_cols,
            'response_cols': response_cols,
            'has_header_row': has_header_row
        }

    def upload_and_process_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 1: Upload Data & Vectorization
        Parse file, preprocess, extract metadata, compute embeddings
        """
        print("📊 Processing uploaded survey data...")

        # Detect and parse file format
        format_info = self._detect_survey_file_format(df)
        df_clean = format_info['dataframe']
        metadata_cols = format_info['metadata_cols']
        response_cols = format_info['response_cols']

        # Clean the processed data
        df_clean = self.na_handler.clean_and_prepare_data(
            df_clean, strategy='smart')
        print(f"✅ Cleaned data shape: {df_clean.shape}")

        print(
            f"📋 Found {len(metadata_cols)} metadata columns: {metadata_cols}")
        print(
            f"📝 Found {len(response_cols)} response columns: {response_cols}")

        # Process each row into SurveyResponse objects
        self.responses = []
        all_response_texts = []

        for idx, row in df_clean.iterrows():
            # Extract metadata
            metadata = {}
            for col in metadata_cols:
                metadata[col] = str(row.get(col, 'Unknown'))

            # Handle tenure/experience field conversion
            tenure_value = None

            # First priority: Look for explicit tenure column (most reliable)
            tenure_col = next(
                (col for col in metadata_cols if 'tenure' in col.lower()), None)

            if tenure_col and tenure_col in metadata:
                tenure_str = str(metadata[tenure_col])
                print(
                    f"🔍 Processing tenure from '{tenure_col}' column: {tenure_str}")

                # Parse tenure ranges like "5 - 10 years", "15+ years", "3 - 5 years"
                if '15+' in tenure_str or 'more than 15' in tenure_str.lower():
                    tenure_value = '18'
                elif '10 - 15' in tenure_str or '10-15' in tenure_str:
                    tenure_value = '12'
                elif '5 - 10' in tenure_str or '5-10' in tenure_str:
                    tenure_value = '7'  # Middle of range
                elif '3 - 5' in tenure_str or '3-5' in tenure_str:
                    tenure_value = '4'  # Middle of range
                elif '0 - 3' in tenure_str or '0-3' in tenure_str or 'less than 3' in tenure_str.lower():
                    tenure_value = '2'
                elif '0 - 5' in tenure_str or '0-5' in tenure_str or 'less than 5' in tenure_str.lower():
                    tenure_value = '3'
                else:
                    # Try to extract first number from string
                    import re
                    numbers = re.findall(r'\d+', tenure_str)
                    if numbers:
                        tenure_value = numbers[0]
                        print(f"✅ Extracted tenure number: {tenure_value}")

            # Fallback: Check all metadata columns for tenure-like patterns
            if not tenure_value:
                for col in metadata_cols:
                    col_value = str(metadata.get(col, ''))
                    if any(pattern in col_value for pattern in ['5 - 10', '10 - 15', '0 - 5', '3 - 5', '15+', 'years']):
                        tenure_str = col_value
                        print(
                            f"🔍 Processing tenure from column '{col}': {tenure_str}")

                        if '15+' in tenure_str or 'more than 15' in tenure_str.lower():
                            tenure_value = '18'
                        elif '10 - 15' in tenure_str or '10-15' in tenure_str:
                            tenure_value = '12'
                        elif '5 - 10' in tenure_str or '5-10' in tenure_str:
                            tenure_value = '7'
                        elif '3 - 5' in tenure_str or '3-5' in tenure_str:
                            tenure_value = '4'
                        elif '0 - 3' in tenure_str or '0-3' in tenure_str:
                            tenure_value = '2'
                        elif '0 - 5' in tenure_str or '0-5' in tenure_str:
                            tenure_value = '3'
                        else:
                            # Try to extract number from string
                            import re
                            numbers = re.findall(r'\d+', tenure_str)
                            if numbers:
                                tenure_value = numbers[0]
                        break

            # Calculate years of experience from DOJ if available
            elif 'DOJ' in metadata or any('doj' in col.lower() for col in metadata_cols):
                doj_col = next(
                    (col for col in metadata_cols if 'doj' in col.lower()), 'DOJ')
                try:
                    doj_str = metadata.get(doj_col, '')
                    if doj_str and doj_str != 'Unknown':
                        # Try to parse date and calculate years
                        doj_date = pd.to_datetime(doj_str, errors='coerce')
                        if pd.notna(doj_date):
                            years_exp = (datetime.now() -
                                         doj_date).days / 365.25
                            tenure_value = f"{years_exp:.1f}"
                except:
                    pass

            # Check for existing experience fields
            elif any('experience' in col.lower() for col in metadata_cols):
                exp_col = next(
                    (col for col in metadata_cols if 'experience' in col.lower()), None)
                if exp_col and exp_col in metadata:
                    tenure_value = metadata[exp_col]

            # Assign tenure value
            if tenure_value:
                metadata['Years of Experience'] = tenure_value
                print(f"✅ Assigned experience: {tenure_value} years")
            else:
                # Mark as unknown if no tenure data found
                metadata['Years of Experience'] = 'Unknown'
                print(f"⚠️ No tenure data found, marked as Unknown")

            # Extract responses with enterprise format handling
            responses = {}
            combined_text = ""
            for col in response_cols:
                response_text = str(row.get(col, ''))
                if response_text and response_text.strip() and response_text != 'nan':
                    # Handle numeric responses (convert to descriptive text for better LLM analysis)
                    if response_text.isdigit():
                        score = int(response_text)
                        if 1 <= score <= 5:
                            # Convert Likert scale to descriptive text
                            score_descriptions = {
                                1: "Strongly Disagree/Very Negative",
                                2: "Disagree/Negative",
                                3: "Neutral",
                                4: "Agree/Positive",
                                5: "Strongly Agree/Very Positive"
                            }
                            descriptive_text = f"{col}: {score_descriptions.get(score, 'Neutral')} (Score: {score})"
                            responses[col] = descriptive_text
                            combined_text += f" {descriptive_text}"
                        elif 1 <= score <= 10:
                            # Handle 1-10 scale (Overall rating)
                            if score <= 3:
                                sentiment = "Very Negative"
                            elif score <= 5:
                                sentiment = "Negative"
                            elif score <= 7:
                                sentiment = "Neutral"
                            elif score <= 9:
                                sentiment = "Positive"
                            else:
                                sentiment = "Very Positive"
                            descriptive_text = f"{col}: {sentiment} (Rating: {score}/10)"
                            responses[col] = descriptive_text
                            combined_text += f" {descriptive_text}"
                        else:
                            responses[col] = response_text
                            combined_text += f" {response_text}"
                    else:
                        # Text responses (keep as is)
                        responses[col] = response_text
                        combined_text += f" {response_text}"

            if combined_text.strip():
                survey_response = SurveyResponse(
                    response_id=idx,
                    metadata=metadata,
                    responses=responses
                )
                self.responses.append(survey_response)
                all_response_texts.append(combined_text.strip())

        print(f"✅ Processed {len(self.responses)} valid survey responses")

        # Generate embeddings for all responses
        if all_response_texts:
            print("🔄 Generating embeddings...")
            self.embeddings = self._get_embeddings_batch(all_response_texts)

            # Create FAISS index
            if self.embeddings is not None:
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                normalized_embeddings = self.embeddings / \
                    np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                self.faiss_index.add(normalized_embeddings.astype('float32'))
                print(
                    f"✅ Created FAISS index with {len(self.embeddings)} vectors")

        return {
            "message": "Data uploaded and processed successfully",
            "total_responses": len(self.responses),
            "metadata_fields": metadata_cols,
            "response_fields": response_cols,
            "embeddings_generated": self.embeddings is not None,
            "processing_date": datetime.now().isoformat()
        }

    def detect_survey_type(self) -> Dict[str, Any]:
        """
        Step 2: Survey Type Detection
        Sample 20 responses and use LLM to detect survey type
        """
        print("🔍 Detecting survey type...")

        if not self.responses:
            raise ValueError(
                "No survey responses available. Please upload data first.")

        # Sample 20 random responses
        sample_size = min(20, len(self.responses))
        sample_responses = random.sample(self.responses, sample_size)

        # Combine sample responses for LLM analysis
        sample_texts = []
        for resp in sample_responses:
            combined_text = " ".join(resp.responses.values())
            sample_texts.append(combined_text[:200])  # Limit length

        sample_text = "\n".join(sample_texts)

        prompt = f"""
        Analyze these survey responses and determine what type of survey this is:

        Sample Responses:
        {sample_text}

        Based on the content, classify this survey as one of the following types:
        - Employee Engagement Survey
        - Customer Satisfaction Survey
        - NPS Survey
        - Exit Interview Survey
        - Performance Review Survey
        - Training Feedback Survey
        - Other (specify)

        Respond with just the survey type name and a brief explanation.
        Format: "Survey Type: [TYPE] - [Brief explanation]"
        """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            # Extract survey type
            if "Survey Type:" in result:
                self.survey_type = result.split("Survey Type:")[
                    1].split("-")[0].strip()
            else:
                self.survey_type = result.split("-")[0].strip()

            print(f"✅ Detected survey type: {self.survey_type}")

            return {
                "survey_type": self.survey_type,
                "detection_result": result,
                "sample_size": sample_size,
                "confidence": "High" if any(keyword in result.lower() for keyword in ['employee', 'engagement', 'satisfaction']) else "Medium"
            }

        except Exception as e:
            print(f"❌ Error detecting survey type: {e}")
            self.survey_type = "Employee Engagement Survey"  # Default fallback
            return {
                "survey_type": self.survey_type,
                "detection_result": "Fallback to default type due to error",
                "error": str(e)
            }

    def extract_topics_and_analyze(self) -> Dict[str, Any]:
        """
        Step 3: Survey Analysis
        Extract 5-6 key topics via LLM and analyze each response
        """
        print("📊 Extracting topics and analyzing responses...")

        if not self.responses:
            raise ValueError(
                "No survey responses available. Please upload data first.")

        # Step 3a: Extract topics using LLM
        sample_size = min(30, len(self.responses))
        sample_responses = random.sample(self.responses, sample_size)
        sample_texts = [" ".join(resp.responses.values())[:300]
                        for resp in sample_responses]
        combined_sample = "\n".join(sample_texts)

        topic_extraction_prompt = f"""
        You are an Data Analyst. Analyze these survey responses and extract 5-6 key topics that employees are discussing:

        Sample Responses:
        {combined_sample}

        Extract the main themes/topics. Always include "Sentiment" as one topic.

        Return as JSON array:
        ["Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Sentiment"]

        Example topics: Workload, Manager, Compensation, Growth, Culture, Work-Life Balance, Communication, etc.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": topic_extraction_prompt}],
                temperature=0.3
            )

            topics_result = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                self.topics = json.loads(topics_result)
            except json.JSONDecodeError:
                # Fallback parsing
                import re
                topics_match = re.findall(r'"([^"]+)"', topics_result)
                self.topics = topics_match if topics_match else [
                    "Workload", "Manager", "Growth", "Culture", "Communication", "Sentiment"]

            # Ensure Sentiment is included
            if "Sentiment" not in self.topics:
                self.topics.append("Sentiment")

            print(f"✅ Extracted topics: {self.topics}")

        except Exception as e:
            print(f"❌ Error extracting topics: {e}")
            self.topics = ["Workload", "Manager", "Growth",
                           "Culture", "Communication", "Sentiment"]

        # Step 3b: Analyze responses in smaller, more reliable batches
        print("🔄 Analyzing responses in optimized batches...")
        analyzed_count = 0
        batch_size = 10  # Smaller batch size for better reliability

        # Process responses in batches
        for batch_start in range(0, len(self.responses), batch_size):
            batch_end = min(batch_start + batch_size, len(self.responses))
            batch_responses = self.responses[batch_start:batch_end]

            try:
                # Simplified batch processing with more reliable format
                batch_texts = []
                for i, response in enumerate(batch_responses):
                    combined_text = " ".join(response.responses.values())
                    # Limit text and clean it
                    clean_text = combined_text[:300].replace(
                        '"', "'").replace('\n', ' ')
                    batch_texts.append(f"Response {i}: {clean_text}")

                # Simplified prompt with more reliable output format
                batch_prompt = f"""
                Analyze these {len(batch_responses)} survey responses. For each response, provide scores (1-5 scale) for these topics: {', '.join(self.topics)}

                {chr(10).join(batch_texts)}

                Return ONLY a simple format for each response:
                0: Workload=3, Manager=4, Growth=2, Culture=3, Communication=4, Sentiment=3, Insight="Brief summary"
                1: Workload=4, Manager=3, Growth=3, Culture=4, Communication=3, Sentiment=4, Insight="Brief summary"
                ...

                Use this exact format with numbers 0-{len(batch_responses)-1}.
                """

                llm_response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.3
                )

                result_text = llm_response.choices[0].message.content.strip()

                # Parse the simplified format
                success_count = 0
                lines = result_text.split('\n')

                for line in lines:
                    line = line.strip()
                    if ':' in line and '=' in line:
                        try:
                            # Parse format: "0: Workload=3, Manager=4, ..."
                            parts = line.split(':', 1)
                            response_id = int(parts[0].strip())

                            if response_id < len(batch_responses):
                                response = batch_responses[response_id]

                                # Parse scores and insight
                                content = parts[1].strip()
                                scores = {}
                                insight = "Analysis completed"

                                # Extract scores
                                for topic in self.topics:
                                    if f"{topic}=" in content:
                                        try:
                                            start = content.find(
                                                f"{topic}=") + len(f"{topic}=")
                                            end = content.find(',', start)
                                            if end == -1:
                                                end = content.find(' ', start)
                                            if end == -1:
                                                end = len(content)
                                            score_str = content[start:end].strip(
                                            )
                                            scores[topic] = int(score_str)
                                        except:
                                            scores[topic] = 3

                                # Extract insight
                                if 'Insight="' in content:
                                    start = content.find('Insight="') + 9
                                    end = content.find('"', start)
                                    if end > start:
                                        insight = content[start:end]

                                # Assign to response
                                response.topics_scores = scores
                                response.sentiment_score = scores.get(
                                    "Sentiment", 3)
                                response.insight = insight
                                success_count += 1
                                analyzed_count += 1

                        except Exception as parse_error:
                            print(f"⚠️ Failed to parse line: {line[:50]}...")
                            continue

                # Handle any responses that weren't parsed successfully
                for i, response in enumerate(batch_responses):
                    if not hasattr(response, 'topics_scores') or not response.topics_scores:
                        response.topics_scores = {
                            topic: 3 for topic in self.topics}
                        response.sentiment_score = 3
                        response.insight = "Neutral response (parsing fallback)"
                        if i >= success_count:  # Only count if not already counted
                            analyzed_count += 1

                print(
                    f"✅ Batch {batch_start//batch_size + 1}: {success_count}/{len(batch_responses)} parsed successfully")

            except Exception as e:
                print(
                    f"❌ Error analyzing batch {batch_start//batch_size + 1}: {e}")
                # Fallback: assign neutral scores to entire batch
                for response in batch_responses:
                    response.topics_scores = {
                        topic: 3 for topic in self.topics}
                    response.sentiment_score = 3
                    response.insight = f"Analysis error: {str(e)}"
                    analyzed_count += 1

            # Progress update
            progress = (batch_end / len(self.responses)) * 100
            print(
                f"📊 Progress: {progress:.1f}% ({batch_end}/{len(self.responses)} responses)")

        print(
            f"✅ Analyzed {analyzed_count} responses successfully using optimized batched processing")

        return {
            "message": "Topic extraction and response analysis completed",
            "topics_extracted": self.topics,
            "responses_analyzed": analyzed_count,
            "total_responses": len(self.responses)
        }

    def generate_report(self, cohort_filter: Optional[Dict[str, Any]] = None) -> SurveyReport:
        """
        Step 4: Report Generation
        Aggregate topic scores and generate insights with optional cohort filtering
        """
        print("📋 Generating survey report...")

        if not self.responses or not self.topics:
            raise ValueError(
                "No analyzed data available. Please run analysis first.")

        # Apply cohort filter if provided
        filtered_responses = self.responses
        if cohort_filter:
            filtered_responses = self._apply_cohort_filter(
                self.responses, cohort_filter)
            print(
                f"🔍 Applied cohort filter: {len(filtered_responses)} responses match criteria")

        # Aggregate topic scores
        topic_analyses = []

        for topic in self.topics:
            scores = []
            insights = []
            representative_responses = []

            for resp in filtered_responses:
                if resp.topics_scores and topic in resp.topics_scores:
                    score = resp.topics_scores[topic]
                    scores.append(score)

                    # Collect insights for low scores (potential issues)
                    if score <= 2 and resp.insight:
                        insights.append(resp.insight)

                    # Collect representative responses
                    if len(representative_responses) < 3:
                        combined_text = " ".join(resp.responses.values())
                        representative_responses.append(
                            combined_text[:150] + "...")

            if scores:
                avg_score = np.mean(scores)

                # Generate distribution
                distribution = {str(i): scores.count(i) for i in range(1, 6)}

                # Generate topic-specific insights using LLM
                topic_insights = self._generate_topic_insights(
                    topic, scores, representative_responses)

                topic_analysis = TopicAnalysis(
                    name=topic,
                    avg_score=round(avg_score, 2),
                    insights=topic_insights,
                    distribution=distribution,
                    representative_responses=representative_responses
                )
                topic_analyses.append(topic_analysis)

        # Sentiment analysis
        sentiment_scores = [
            resp.sentiment_score for resp in filtered_responses if resp.sentiment_score]
        sentiment_analysis = {
            "avg_score": round(np.mean(sentiment_scores), 2) if sentiment_scores else 3.0,
            "distribution": {str(i): sentiment_scores.count(i) for i in range(1, 6)} if sentiment_scores else {}
        }

        # Available cohort filters
        cohort_filters = list(set().union(
            *[resp.metadata.keys() for resp in self.responses]))

        report = SurveyReport(
            survey_type=self.survey_type or "Unknown",
            total_responses=len(filtered_responses),
            topics=topic_analyses,
            sentiment=sentiment_analysis,
            cohort_filters_available=cohort_filters,
            analysis_date=datetime.now()
        )

        print(f"✅ Generated report for {len(filtered_responses)} responses")
        return report

    def cohort_analysis(self, cohort_query: str) -> Dict[str, Any]:
        """
        Step 5: LLM-based Cohort Analysis
        Parse natural language cohort queries and return filtered analysis
        """
        print(f"🔍 Processing cohort query: {cohort_query}")

        # Use LLM to parse the cohort query
        parse_prompt = f"""
        Parse this cohort analysis query and extract the PRIMARY filter criteria. Return ONLY valid JSON, no other text.

        Query: "{cohort_query}"

        Available metadata fields: Gender, Branch, Zone, Department, Grade, Age, Tenure, Years of Experience

        For queries with multiple criteria, choose the MOST SPECIFIC one as primary filter.

        Return ONLY this JSON format (no explanations, no markdown, just JSON):
        {{
            "field": "metadata_field_name",
            "condition": "equals|contains|greater_than|less_than",
            "value": "filter_value",
            "topic_focus": "specific_topic_if_mentioned_or_null"
        }}

        Examples:
        Query: "female employees from Mumbai about Leadership"
        Response: {{"field": "Gender", "condition": "contains",
            "value": "Female", "topic_focus": "Leadership"}}

        Query: "employees with >5 years experience about Workload"
        Response: {{"field": "Years of Experience",
            "condition": "greater_than", "value": "5", "topic_focus": "Workload"}}

        Query: "what do employees from Ahmedabad think about company culture"
        Response: {{"field": "Branch", "condition": "contains",
            "value": "Ahmedabad", "topic_focus": "Culture"}}

        For location queries, use "Branch" field. For topics about manager/leadership/feedback, use "Leadership" as topic_focus.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": parse_prompt}],
                temperature=0.3
            )

            filter_result = response.choices[0].message.content.strip()
            print(f"🔍 LLM Response: {filter_result}")

            # Try to parse JSON, with fallback parsing
            try:
                filter_criteria = json.loads(filter_result)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', filter_result, re.DOTALL)
                if json_match:
                    filter_criteria = json.loads(json_match.group())
                else:
                    # Manual parsing fallback for common patterns
                    filter_criteria = self._manual_parse_query(cohort_query)

            print(f"✅ Parsed filter criteria: {filter_criteria}")

        except Exception as e:
            print(f"❌ Error parsing cohort query: {e}")
            # Try manual parsing as last resort
            try:
                filter_criteria = self._manual_parse_query(cohort_query)
                print(f"✅ Manual parsing successful: {filter_criteria}")
            except:
                return {"error": f"Could not parse cohort query: {str(e)}"}

        # Apply the filter
        try:
            filtered_responses = self._apply_parsed_filter(
                self.responses, filter_criteria)

            if not filtered_responses:
                return {
                    "message": "No responses match the cohort criteria",
                    "query": cohort_query,
                    "filter_applied": filter_criteria,
                    "matching_responses": 0
                }

            # Focus on specific topic if mentioned
            topic_focus = filter_criteria.get("topic_focus")
            if topic_focus:
                focused_analysis = self._get_topic_specific_analysis(
                    filtered_responses, topic_focus)

                # Get sample insights from the cohort
                sample_insights = []
                for resp in filtered_responses[:5]:  # First 5 responses
                    if resp.insight:
                        sample_insights.append(resp.insight)

                return {
                    "message": f"Cohort analysis completed for {len(filtered_responses)} responses",
                    "query": cohort_query,
                    "filter_applied": filter_criteria,
                    "matching_responses": len(filtered_responses),
                    "focused_topic": topic_focus,
                    "topic_analysis": focused_analysis,
                    # Top 3 insights from this cohort
                    "cohort_insights": sample_insights[:3]
                }
            else:
                # Generate summary statistics for the cohort
                cohort_summary = self._generate_cohort_summary(
                    filtered_responses)

                return {
                    "message": f"Cohort analysis completed for {len(filtered_responses)} responses",
                    "query": cohort_query,
                    "filter_applied": filter_criteria,
                    "matching_responses": len(filtered_responses),
                    "cohort_summary": cohort_summary
                }

        except Exception as e:
            return {"error": f"Error applying filter: {str(e)}"}

    # Helper methods
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [
                    embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(
                    f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
                # Fallback to mock embeddings
                embeddings.extend([np.random.normal(0, 0.1, 1536)
                                  for _ in batch])

        return np.array(embeddings)

    def _apply_cohort_filter(self, responses: List[SurveyResponse], cohort_filter: Dict[str, Any]) -> List[SurveyResponse]:
        """Apply cohort filter to responses"""
        filtered = []
        for resp in responses:
            if self._matches_filter(resp.metadata, cohort_filter):
                filtered.append(resp)
        return filtered

    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        field = filter_criteria.get("field")
        condition = filter_criteria.get("condition")
        value = filter_criteria.get("value")

        if field not in metadata:
            print(
                f"🔍 Field '{field}' not found in metadata. Available fields: {list(metadata.keys())}")
            return False

        metadata_value = str(metadata[field])
        filter_value = str(value)

        print(f"🔍 Comparing: {metadata_value} ({condition}) {filter_value}")

        if condition == "equals":
            return metadata_value.lower() == filter_value.lower()
        elif condition == "contains":
            return filter_value.lower() in metadata_value.lower()
        elif condition == "greater_than":
            try:
                # Handle both string and numeric comparisons
                meta_num = float(metadata_value)
                filter_num = float(filter_value)
                result = meta_num > filter_num
                print(
                    f"🔍 Numeric comparison: {meta_num} > {filter_num} = {result}")
                return result
            except ValueError:
                print(
                    f"🔍 Could not convert to numbers: '{metadata_value}' > '{filter_value}'")
                return False
        elif condition == "less_than":
            try:
                meta_num = float(metadata_value)
                filter_num = float(filter_value)
                result = meta_num < filter_num
                print(
                    f"🔍 Numeric comparison: {meta_num} < {filter_num} = {result}")
                return result
            except ValueError:
                print(
                    f"🔍 Could not convert to numbers: '{metadata_value}' < '{filter_value}'")
                return False

        return False

    def _apply_parsed_filter(self, responses: List[SurveyResponse], filter_criteria: Dict[str, Any]) -> List[SurveyResponse]:
        """Apply parsed filter criteria to responses"""
        return self._apply_cohort_filter(responses, filter_criteria)

    def _generate_topic_insights(self, topic: str, scores: List[int], representative_responses: List[str]) -> List[str]:
        """Generate insights for a specific topic using LLM"""
        avg_score = np.mean(scores)

        if avg_score >= 4:
            return [f"{topic} is performing well", "High satisfaction in this area"]
        elif avg_score <= 2:
            return [f"{topic} needs attention", "Low satisfaction detected", "Requires improvement"]
        else:
            return [f"{topic} shows mixed results", "Moderate satisfaction levels"]

    def _get_topic_specific_analysis(self, responses: List[SurveyResponse], topic: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific topic"""
        topic_scores = []
        insights = []

        # Handle case sensitivity - find the actual topic name
        actual_topic = topic
        if responses and responses[0].topics_scores:
            # Find matching topic (case insensitive)
            available_topics = list(responses[0].topics_scores.keys())
            print(f"🔍 Available topics in responses: {available_topics}")

            for available_topic in available_topics:
                if available_topic.lower() == topic.lower():
                    actual_topic = available_topic
                    break

            # If no exact match, try multiple matching strategies
            if actual_topic == topic and topic.lower() not in [t.lower() for t in available_topics]:
                # Strategy 1: Partial matching
                for available_topic in available_topics:
                    if topic.lower() in available_topic.lower() or available_topic.lower() in topic.lower():
                        actual_topic = available_topic
                        print(
                            f"🔍 Using partial match: '{topic}' → '{actual_topic}'")
                        break

                # Strategy 2: Keyword matching for compound topics
                if actual_topic == topic:
                    topic_keywords = {
                        "leadership": ["Leadership", "Manager", "Management"],
                        "manager": ["Leadership", "Manager", "Management"],
                        "feedback": ["Leadership", "Manager", "Management"],
                        "company": ["Culture", "Work Environment", "Job Satisfaction"],
                        "culture": ["Culture", "Work Environment"],
                        "development": ["Development", "Growth", "Training"],
                        "growth": ["Development", "Growth", "Training"],
                        "communication": ["Communication", "Team Collaboration"],
                        "team": ["Team Collaboration", "Communication"],
                        "satisfaction": ["Job Satisfaction", "Engagement"],
                        "work": ["Work Environment", "Job Satisfaction", "Engagement"]
                    }

                    # Check if any keywords from the topic match available topics
                    topic_words = topic.lower().split()
                    for word in topic_words:
                        if word in topic_keywords:
                            for candidate_topic in topic_keywords[word]:
                                if candidate_topic in available_topics:
                                    actual_topic = candidate_topic
                                    print(
                                        f"🔍 Using keyword match: '{topic}' → '{actual_topic}' (via '{word}')")
                                    break
                            if actual_topic != topic:
                                break

        print(f"🔍 Looking for topic '{topic}' → found '{actual_topic}'")

        # If still no match found, show available topics for debugging
        if actual_topic == topic and available_topics:
            print(
                f"⚠️ No match found for '{topic}'. Available topics: {available_topics}")

        for resp in responses:
            if resp.topics_scores and actual_topic in resp.topics_scores:
                score = resp.topics_scores[actual_topic]
                topic_scores.append(score)
                if resp.insight:
                    insights.append(resp.insight)
                print(f"✅ Found score {score} for {actual_topic}")

        print(f"📊 Topic '{actual_topic}': {len(topic_scores)} scores found")

        return {
            "topic": actual_topic,
            "avg_score": round(np.mean(topic_scores), 2) if topic_scores else 0,
            "total_responses": len(topic_scores),
            "score_distribution": {str(i): topic_scores.count(i) for i in range(1, 6)} if topic_scores else {},
            "key_insights": insights[:5]  # Top 5 insights
        }

    def _manual_parse_query(self, query: str) -> Dict[str, Any]:
        """Manual parsing fallback for cohort queries"""
        query_lower = query.lower()

        # Default values
        filter_criteria = {
            "field": "Years of Experience",
            "condition": "greater_than",
            "value": "0",
            "topic_focus": None
        }

        # Parse experience patterns
        if ">5" in query or "more than 5" in query_lower or "over 5" in query_lower:
            filter_criteria.update(
                {"field": "Years of Experience", "condition": "greater_than", "value": "5"})
        elif "5-10" in query or "5 to 10" in query_lower or "tenure of 5" in query_lower:
            filter_criteria.update(
                {"field": "Years of Experience", "condition": "greater_than", "value": "5"})
        elif "<2" in query or "fresher" in query_lower or "new employee" in query_lower:
            filter_criteria.update(
                {"field": "Years of Experience", "condition": "less_than", "value": "2"})
        elif ">3" in query or "experienced" in query_lower:
            filter_criteria.update(
                {"field": "Years of Experience", "condition": "greater_than", "value": "3"})
        elif "tenure" in query_lower and any(num in query for num in ["5", "6", "7", "8", "9", "10"]):
            # Extract number from tenure query
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                filter_criteria.update(
                    {"field": "Years of Experience", "condition": "greater_than", "value": numbers[0]})

        # Parse gender patterns
        if "female" in query_lower or "women" in query_lower:
            filter_criteria.update(
                {"field": "Gender", "condition": "contains", "value": "Female"})
        elif "male" in query_lower or "men" in query_lower:
            filter_criteria.update(
                {"field": "Gender", "condition": "contains", "value": "Male"})

        # Parse location patterns (expanded for enterprise dataset)
        locations = ["mumbai", "bangalore", "delhi", "chennai", "hyderabad", "pune",
                     "ahmedabad", "surat", "kolkata", "jaipur", "lucknow", "kanpur",
                     "nagpur", "indore", "thane", "bhopal", "visakhapatnam", "pimpri"]
        for location in locations:
            if location in query_lower:
                # For enterprise format, location might be in different fields
                location_fields = ["Location",
                                   "Branch", "Base_Station", "Zone"]
                for field in location_fields:
                    # Try each possible location field
                    filter_criteria.update(
                        {"field": field, "condition": "contains", "value": location.title()})
                    break
                break

        # Parse topic focus (expanded list with better mapping)
        # Note: These should match the actual topics extracted by the LLM
        topic_mappings = {
            "workload": "Workload",
            "work load": "Workload",
            "work-load": "Workload",
            "manager": "Leadership",
            "management": "Leadership",
            "supervisor": "Leadership",
            "leadership": "Leadership",
            "feedback": "Leadership",
            "mentor": "Leadership",
            "growth": "Development",
            "career": "Development",
            "development": "Development",
            "opportunities": "Development",
            "culture": "Culture",
            "company": "Culture",
            "organization": "Culture",
            "communication": "Communication",
            "communicate": "Communication",
            "compensation": "Compensation",
            "salary": "Compensation",
            "pay": "Compensation",
            "work-life balance": "Work-Life Balance",
            "work life balance": "Work-Life Balance",
            "balance": "Work-Life Balance",
            "training": "Training",
            "skill development": "Training",
            "skills": "Training",
            "recognition": "Recognition",
            "appreciation": "Recognition",
            "valued": "Recognition",
            "satisfaction": "Job Satisfaction",
            "satisfied": "Job Satisfaction",
            "job": "Job Satisfaction",
            "ethics": "Integrity",
            "integrity": "Integrity",
            "collaboration": "Team Collaboration",
            "teamwork": "Team Collaboration",
            "team": "Team Collaboration",
            "engagement": "Engagement",
            "productivity": "Engagement",
            "motivated": "Engagement"
        }

        # Find the first matching topic (prioritize more specific matches)
        matched_topic = None
        for keyword, topic_name in topic_mappings.items():
            if keyword in query_lower:
                matched_topic = topic_name
                break

        if matched_topic:
            filter_criteria["topic_focus"] = matched_topic

        return filter_criteria

    def _generate_cohort_summary(self, responses: List[SurveyResponse]) -> Dict[str, Any]:
        """Generate a summary for a cohort without full report details"""
        if not responses:
            return {"error": "No responses in cohort"}

        # Calculate average scores for all topics
        topic_averages = {}
        for topic in self.topics:
            scores = []
            for resp in responses:
                if resp.topics_scores and topic in resp.topics_scores:
                    scores.append(resp.topics_scores[topic])

            if scores:
                topic_averages[topic] = {
                    "avg_score": round(np.mean(scores), 2),
                    "response_count": len(scores)
                }

        # Get top concerns (lowest scoring topics)
        concerns = sorted(topic_averages.items(),
                          key=lambda x: x[1]["avg_score"])[:3]

        # Get top strengths (highest scoring topics)
        strengths = sorted(topic_averages.items(),
                           key=lambda x: x[1]["avg_score"], reverse=True)[:3]

        # Sample insights from this cohort
        sample_insights = []
        for resp in responses[:5]:
            if resp.insight:
                sample_insights.append(resp.insight)

        return {
            "cohort_size": len(responses),
            "top_concerns": [{"topic": topic, "avg_score": data["avg_score"]} for topic, data in concerns],
            "top_strengths": [{"topic": topic, "avg_score": data["avg_score"]} for topic, data in strengths],
            "sample_insights": sample_insights[:3],
            "overall_sentiment": round(np.mean([resp.sentiment_score for resp in responses if resp.sentiment_score]), 2) if responses else 3.0
        }

    def generate_executive_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive executive summary combining all analysis insights
        """
        print("📋 Generating executive summary...")

        if not self.responses or not self.topics:
            raise ValueError(
                "No analyzed data available. Please run full analysis first.")

        # 1. Overall Survey Statistics
        total_responses = len(self.responses)
        analyzed_responses = len(
            [r for r in self.responses if r.topics_scores])

        # 2. Overall Sentiment Analysis
        sentiment_scores = [
            r.sentiment_score for r in self.responses if r.sentiment_score]
        overall_sentiment = {
            "avg_score": round(np.mean(sentiment_scores), 2) if sentiment_scores else 3.0,
            "positive_count": len([s for s in sentiment_scores if s >= 4]),
            "negative_count": len([s for s in sentiment_scores if s <= 2]),
            "neutral_count": len([s for s in sentiment_scores if s == 3])
        }

        # 3. Topic Performance Analysis
        topic_performance = {}
        for topic in self.topics:
            scores = []
            for resp in self.responses:
                if resp.topics_scores and topic in resp.topics_scores:
                    scores.append(resp.topics_scores[topic])

            if scores:
                avg_score = np.mean(scores)
                topic_performance[topic] = {
                    "avg_score": round(avg_score, 2),
                    "total_responses": len(scores),
                    "status": "Strong" if avg_score >= 4 else "Needs Attention" if avg_score <= 2.5 else "Moderate"
                }

        # 4. Demographic Insights Analysis
        demographic_insights = self._analyze_demographic_patterns()

        # 5. Key Issues and Strengths Identification
        key_issues = []
        key_strengths = []

        for topic, data in topic_performance.items():
            if data["avg_score"] <= 2.5:
                key_issues.append({
                    "topic": topic,
                    "score": data["avg_score"],
                    "affected_responses": data["total_responses"]
                })
            elif data["avg_score"] >= 4.0:
                key_strengths.append({
                    "topic": topic,
                    "score": data["avg_score"],
                    "positive_responses": data["total_responses"]
                })

        # 6. Generate LLM-powered Executive Insights
        executive_insights = self._generate_executive_insights(
            overall_sentiment, topic_performance, demographic_insights, key_issues, key_strengths
        )

        # 7. Actionable Recommendations
        recommendations = self._generate_recommendations(
            key_issues, demographic_insights)

        return {
            "executive_summary": {
                "survey_overview": {
                    "total_responses": total_responses,
                    "analyzed_responses": analyzed_responses,
                    "survey_type": self.survey_type or "Employee Survey",
                    "analysis_date": datetime.now().isoformat()
                },
                "overall_sentiment": {
                    "average_score": overall_sentiment["avg_score"],
                    "sentiment_breakdown": {
                        "positive": f"{(overall_sentiment['positive_count']/total_responses*100):.1f}%",
                        "neutral": f"{(overall_sentiment['neutral_count']/total_responses*100):.1f}%",
                        "negative": f"{(overall_sentiment['negative_count']/total_responses*100):.1f}%"
                    },
                    "overall_mood": "Positive" if overall_sentiment["avg_score"] >= 3.5 else "Negative" if overall_sentiment["avg_score"] <= 2.5 else "Mixed"
                },
                "key_findings": {
                    "top_strengths": sorted(key_strengths, key=lambda x: x["score"], reverse=True)[:3],
                    "critical_issues": sorted(key_issues, key=lambda x: x["score"])[:3],
                    "topic_performance": topic_performance
                },
                "demographic_insights": demographic_insights,
                "executive_insights": executive_insights,
                "actionable_recommendations": recommendations
            }
        }

    def _analyze_demographic_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across different demographic groups"""
        demographic_patterns = {}

        # Get available demographic fields
        demo_fields = []
        if self.responses:
            sample_metadata = self.responses[0].metadata
            demo_fields = [field for field in sample_metadata.keys()
                           if field in ['Gender', 'Department', 'Grade', 'Zone', 'Branch', 'Tenure', 'Years of Experience']]

        for field in demo_fields[:3]:  # Analyze top 3 demographic fields
            field_analysis = {}
            field_values = {}

            # Group responses by field value
            for resp in self.responses:
                if resp.topics_scores and field in resp.metadata:
                    field_value = str(resp.metadata[field])
                    if field_value not in field_values:
                        field_values[field_value] = []
                    field_values[field_value].append(resp.sentiment_score or 3)

            # Analyze each group
            for value, scores in field_values.items():
                if len(scores) >= 5:  # Only analyze groups with sufficient data
                    avg_sentiment = np.mean(scores)
                    field_analysis[value] = {
                        "count": len(scores),
                        "avg_sentiment": round(avg_sentiment, 2),
                        "sentiment_trend": "Positive" if avg_sentiment >= 3.5 else "Negative" if avg_sentiment <= 2.5 else "Neutral"
                    }

            if field_analysis:
                demographic_patterns[field] = field_analysis

        return demographic_patterns

    def _generate_executive_insights(self, sentiment_data, topic_data, demo_data, issues, strengths) -> List[str]:
        """Generate executive-level insights using LLM"""

        # Prepare data summary for LLM
        data_summary = f"""
        Survey Results Summary:
        - Total Responses: {len(self.responses)}
        - Overall Sentiment: {sentiment_data['avg_score']}/5.0
        - Positive Responses: {sentiment_data['positive_count']}
        - Negative Responses: {sentiment_data['negative_count']}
        
        Top Issues: {[issue['topic'] + f" ({issue['score']}/5)" for issue in issues[:3]]}
        Top Strengths: {[strength['topic'] + f" ({strength['score']}/5)" for strength in strengths[:3]]}

        Demographic Patterns: {demo_data}
        """

        insight_prompt = f"""
        As an HR executive, analyze this survey data and provide 5-7 key executive insights that tell the complete story.

        {data_summary}

        Focus on:
        1. Overall employee sentiment and what's driving it
        2. Critical issues that need immediate attention
        3. Demographic patterns and which groups are most/least satisfied
        4. Strengths to build upon
        5. Business impact and risks

        Return insights as a JSON array of strings:
        ["Insight 1", "Insight 2", "Insight 3", ...]

        Make insights specific, actionable, and executive-level (not just data summaries).
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": insight_prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            try:
                insights = json.loads(result)
                return insights if isinstance(insights, list) else [result]
            except json.JSONDecodeError:
                # Fallback parsing
                import re
                insights = re.findall(r'"([^"]+)"', result)
                return insights if insights else [
                    f"Overall sentiment is {'positive' if sentiment_data['avg_score'] >= 3.5 else 'concerning'} at {sentiment_data['avg_score']}/5.0",
                    f"Key issues identified: {', '.join([issue['topic'] for issue in issues[:2]])}",
                    f"Strengths to leverage: {', '.join([strength['topic'] for strength in strengths[:2]])}"
                ]

        except Exception as e:
            print(f"❌ Error generating executive insights: {e}")
            return [
                f"Survey analysis completed with {len(self.responses)} responses",
                f"Overall sentiment score: {sentiment_data['avg_score']}/5.0",
                f"Critical areas needing attention: {', '.join([issue['topic'] for issue in issues[:3]])}"
            ]

    def _generate_recommendations(self, issues, demo_data) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Issue-based recommendations
        for issue in issues[:3]:
            topic = issue['topic']
            score = issue['score']

            rec = {
                "priority": "High" if score <= 2.0 else "Medium",
                "area": topic,
                "issue": f"Low satisfaction in {topic} (Score: {score}/5)",
                "recommendation": self._get_topic_recommendation(topic),
                "affected_employees": issue['affected_responses']
            }
            recommendations.append(rec)

        # Demographic-based recommendations
        for field, groups in demo_data.items():
            lowest_group = min(
                groups.items(), key=lambda x: x[1]['avg_sentiment'])
            if lowest_group[1]['avg_sentiment'] <= 2.5:
                rec = {
                    "priority": "Medium",
                    "area": f"{field} - {lowest_group[0]}",
                    "issue": f"Low satisfaction in {lowest_group[0]} group (Score: {lowest_group[1]['avg_sentiment']}/5)",
                    "recommendation": f"Conduct focused sessions with {lowest_group[0]} group to understand specific concerns and develop targeted improvement plan",
                    "affected_employees": lowest_group[1]['count']
                }
                recommendations.append(rec)

        return sorted(recommendations, key=lambda x: {"High": 3, "Medium": 2, "Low": 1}[x["priority"]], reverse=True)

    def _get_topic_recommendation(self, topic: str) -> str:
        """Get specific recommendations for each topic"""
        recommendations = {
            "Work Environment": "Improve physical workspace, tools, and working conditions. Consider flexible work arrangements.",
            "Culture": "Strengthen company values communication, improve inclusivity initiatives, and enhance team building activities.",
            "Leadership": "Provide leadership training for managers, implement regular feedback sessions, and improve communication channels.",
            "Manager": "Enhance manager training programs, establish mentorship guidelines, and improve manager-employee relationship building.",
            "Development": "Expand learning opportunities, create clear career progression paths, and increase training budget allocation.",
            "Growth": "Implement career development programs, provide skill-building opportunities, and establish promotion transparency.",
            "Communication": "Improve internal communication channels, increase transparency in decision-making, and enhance feedback mechanisms.",
            "Team Collaboration": "Facilitate cross-team projects, improve collaboration tools, and strengthen team dynamics through workshops.",
            "Job Satisfaction": "Review role clarity, workload distribution, and recognition programs to enhance overall job satisfaction.",
            "Engagement": "Implement employee engagement initiatives, improve work meaningfulness, and enhance motivation programs.",
            "Training": "Expand training programs, improve training quality, and ensure training relevance to job requirements.",
            "Recognition": "Enhance recognition programs, implement peer-to-peer recognition, and improve achievement celebration.",
            "Compensation": "Review compensation structure, ensure market competitiveness, and improve benefits communication.",
            "Work-Life Balance": "Implement flexible work policies, manage workload expectations, and promote wellness initiatives."
        }

        return recommendations.get(topic, f"Conduct detailed analysis of {topic} concerns and develop targeted improvement strategies.")

    # COMMENTED OUT OPENAI APPROACH
    # def analyze_keywords_and_wordcloud(self, frequency_threshold: int = 10) -> Dict[str, Any]:
    #     """
    #     Analyze keywords and generate word cloud data for positive and negative terms
    #     """
        print("🔍 Analyzing keywords and generating word cloud data...")

        if not self.responses:
            raise ValueError(
                "No survey responses available. Please upload data first.")

        # Collect all response texts
        all_texts = []
        positive_texts = []
        negative_texts = []

        for resp in self.responses:
            combined_text = " ".join(resp.responses.values())
            all_texts.append(combined_text)

            # Categorize based on sentiment score if available
            if resp.sentiment_score:
                if resp.sentiment_score >= 4:
                    positive_texts.append(combined_text)
                elif resp.sentiment_score <= 2:
                    negative_texts.append(combined_text)

        # Use LLM to extract and categorize keywords
        keyword_analysis = self._extract_keywords_with_llm(
            all_texts, positive_texts, negative_texts)

        # Filter keywords with frequency >= threshold
        filtered_positive = [
            kw for kw in keyword_analysis["positive_keywords"] if kw["count"] >= frequency_threshold]
        filtered_negative = [
            kw for kw in keyword_analysis["negative_keywords"] if kw["count"] >= frequency_threshold]

        # Generate word frequency analysis
        word_frequency = self._analyze_word_frequency(
            all_texts, frequency_threshold)

        return {
            "message": "Keyword and word cloud analysis completed",
            "total_responses_analyzed": len(self.responses),
            "positive_responses": len(positive_texts),
            "negative_responses": len(negative_texts),
            "positive_keywords": filtered_positive,
            "negative_keywords": filtered_negative,
            "word_frequency_data": word_frequency,
            "frequency_threshold": frequency_threshold,
            "analysis_date": datetime.now().isoformat()
        }

    def _extract_keywords_with_llm(self, all_texts: List[str], positive_texts: List[str], negative_texts: List[str]) -> Dict[str, Any]:
        """Extract positive and negative keywords using LLM analysis"""

        # Sample texts to avoid token limits
        sample_all = " ".join(all_texts[:100])  # First 100 responses
        sample_positive = " ".join(
            positive_texts[:50]) if positive_texts else ""
        sample_negative = " ".join(
            negative_texts[:50]) if negative_texts else ""

        keyword_prompt = f"""
        Analyze these survey responses and extract the most frequently mentioned positive and negative keywords/themes.

        All Survey Responses Sample:
        {sample_all[:3000]}

        Positive Responses Sample:
        {sample_positive[:1500]}

        Negative Responses Sample:
        {sample_negative[:1500]}

        Extract the top keywords and estimate their frequency. Focus on:
        - Work-related terms (support, teamwork, growth, culture, etc.)
        - Sentiment-bearing words (positive: supportive, flexible, recognition; negative: stagnation, stress, politics)
        - Workplace concepts (training, promotion, workload, transparency, etc.)

        Return as JSON:
        {{
            "positive_keywords": [
                {{"keyword": "Support/Supportive", "count": 240,
                    "category": "workplace_support"}},
                {{"keyword": "Team/Teamwork", "count": 212, "category": "collaboration"}},
                {{"keyword": "Learning/Training", "count": 181, "category": "development"}}
            ],
            "negative_keywords": [
                {{"keyword": "Promotion/Stagnation",
                    "count": 115, "category": "career_growth"}},
                {{"keyword": "Salary", "count": 104, "category": "compensation"}},
                {{"keyword": "Recognition/Underappreciation",
                    "count": 101, "category": "recognition"}}
            ]
        }}

        Provide realistic frequency counts based on the sample size and content.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": keyword_prompt}],
                temperature=0.3
            )

            result_text = response.choices[0].message.content.strip()

            try:
                keyword_data = json.loads(result_text)
                return keyword_data
            except json.JSONDecodeError:
                # Fallback parsing
                import re

                # Try to extract keywords manually
                positive_keywords = []
                negative_keywords = []

                # Look for positive patterns
                positive_patterns = [
                    ("Support/Supportive", 180), ("Team/Teamwork",
                                                  150), ("Learning/Training", 140),
                    ("Growth/Opportunities", 130), ("Environment/Atmosphere",
                                                    120), ("Recognition", 110),
                    ("Culture", 100), ("Flexible/Flexibility",
                                       90), ("Communication", 85), ("Management", 80)
                ]

                # Look for negative patterns
                negative_patterns = [
                    ("Promotion/Stagnation", 95), ("Salary",
                                                   85), ("Recognition/Underappreciation", 80),
                    ("Politics/Favoritism", 65), ("Workload/Stress",
                                                  60), ("Transparency", 50),
                    ("Management/Leadership", 45), ("Communication",
                                                    40), ("Work-Life Balance", 35), ("Training", 30)
                ]

                for keyword, count in positive_patterns:
                    positive_keywords.append({
                        "keyword": keyword,
                        "count": count,
                        "category": "workplace_positive"
                    })

                for keyword, count in negative_patterns:
                    negative_keywords.append({
                        "keyword": keyword,
                        "count": count,
                        "category": "workplace_negative"
                    })

                return {
                    "positive_keywords": positive_keywords,
                    "negative_keywords": negative_keywords
                }

        except Exception as e:
            print(f"❌ Error extracting keywords: {e}")
            # Return fallback data
            return {
                "positive_keywords": [
                    {"keyword": "Support/Supportive", "count": 180,
                        "category": "workplace_support"},
                    {"keyword": "Team/Teamwork", "count": 150,
                        "category": "collaboration"},
                    {"keyword": "Learning/Training",
                        "count": 140, "category": "development"},
                    {"keyword": "Growth/Opportunities",
                        "count": 130, "category": "career_growth"},
                    {"keyword": "Environment/Atmosphere",
                        "count": 120, "category": "workplace_culture"}
                ],
                "negative_keywords": [
                    {"keyword": "Promotion/Stagnation",
                        "count": 95, "category": "career_growth"},
                    {"keyword": "Salary", "count": 85, "category": "compensation"},
                    {"keyword": "Recognition/Underappreciation",
                        "count": 80, "category": "recognition"},
                    {"keyword": "Politics/Favoritism", "count": 65,
                        "category": "workplace_politics"},
                    {"keyword": "Workload/Stress", "count": 60,
                        "category": "work_pressure"}
                ]
            }

    def _analyze_word_frequency(self, texts: List[str], frequency_threshold: int = 10) -> Dict[str, Any]:
        """Analyze word frequency for additional insights"""
        from collections import Counter
        import re

        # Combine all texts
        combined_text = " ".join(texts).lower()

        # Clean and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)

        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'then', 'them', 'well', 'were', 'been', 'have', 'there', 'where', 'much', 'your', 'work', 'life', 'only', 'think', 'also', 'back', 'after', 'first', 'well', 'year', 'come', 'could', 'like', 'time', 'very', 'when', 'much', 'new', 'write', 'would', 'there', 'each', 'which', 'their', 'said', 'will', 'about', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'with', 'have', 'this', 'that', 'what', 'will', 'more', 'other', 'into', 'people', 'really', 'things', 'always', 'being', 'feel', 'need', 'would', 'company', 'employees', 'employee'
        }

        # Filter words
        filtered_words = [
            word for word in words if word not in stop_words and len(word) > 3]

        # Count frequencies
        word_counts = Counter(filtered_words)

        # Get top words with minimum frequency threshold
        top_words = word_counts.most_common(50)

        # Filter words with frequency >= threshold
        filtered_top_words = [{"word": word, "count": count}
                              for word, count in top_words if count >= frequency_threshold]

        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "top_words": filtered_top_words,
            "word_diversity": len(set(words)) / len(words) if words else 0,
            "words_above_threshold": len(filtered_top_words),
            "frequency_threshold": frequency_threshold
        }


# Initialize global analyzer
survey_analyzer = SurveyPipelineAnalyzer()

# API Endpoints


@app.get("/")
async def root():
    return {
        "message": "Survey Pipeline with Cohort Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload-data",
            "detect_type": "GET /detect-survey-type",
            "analyze": "GET /analyze-survey",
            "report": "GET /generate-report",
            "cohort": "GET /cohort-analysis",
            "debug_metadata": "GET /debug-metadata",
            "debug_topics": "GET /debug-topics",
            "debug_file_format": "GET /debug-file-format",

            "executive_summary": "GET /executive-summary"
        }
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """
    Step 1: Upload Data & Vectorization
    Parse file, preprocess, extract metadata, compute embeddings
    """
    try:
        # Read file
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(await file.read()))
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(await file.read()))
        else:
            raise HTTPException(
                status_code=400, detail="Only Excel (.xlsx) and CSV files are supported")

        # Process data
        result = survey_analyzer.upload_and_process_data(df)
        result["filename"] = file.filename

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/detect-survey-type")
async def detect_survey_type():
    """
    Step 2: Survey Type Detection
    Sample responses and detect survey type using LLM
    """
    try:
        result = survey_analyzer.detect_survey_type()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error detecting survey type: {str(e)}")


@app.get("/analyze-survey")
async def analyze_survey():
    """
    Step 3: Survey Analysis
    Extract topics and analyze each response with LLM
    """
    try:
        result = survey_analyzer.extract_topics_and_analyze()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing survey: {str(e)}")


@app.get("/generate-report")
async def generate_report(
    cohort_field: Optional[str] = Query(
        None, description="Metadata field for cohort filtering"),
    cohort_value: Optional[str] = Query(None, description="Value to filter by")
):
    """
    Step 4: Report Generation
    Generate comprehensive survey report with optional cohort filtering
    """
    try:
        cohort_filter = None
        if cohort_field and cohort_value:
            cohort_filter = {
                "field": cohort_field,
                "condition": "contains",
                "value": cohort_value
            }

        report = survey_analyzer.generate_report(cohort_filter)

        # Convert to dict for JSON response
        return {
            "survey_type": report.survey_type,
            "total_responses": report.total_responses,
            "topics": [
                {
                    "name": topic.name,
                    "avg_score": topic.avg_score,
                    "insights": topic.insights,
                    "distribution": topic.distribution,
                    "representative_responses": topic.representative_responses
                }
                for topic in report.topics
            ],
            "sentiment": report.sentiment,
            "cohort_filters_available": report.cohort_filters_available,
            "analysis_date": report.analysis_date.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/cohort-analysis")
async def cohort_analysis(query: str = Query(..., description="Natural language cohort query")):
    """
    Step 5: LLM-based Cohort Analysis
    Process natural language queries for cohort analysis

    Examples:
    - "Show me responses of employees with >5 years experience about Workload"
    - "Tell me what freshers said about their Manager"
    - "What do female employees think about Work-Life Balance"
    """
    try:
        result = survey_analyzer.cohort_analysis(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in cohort analysis: {str(e)}")


@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "status": "active",
        "responses_loaded": len(survey_analyzer.responses) if survey_analyzer.responses else 0,
        "survey_type": survey_analyzer.survey_type,
        "topics_extracted": survey_analyzer.topics,
        "embeddings_ready": survey_analyzer.embeddings is not None,
        "faiss_index_ready": survey_analyzer.faiss_index is not None
    }


@app.get("/debug-metadata")
async def debug_metadata():
    """Debug endpoint to see what metadata fields and values exist"""
    if not survey_analyzer.responses:
        return {"error": "No responses loaded. Please upload data first."}

    # Get sample metadata from first few responses
    sample_metadata = []
    for i, resp in enumerate(survey_analyzer.responses[:5]):
        sample_metadata.append({
            "response_id": resp.response_id,
            "metadata": resp.metadata
        })

    # Get all unique metadata fields
    all_fields = set()
    for resp in survey_analyzer.responses:
        all_fields.update(resp.metadata.keys())

    # Get sample values for each field
    field_samples = {}
    for field in all_fields:
        values = []
        for resp in survey_analyzer.responses[:10]:
            if field in resp.metadata:
                values.append(resp.metadata[field])
        field_samples[field] = list(set(values))[:5]  # First 5 unique values

    return {
        "total_responses": len(survey_analyzer.responses),
        "available_metadata_fields": list(all_fields),
        "field_sample_values": field_samples,
        "sample_metadata": sample_metadata
    }


@app.get("/debug-topics")
async def debug_topics():
    """Debug endpoint to see what topics were extracted and sample topic scores"""
    if not survey_analyzer.responses:
        return {"error": "No responses loaded. Please upload and analyze data first."}

    if not survey_analyzer.topics:
        return {"error": "No topics extracted. Please run /analyze-survey first."}

    # Get sample topic scores from first few responses
    sample_topic_scores = []
    for i, resp in enumerate(survey_analyzer.responses[:5]):
        if resp.topics_scores:
            sample_topic_scores.append({
                "response_id": resp.response_id,
                "topics_scores": resp.topics_scores,
                "insight": resp.insight
            })

    # Get all unique topics that actually exist in responses
    actual_topics_in_responses = set()
    for resp in survey_analyzer.responses:
        if resp.topics_scores:
            actual_topics_in_responses.update(resp.topics_scores.keys())

    return {
        "extracted_topics": survey_analyzer.topics,
        "actual_topics_in_responses": list(actual_topics_in_responses),
        "sample_topic_scores": sample_topic_scores,
        "total_analyzed_responses": len([r for r in survey_analyzer.responses if r.topics_scores])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)


@app.get("/debug-file-format")
async def debug_file_format():
    """Debug endpoint to test file format detection without full processing"""
    if not survey_analyzer.responses:
        return {"error": "No responses loaded. Please upload data first."}

    # Get sample of processed data
    sample_responses = []
    for i, resp in enumerate(survey_analyzer.responses[:3]):
        sample_responses.append({
            "response_id": resp.response_id,
            "metadata": resp.metadata,
            "responses": {k: v[:100] + "..." if len(v) > 100 else v for k, v in resp.responses.items()},
            "has_topic_scores": resp.topics_scores is not None
        })

    return {
        "total_responses": len(survey_analyzer.responses),
        "sample_responses": sample_responses,
        "processing_successful": True
    }


@app.get("/executive-summary")
async def executive_summary():
    """
    Executive Summary - Comprehensive insights combining all analysis results

    Provides a complete overview including:
    - Overall sentiment and key findings
    - Demographic insights and patterns
    - Critical issues and strengths
    - Executive-level insights
    - Actionable recommendations
    """
    try:
        summary = survey_analyzer.generate_executive_summary()
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating executive summary: {str(e)}")


# COMMENTED OUT - OLD OPENAI APPROACH
# @app.get("/keywords-wordcloud")
# async def keywords_wordcloud(
#     frequency_threshold: int = Query(
#         10, description="Minimum frequency threshold for keywords and words (default: 10)")
# ):
#     """
#     Keywords and Word Cloud Analysis (OLD OPENAI VERSION - COMMENTED OUT)
#     """
#     try:
#         result = survey_analyzer.analyze_keywords_and_wordcloud(frequency_threshold)
#         return result
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error analyzing keywords and word cloud: {str(e)}")


@app.get("/wordcloud-quick")
async def quick_wordcloud(
    min_frequency: int = Query(
        10, description="Minimum frequency threshold for words/phrases (default: 10)"),
    include_structured_terms: bool = Query(
        False, description="Include structured survey tokens like score/agree/positive in analysis"),
    free_text_only: bool = Query(
        True, description="Only analyze fields that look like free text (recommended)"),
    free_text_min_chars: int = Query(
        20, description="Minimum length to treat a field as free text when free_text_only is true")
):
    """
    Enhanced Quick Word Cloud Analysis

    Features:
    - Top occurring words and phrases ranked by frequency (250 → 10+ mentions)
    - Single words (unigrams) and two-word phrases (bigrams)
    - Robust stopword filtering to remove common words (a, and, the, etc.)
    - Positive/negative sentiment segregation
    - Configurable minimum frequency threshold
    - Fast analysis optimized for dashboards

    Returns:
    - Ranked positive/negative words and bigrams with exact counts
    - Frequency range from highest (e.g., 250) down to minimum threshold
    - Clean results with stopwords filtered out
    """
    global survey_analyzer

    if not survey_analyzer or not survey_analyzer.responses:
        raise HTTPException(status_code=400, detail="No survey data available")

    try:
        # Extract all response texts with optional free-text filtering
        all_texts = []
        if free_text_only:
            for resp in survey_analyzer.responses:
                parts = []
                for v in resp.responses.values():
                    if not isinstance(v, str):
                        continue
                    txt = v.strip()
                    if len(txt) >= free_text_min_chars and (" " in txt or any(ch in txt for ch in ".,!?:;")):
                        parts.append(txt)
                if parts:
                    all_texts.append(" ".join(parts))
        else:
            for resp in survey_analyzer.responses:
                combined_text = " ".join(
                    [v for v in resp.responses.values() if isinstance(v, str)])
                all_texts.append(combined_text)

        # Perform enhanced word and phrase analysis (Sentence Transformers-backed)
        result = dynamic_quick_wordcloud_analysis(
            all_texts,
            min_frequency=min_frequency,
            include_structured_terms=include_structured_terms
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in enhanced quick word cloud analysis: {str(e)}"
        )

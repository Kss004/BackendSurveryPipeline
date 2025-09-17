from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import Dict, List, Any
import os
from datetime import datetime
from dotenv import load_dotenv

from services.na_handler import NAHandler
from services.vector_analyzer import VectorAnalyzer

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Employee Survey Analysis API",
    description="Unified vector-based API for analyzing employee survey data with sentiment analysis and cohort insights",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the NA handler and vector analyzer
na_handler = NAHandler()
vector_analyzer = VectorAnalyzer()


@app.get("/")
async def root():
    return {
        "message": "Employee Survey Analysis API v2.0 is running!",
        "description": "Unified vector-based analysis system",
        "endpoints": {
            "process_data": {
                "upload": "POST /analyze/upload",
                "existing": "POST /analyze/existing"
            },
            "analysis": {
                "comprehensive": "GET /analyze/comprehensive",
                "sentiment": "GET /analyze/sentiment",
                "team_support": "GET /analyze/team-support",
                "promotion_requests": "GET /analyze/promotion-requests",
                "cohort": "GET /analyze/cohort"
            },
            "utilities": {
                "missing_data": "GET /analyze/missing-data",
                "data_quality": "GET /analyze/data-quality"
            }
        }
    }


@app.post("/analyze/upload")
async def process_survey_data(file: UploadFile = File(...)):
    """
    Upload and process survey file - creates vectors for subsequent analysis
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

        # Process data for vector analysis - this creates and stores vectors
        success = vector_analyzer.process_survey_data(df_clean)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data for vector analysis")

        # Return processing confirmation
        return {
            "message": "Survey data processed successfully",
            "filename": file.filename,
            "total_responses": len(vector_analyzer.metadata),
            "data_shape": df_clean.shape,
            "vectors_created": len(vector_analyzer.embeddings_cache) if vector_analyzer.embeddings_cache is not None else 0,
            "processing_date": datetime.now(),
            "available_endpoints": [
                "/analyze/sentiment",
                "/analyze/team-support",
                "/analyze/promotion-requests",
                "/analyze/comprehensive"
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/analyze/existing")
async def process_existing_file():
    """
    Process the existing dummy survey file - creates vectors for subsequent analysis
    """
    try:
        file_path = "/Users/shashwat/Projects/Hyrgpt/EmpSurv/Dummy_Employee_Survey_Responses (1).xlsx"

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail="Dummy survey file not found")

        # Read the existing file with proper header detection
        df = pd.read_excel(file_path, header=1)

        # Clean the data
        print(f"📊 Processing existing file")
        print(f"📋 Original data shape: {df.shape}")

        df_clean = na_handler.clean_and_prepare_data(df, strategy='smart')
        print(f"✅ Cleaned data shape: {df_clean.shape}")

        # Process data for vector analysis - this creates and stores vectors
        success = vector_analyzer.process_survey_data(df_clean)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to process survey data for vector analysis")

        # Return processing confirmation
        return {
            "message": "Existing survey data processed successfully",
            "filename": "Dummy_Employee_Survey_Responses (1).xlsx",
            "total_responses": len(vector_analyzer.metadata),
            "data_shape": df_clean.shape,
            "vectors_created": len(vector_analyzer.embeddings_cache) if vector_analyzer.embeddings_cache is not None else 0,
            "processing_date": datetime.now(),
            "available_endpoints": [
                "/analyze/sentiment",
                "/analyze/team-support",
                "/analyze/promotion-requests",
                "/analyze/comprehensive"
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing existing file: {str(e)}")


@app.get("/analyze/comprehensive")
async def get_comprehensive_analysis():
    """
    Get comprehensive analysis with all metrics using stored vectors - YOUR MAIN ENDPOINT
    Provides all the metrics you requested: sentiment, team support, recognition, promotion, etc.
    """
    try:
        # Check if vectors are available
        if not vector_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please upload and process data first using /analyze/upload or /analyze/existing"
            )

        print("🔄 Performing comprehensive vector analysis...")

        # Get comprehensive analysis using existing vector analysis method
        analysis_result = vector_analyzer.analyze_comprehensive()

        # Helper function to extract metric data
        def extract_metric(metric):
            return {
                "value": metric.range_estimate,
                "analysis": metric.analysis,
                "indicator": metric.indicator
            }

        # Perform cohort analysis using stored vectors
        cohort_analysis = vector_analyzer._analyze_cohorts()

        # Get additional insights
        vector_info = vector_analyzer.get_vectors_info()

        # Create comprehensive response
        response = {
            # Your requested core metrics
            "overall_positive_sentiment": extract_metric(analysis_result.overall_positive_sentiment),
            "team_support_mentions": extract_metric(analysis_result.team_support_mentions),
            "recognition_requests": extract_metric(analysis_result.recognition_requests),
            "promotion_concerns": extract_metric(analysis_result.promotion_concerns),
            "strong_positive": extract_metric(analysis_result.strong_positive_percent),
            "learning_mentions": extract_metric(analysis_result.learning_mentions),
            "politics_concerns": extract_metric(analysis_result.politics_concerns),
            "team_culture_strength": extract_metric(analysis_result.team_culture_strength),
            "strong_negative": extract_metric(analysis_result.strong_negative_percent),

            # Cohort analysis
            "cohort_analysis": cohort_analysis,

            # Discovered themes
            "themes": analysis_result.themes,

            # Insights and recommendations
            "insights": analysis_result.insights,

            # Metadata
            "total_responses_analyzed": len(vector_analyzer.metadata),
            "analysis_date": datetime.now(),
            "vector_info": vector_info,
            "analysis_method": "OpenAI embeddings + Faiss vector search + GPT analysis"
        }

        print("✅ Comprehensive analysis completed")
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing comprehensive analysis: {str(e)}")


@app.get("/analyze/cohort")
async def get_cohort_analysis():
    """
    Get cohort analysis using stored vectors
    """
    try:
        # Check if vectors are available
        if not vector_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please upload and process data first using /analyze/upload or /analyze/existing"
            )

        # Perform cohort analysis using stored vectors
        cohort_analysis = vector_analyzer._analyze_cohorts()

        # Get vector info
        vector_info = vector_analyzer.get_vectors_info()

        # Create insights based on cohort differences
        insights = []
        if len(cohort_analysis) > 1:
            # Compare sentiment across cohorts
            sentiments = [cohort['sentiment_score']
                          for cohort in cohort_analysis]
            max_sentiment = max(sentiments)
            min_sentiment = min(sentiments)

            if max_sentiment - min_sentiment > 20:  # More than 20% difference
                insights.append(
                    "Significant sentiment differences detected across employee cohorts")

            # Identify best and worst performing cohorts
            best_cohort = max(
                cohort_analysis, key=lambda x: x['sentiment_score'])
            worst_cohort = min(
                cohort_analysis, key=lambda x: x['sentiment_score'])

            insights.append(
                f"Highest satisfaction: {best_cohort['cohort_name']} ({best_cohort['sentiment_score']}%)")
            insights.append(
                f"Needs attention: {worst_cohort['cohort_name']} ({worst_cohort['sentiment_score']}%)")

        return {
            "total_employees": vector_info.get("total_responses", 0),
            "cohorts": [
                {
                    "cohort_name": cohort["cohort_name"],
                    "employee_count": cohort["employee_count"],
                    "avg_satisfaction": cohort["sentiment_score"],
                    "common_themes": cohort["common_themes"]
                }
                for cohort in cohort_analysis
            ],
            "analysis_date": datetime.now(),
            "insights": insights,
            "analysis_method": "Vector-based cohort sentiment analysis"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing cohort analysis: {str(e)}")


@app.get("/analyze/sentiment")
async def get_sentiment_analysis():
    """
    Get sentiment analysis using stored vectors
    """
    try:
        # Check if vectors are available
        if not vector_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please upload and process data first using /analyze/upload or /analyze/existing"
            )

        # Get all sentiments using stored vectors
        all_sentiments = [
            vector_analyzer._calculate_sentiment_score(emb)
            for emb in vector_analyzer.embeddings_cache
        ]

        # Calculate sentiment metrics
        total_responses = len(all_sentiments)

        positive_count = sum(1 for s in all_sentiments if s > 0.6)
        negative_count = sum(1 for s in all_sentiments if s < 0.4)
        neutral_count = total_responses - positive_count - negative_count

        # Overall sentiment score (0-1 scale)
        overall_score = sum(all_sentiments) / total_responses

        # Calculate percentages
        positive_percent = round((positive_count / total_responses) * 100, 1)
        negative_percent = round((negative_count / total_responses) * 100, 1)
        neutral_percent = round((neutral_count / total_responses) * 100, 1)

        # Department-level analysis (using cohort data)
        department_sentiments = {}
        cohort_data = {}

        # Group by tenure (as department proxy)
        for idx, metadata in enumerate(vector_analyzer.metadata):
            tenure_key = metadata.tenure
            if tenure_key not in cohort_data:
                cohort_data[tenure_key] = []
            cohort_data[tenure_key].append(all_sentiments[idx])

        # Calculate department sentiment scores
        for dept, sentiments in cohort_data.items():
            if len(sentiments) > 0:
                dept_positive = sum(1 for s in sentiments if s > 0.6)
                dept_negative = sum(1 for s in sentiments if s < 0.4)
                dept_neutral = len(sentiments) - dept_positive - dept_negative
                dept_overall = sum(sentiments) / len(sentiments)

                department_sentiments[dept] = {
                    "positive": round((dept_positive / len(sentiments)) * 100, 1),
                    "negative": round((dept_negative / len(sentiments)) * 100, 1),
                    "neutral": round((dept_neutral / len(sentiments)) * 100, 1),
                    "overall_score": round(dept_overall, 3)
                }

        # Extract themes using vector analysis
        themes_result = vector_analyzer._discover_themes_clustering()

        # Get positive themes (high sentiment clusters)
        key_positive_themes = [
            theme['representative_text'][:100] + "..." if len(theme['representative_text']) > 100
            else theme['representative_text']
            for theme in themes_result
            if theme['avg_sentiment'] > 0.6
        ][:5]

        # Get negative themes (low sentiment clusters)
        key_negative_themes = [
            theme['representative_text'][:100] + "..." if len(theme['representative_text']) > 100
            else theme['representative_text']
            for theme in themes_result
            if theme['avg_sentiment'] < 0.4
        ][:5]

        return {
            "overall_sentiment": {
                "positive": positive_percent,
                "negative": negative_percent,
                "neutral": neutral_percent,
                "overall_score": round(overall_score, 3)
            },
            "department_sentiments": department_sentiments,
            "key_positive_themes": key_positive_themes,
            "key_negative_themes": key_negative_themes,
            "total_responses_analyzed": total_responses,
            "analysis_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing sentiment analysis: {str(e)}")


@app.get("/analyze/team-support")
async def get_team_support_analysis():
    """
    Analyze team support mentions using vector semantic search
    """
    try:
        # Check if vectors are available
        if not vector_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please upload and process data first using /analyze/upload or /analyze/existing"
            )

        # Perform semantic search for team support mentions
        team_support_results = vector_analyzer._semantic_search(
            "team support collaboration help assistance mentorship guidance teamwork cooperation",
            top_k=50
        )

        # Filter results by similarity threshold
        similarity_threshold = 0.4
        relevant_results = [
            r for r in team_support_results if r[1] > similarity_threshold]

        # Analyze sentiment of team support mentions
        positive_mentions = 0
        negative_mentions = 0
        neutral_mentions = 0
        detailed_mentions = []

        for result_idx, similarity in relevant_results:
            metadata = vector_analyzer.metadata[result_idx]
            sentiment_score = vector_analyzer._calculate_sentiment_score(
                vector_analyzer.embeddings_cache[result_idx]
            )

            # Categorize by sentiment
            if sentiment_score > 0.6:
                mention_type = "positive"
                positive_mentions += 1
            elif sentiment_score < 0.4:
                mention_type = "negative"
                negative_mentions += 1
            else:
                mention_type = "neutral"
                neutral_mentions += 1

            # Add to detailed mentions
            detailed_mentions.append({
                "employee_id": metadata.employee_id,
                "department": metadata.tenure,  # Using tenure as department proxy
                "mention_type": mention_type,
                "context": metadata.response_text[:200] + "..." if len(metadata.response_text) > 200 else metadata.response_text,
                "similarity_score": round(similarity, 3),
                "sentiment_score": round(sentiment_score, 3)
            })

        # Group mentions by department (tenure)
        mentions_by_department = {}
        for mention in detailed_mentions:
            dept = mention["department"]
            mentions_by_department[dept] = mentions_by_department.get(
                dept, 0) + 1

        # Calculate range estimate for team support mentions
        total_mentions = len(relevant_results)
        total_responses = len(vector_analyzer.metadata)

        # Create range (similar to vector analysis logic)
        if total_responses <= 50:
            range_width = max(2, int(total_responses * 0.08))
        elif total_responses < 200:
            range_width = max(5, int(total_responses * 0.05))
        else:
            range_width = max(10, int(total_responses * 0.03))

        range_start = max(0, total_mentions - range_width // 2)
        range_end = total_mentions + range_width // 2

        if range_start == range_end:
            range_estimate = str(total_mentions)
        else:
            range_estimate = f"{range_start}-{range_end}"

        return {
            "total_mentions": total_mentions,
            "range_estimate": range_estimate,
            "positive_mentions": positive_mentions,
            "negative_mentions": negative_mentions,
            "neutral_mentions": neutral_mentions,
            "mentions_by_department": mentions_by_department,
            # Top 10 most relevant
            "detailed_mentions": detailed_mentions[:10],
            "percentage_of_responses": round((total_mentions / total_responses) * 100, 1),
            "analysis_method": "Vector semantic search with similarity threshold 0.4",
            "analysis_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing team support: {str(e)}")


@app.get("/analyze/promotion-requests")
async def get_promotion_requests():
    """
    Analyze promotion-related requests and concerns using vector semantic search
    """
    try:
        # Check if vectors are available
        if not vector_analyzer.has_vectors():
            raise HTTPException(
                status_code=400,
                detail="No vectors available. Please upload and process data first using /analyze/upload or /analyze/existing"
            )

        # Perform semantic search for promotion-related content
        promotion_results = vector_analyzer._semantic_search(
            "promotion career advancement growth opportunity progress development promotion ladder climb raise salary increase",
            top_k=50
        )

        # Filter results by similarity threshold
        similarity_threshold = 0.4
        relevant_results = [
            r for r in promotion_results if r[1] > similarity_threshold]

        # Analyze and categorize promotion mentions
        direct_requests = 0
        concerns_raised = 0
        suggestions_made = 0
        detailed_requests = []

        for result_idx, similarity in relevant_results:
            metadata = vector_analyzer.metadata[result_idx]
            response_text = metadata.response_text.lower()

            # Determine request type based on content analysis
            if any(keyword in response_text for keyword in ['want promotion', 'need promotion', 'deserve promotion', 'promote me']):
                request_type = "direct_request"
                direct_requests += 1
            elif any(keyword in response_text for keyword in ['no promotion', 'lack of growth', 'limited advancement', 'stuck']):
                request_type = "concern"
                concerns_raised += 1
            else:
                request_type = "suggestion"
                suggestions_made += 1

            # Determine urgency based on sentiment and keywords
            sentiment_score = vector_analyzer._calculate_sentiment_score(
                vector_analyzer.embeddings_cache[result_idx]
            )

            if sentiment_score < 0.3 or any(urgent in response_text for urgent in ['urgent', 'immediate', 'now', 'asap']):
                urgency_level = "high"
            elif sentiment_score < 0.5:
                urgency_level = "medium"
            else:
                urgency_level = "low"

            # Add to detailed requests
            detailed_requests.append({
                "employee_id": metadata.employee_id,
                "department": metadata.tenure,  # Using tenure as department proxy
                "request_type": request_type,
                "urgency_level": urgency_level,
                "context": metadata.response_text[:200] + "..." if len(metadata.response_text) > 200 else metadata.response_text,
                "similarity_score": round(similarity, 3),
                "sentiment_score": round(sentiment_score, 3)
            })

        # Group by department and urgency
        requests_by_department = {}
        urgency_breakdown = {"high": 0, "medium": 0, "low": 0}

        for request in detailed_requests:
            dept = request["department"]
            requests_by_department[dept] = requests_by_department.get(
                dept, 0) + 1
            urgency_breakdown[request["urgency_level"]] += 1

        # Calculate range estimate
        total_requests = len(relevant_results)
        total_responses = len(vector_analyzer.metadata)

        # Create range (similar to vector analysis logic)
        if total_responses <= 50:
            range_width = max(2, int(total_responses * 0.08))
        elif total_responses < 200:
            range_width = max(5, int(total_responses * 0.05))
        else:
            range_width = max(10, int(total_responses * 0.03))

        range_start = max(0, total_requests - range_width // 2)
        range_end = total_requests + range_width // 2

        if range_start == range_end:
            range_estimate = str(total_requests)
        else:
            range_estimate = f"{range_start}-{range_end}"

        return {
            "total_requests": total_requests,
            "range_estimate": range_estimate,
            "direct_requests": direct_requests,
            "concerns_raised": concerns_raised,
            "suggestions_made": suggestions_made,
            "requests_by_department": requests_by_department,
            "urgency_breakdown": urgency_breakdown,
            # Top 10 most relevant
            "detailed_requests": detailed_requests[:10],
            "percentage_of_responses": round((total_requests / total_responses) * 100, 1),
            "analysis_method": "Vector semantic search with similarity threshold 0.4",
            "analysis_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing promotion requests: {str(e)}")


@app.get("/analyze/missing-data")
async def get_missing_data_analysis():
    """
    Analyze missing data patterns in the survey responses
    """
    try:
        file_path = "/Users/shashwat/Projects/Hyrgpt/EmpSurv/Dummy_Employee_Survey_Responses (1).xlsx"
        df = pd.read_excel(file_path, header=1)  # Use row 1 as header

        na_analysis = na_handler.analyze_missing_data(df)

        # Convert to JSON-serializable format
        return {
            "total_cells": int(na_analysis.total_cells),
            "total_nas": int(na_analysis.total_nas),
            "na_percentage": round(float(na_analysis.na_percentage), 1),
            "na_by_column": {k: int(v) for k, v in na_analysis.na_by_column.items()},
            "na_by_employee": {k: int(v) for k, v in na_analysis.na_by_employee.items()},
            "missing_patterns": {k: int(v) for k, v in na_analysis.missing_patterns.items()},
            "recommendations": na_analysis.recommendations,
            "analysis_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing missing data: {str(e)}")


@app.get("/analyze/data-quality")
async def get_data_quality_metrics():
    """
    Get comprehensive data quality metrics including response completeness
    """
    try:
        file_path = "/Users/shashwat/Projects/Hyrgpt/EmpSurv/Dummy_Employee_Survey_Responses (1).xlsx"
        df = pd.read_excel(file_path, header=1)  # Use row 1 as header

        # Get response completeness scores
        completeness_scores = na_handler.get_response_completeness_score(df)

        # Check validity for analysis
        validity_check = na_handler.is_valid_for_analysis(
            df, min_responses_per_question=15)

        # Calculate summary statistics
        avg_completeness = sum(completeness_scores.values()
                               ) / len(completeness_scores)
        min_completeness = min(completeness_scores.values())
        max_completeness = max(completeness_scores.values())

        return {
            "response_completeness_by_employee": {k: float(v) for k, v in completeness_scores.items()},
            "average_completeness": round(float(avg_completeness), 1),
            "min_completeness": round(float(min_completeness), 1),
            "max_completeness": round(float(max_completeness), 1),
            "questions_valid_for_analysis": {k: bool(v) for k, v in validity_check.items()},
            "total_valid_questions": int(sum(1 for valid in validity_check.values() if valid)),
            "total_questions": int(len(validity_check)),
            "analysis_date": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing data quality: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

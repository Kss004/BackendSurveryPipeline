"""
NA (Missing Data) Handler for Employee Survey Analysis

This module provides comprehensive handling of missing data in survey responses,
including detection, categorization, and intelligent treatment strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class NAAnalysis:
    """Container for NA analysis results"""
    total_cells: int
    total_nas: int
    na_percentage: float
    na_by_column: Dict[str, int]
    na_by_employee: Dict[str, int]
    missing_patterns: Dict[str, int]
    recommendations: List[str]


class NAHandler:
    """
    Comprehensive NA handling for survey data
    """

    def __init__(self):
        # Define what constitutes "missing" data
        self.missing_indicators = [
            'na', 'n/a', 'nan', 'null', 'none', '',
            'not applicable', 'skip', 'no response',
            'prefer not to answer', 'no answer'
        ]

    def analyze_missing_data(self, df: pd.DataFrame) -> NAAnalysis:
        """
        Comprehensive analysis of missing data patterns
        """
        # Basic statistics
        total_cells = df.shape[0] * df.shape[1]
        total_nas = df.isna().sum().sum()
        na_percentage = (total_nas / total_cells) * 100

        # NA by column
        na_by_column = df.isna().sum().to_dict()

        # NA by employee (using Participant or index)
        na_by_employee = {}
        participant_col = 'Participant' if 'Participant' in df.columns else None

        for idx, row in df.iterrows():
            employee_id = row[participant_col] if participant_col else f"Employee_{idx}"
            na_count = row.isna().sum()
            na_by_employee[str(employee_id)] = na_count

        # Missing patterns (which combinations of questions are missing)
        missing_patterns = self._identify_missing_patterns(df)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            df, na_percentage, na_by_column)

        return NAAnalysis(
            total_cells=total_cells,
            total_nas=total_nas,
            na_percentage=na_percentage,
            na_by_column=na_by_column,
            na_by_employee=na_by_employee,
            missing_patterns=missing_patterns,
            recommendations=recommendations
        )

    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Identify common patterns of missing data"""
        patterns = {}

        for idx, row in df.iterrows():
            # Get columns that are missing for this row
            missing_cols = [col for col in df.columns if pd.isna(row[col])]

            if missing_cols:
                # Create a pattern signature
                pattern = tuple(sorted(missing_cols))
                pattern_name = f"Missing: {', '.join(missing_cols[:2])}{'...' if len(missing_cols) > 2 else ''}"

                if pattern_name not in patterns:
                    patterns[pattern_name] = 0
                patterns[pattern_name] += 1

        return patterns

    def _generate_recommendations(self, df: pd.DataFrame, na_percentage: float, na_by_column: Dict[str, int]) -> List[str]:
        """Generate recommendations for handling missing data"""
        recommendations = []

        if na_percentage < 5:
            recommendations.append(
                "Low missing data rate (<5%) - safe to use listwise deletion")
        elif na_percentage < 15:
            recommendations.append(
                "Moderate missing data rate - consider mean/mode imputation for numeric analysis")
        elif na_percentage < 30:
            recommendations.append(
                "High missing data rate - use advanced imputation or pattern analysis")
        else:
            recommendations.append(
                "Very high missing data rate - investigate data collection issues")

        # Column-specific recommendations
        high_missing_cols = [
            col for col, count in na_by_column.items() if count > len(df) * 0.3]
        if high_missing_cols:
            recommendations.append(
                f"Columns with >30% missing: {', '.join(high_missing_cols)} - consider excluding from analysis")

        # Check for systematic patterns
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            recommendations.append(
                "For text analysis: treat NA as 'no response' and include in sentiment analysis")

        return recommendations

    def clean_and_prepare_data(self, df: pd.DataFrame, strategy: str = "smart") -> pd.DataFrame:
        """
        Clean and prepare data based on the chosen strategy

        Strategies:
        - 'smart': Intelligent handling based on data type and context
        - 'exclude_na': Remove rows/columns with too many NAs
        - 'include_na': Keep NAs but handle them explicitly
        - 'impute': Fill missing values with reasonable defaults
        """
        df_clean = df.copy()

        if strategy == "smart":
            df_clean = self._smart_na_handling(df_clean)
        elif strategy == "exclude_na":
            df_clean = self._exclude_na_strategy(df_clean)
        elif strategy == "include_na":
            df_clean = self._include_na_strategy(df_clean)
        elif strategy == "impute":
            df_clean = self._impute_na_strategy(df_clean)

        return df_clean

    def _smart_na_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart NA handling strategy - different approaches for different types of analysis
        """
        df_clean = df.copy()

        # For demographic columns - keep as is (should have no NAs anyway)
        demographic_cols = ['Participant',
                            'GENDER', 'TENURE IN JAQ', 'Department']

        # For text response columns - replace NA with explicit "No response" for better analysis
        text_response_cols = [
            col for col in df.columns if col not in demographic_cols]

        for col in text_response_cols:
            if col in df_clean.columns:
                # Replace NAs with "No response provided" for text analysis
                df_clean[col] = df_clean[col].fillna("No response provided")

        return df_clean

    def _exclude_na_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude rows/columns with excessive NAs"""
        df_clean = df.copy()

        # Remove columns with more than 50% NAs
        threshold = len(df_clean) * 0.5
        df_clean = df_clean.dropna(axis=1, thresh=threshold)

        # Remove rows with more than 50% NAs
        threshold = len(df_clean.columns) * 0.5
        df_clean = df_clean.dropna(axis=0, thresh=threshold)

        return df_clean

    def _include_na_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep NAs but mark them explicitly"""
        df_clean = df.copy()

        # Add binary columns indicating missingness
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['Participant', 'GENDER', 'TENURE IN JAQ']:
                df_clean[f"{col}_is_missing"] = df_clean[col].isna()

        # Fill text NAs with explicit marker
        text_cols = df_clean.select_dtypes(include=['object']).columns
        for col in text_cols:
            if col not in ['Participant', 'GENDER', 'TENURE IN JAQ']:
                df_clean[col] = df_clean[col].fillna("[MISSING_RESPONSE]")

        return df_clean

    def _impute_na_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with reasonable defaults"""
        df_clean = df.copy()

        # For text responses, use mode (most common response) or neutral response
        text_cols = df_clean.select_dtypes(include=['object']).columns
        demographic_cols = ['Participant', 'GENDER', 'TENURE IN JAQ']

        for col in text_cols:
            if col not in demographic_cols:
                # Try to use mode, fallback to neutral response
                try:
                    mode_value = df_clean[col].mode(
                    ).iloc[0] if not df_clean[col].mode().empty else "Neutral response"
                    df_clean[col] = df_clean[col].fillna(mode_value)
                except:
                    df_clean[col] = df_clean[col].fillna("Neutral response")

        return df_clean

    def get_response_completeness_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate response completeness score for each employee
        """
        scores = {}
        participant_col = 'Participant' if 'Participant' in df.columns else None

        # Count non-demographic columns
        total_questions = len([col for col in df.columns if col not in [
                              'Participant', 'GENDER', 'TENURE IN JAQ']])

        for idx, row in df.iterrows():
            employee_id = row[participant_col] if participant_col else f"Employee_{idx}"

            # Count answered questions (non-NA)
            answered = sum(1 for col in df.columns
                           if col not in ['Participant', 'GENDER', 'TENURE IN JAQ'] and pd.notna(row[col]))

            completeness = (answered / total_questions) * \
                100 if total_questions > 0 else 100
            scores[str(employee_id)] = completeness

        return scores

    def is_valid_for_analysis(self, df: pd.DataFrame, min_responses_per_question: int = 15) -> Dict[str, bool]:
        """
        Check if each column has enough responses for meaningful analysis
        """
        validity = {}

        for col in df.columns:
            if col not in ['Participant', 'GENDER', 'TENURE IN JAQ']:
                response_count = df[col].notna().sum()
                validity[col] = response_count >= min_responses_per_question

        return validity

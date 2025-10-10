# 🔍 Survey Analysis API

**Enterprise-scale survey analysis system with LLM-powered insights, cohort analysis, and executive reporting.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [API Workflow](#api-workflow)
- [Setup Instructions](#setup-instructions)


## 🎯 Overview

This API transforms raw survey data into actionable business insights using advanced LLM technology. It processes large-scale employee surveys (1000+ responses) and provides comprehensive analysis including sentiment analysis, topic extraction, demographic insights, and executive-level recommendations.

### 🏢 Built For
- **HR Teams** - Employee engagement analysis
- **Management** - Strategic decision making
- **Researchers** - Survey data analysis
- **Consultants** - Client reporting

## ✨ Key Features

### 🤖 **LLM-Powered Analysis**
- **Topic Extraction**: Automatically identifies key themes from survey responses
- **Sentiment Analysis**: Comprehensive positive/negative sentiment scoring
- **Natural Language Cohort Queries**: Ask questions like "What do female employees with >5 years experience think about leadership?"

### 📊 **Advanced Analytics**
- **Batch Processing**: Handles 1000+ responses efficiently (5-10 minutes vs 45+ minutes)
- **Demographic Insights**: Analyzes patterns across departments, tenure, gender, etc.
- **Statistical Significance**: Identifies meaningful differences between groups

### 📈 **Executive Reporting**
- **Executive Summary**: Comprehensive insights combining all analysis
- **Actionable Recommendations**: Prioritized improvement strategies
- **Business Impact Analysis**: Risk identification and opportunity mapping

### 🔧 **Enterprise Ready**
- **Multiple File Formats**: Excel (.xlsx), CSV support
- **Large Dataset Support**: Optimized for 1000+ responses
- **Flexible Data Structure**: Handles various survey formats automatically

## 🏗️ Architecture

```mermaid
graph TB
    A[Survey Data Upload] --> B[File Format Detection]
    B --> C[Data Preprocessing]
    C --> D[Survey Type Detection]
    D --> E[Topic Extraction via LLM]
    E --> F[Batch Response Analysis]
    F --> G[Report Generation]
    G --> H[Executive Summary]
    
    I[Cohort Analysis] --> J[Natural Language Query Parsing]
    J --> K[Demographic Filtering]
    K --> L[Topic-Specific Analysis]
    
    M[Sentiment Analysis] --> N[Overall Sentiment Scoring]
    
    subgraph "LLM Integration"
        E
        F
        J
        N
    end
    
    subgraph "Data Processing"
        B
        C
        K
    end
    
    subgraph "Analytics Engine"
        G
        H
        L
    end
```

## 🔄 API Workflow

### **Phase 1: Data Ingestion & Processing**
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant LLM
    participant Database
    
    Client->>API: POST /upload-data (Excel/CSV)
    API->>API: Detect file format
    API->>API: Clean & preprocess data
    API->>API: Extract metadata & responses
    API->>Database: Store processed data
    API->>Client: Upload confirmation
```

### **Phase 2: Analysis Pipeline**
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant LLM
    
    Client->>API: GET /analyze-survey
    API->>LLM: Sample 30 responses for topic extraction
    LLM->>API: Return 5-6 key topics
    API->>LLM: Batch analyze responses (10 per batch)
    LLM->>API: Return topic scores & insights
    API->>API: Aggregate results
    API->>Client: Analysis complete
```

### **Phase 3: Insights & Reporting**
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant LLM
    
    Client->>API: GET /executive-summary
    API->>API: Calculate demographic patterns
    API->>API: Identify key issues & strengths
    API->>LLM: Generate executive insights
    LLM->>API: Return strategic recommendations
    API->>Client: Comprehensive executive report
```

## 🚀 Setup Instructions

### **Prerequisites**
- Python 3.8+
- OpenAI API key
- 4GB+ RAM (for large datasets)

### **1. Clone Repository**
```bash
git clone https://github.com/Kss004/BackendSurveryPipeline
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Environment Configuration**
Create `.env` file:
```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
EMBEDDING_MODEL= EMBEDDING_MODEL (Optional)
LLM_MODEL=gpt-4o-mini (Optional)
```

### **4. Run Application**
```bash
uvicorn main_4_cohort:app      
```

### **5. Access API Documentation**
Visit: `http://localhost:8000/docs`

## 🔧 Data Processing Pipeline

### **File Format Detection**
```mermaid
flowchart TD
    A[Upload File] --> B{File Type?}
    B -->|Excel| C[Read .xlsx]
    B -->|CSV| D[Read .csv]
    C --> E{Header Row Detection}
    D --> E
    E -->|Questions in Row 1| F[Use Row 1 as Headers]
    E -->|Meaningful Columns| G[Use Existing Headers]
    F --> H[Column Classification]
    G --> H
    H --> I[Metadata vs Response Columns]
    I --> J[Data Cleaning & Preprocessing]
```

### **Batch Processing Strategy**
```mermaid
flowchart LR
    A[1784 Responses] --> B[Split into Batches of 10]
    B --> C[Batch 1: Responses 1-10]
    B --> D[Batch 2: Responses 11-20]
    B --> E[Batch N: Responses 1781-1784]
    
    C --> F[LLM Analysis]
    D --> F
    E --> F
    
    F --> G[Parse Results]
    G --> H[Aggregate Scores]
    H --> I[Generate Insights]
```

### **Topic Extraction Process**
```mermaid
flowchart TD
    A[Sample 30 Random Responses] --> B[Send to GPT-4]
    B --> C[Extract 5-6 Key Topics]
    C --> D{Topics Valid?}
    D -->|Yes| E[Use Extracted Topics]
    D -->|No| F[Use Fallback Topics]
    E --> G[Analyze All Responses]
    F --> G
    G --> H[Score Each Response 1-5]
    H --> I[Generate Topic Insights]
```

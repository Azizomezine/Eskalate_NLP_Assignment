# NLP Assignment: Intelligent Text Analysis System

## Overview

This project implements an intelligent system capable of extracting entities from unstructured text, generating document summaries, and designing an AI agent for higher-level text analysis tasks. The system is built as part of an NLP assessment focusing on information extraction, summarization, and agentic system design.

## Project Structure

```
eskalte/
├── Eskalate_NLP_Assignment.ipynb    # Main notebook with implementation
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
└── Eskalate ML_DS Interview Questions - August 10, 2025.pdf
```

## Dataset

**Amazon Polarity Dataset** (from Hugging Face datasets)
- **Source**: `fancyzhx/amazon_polarity`
- **Content**: Customer product reviews with sentiment labels
- **Sample Size**: 1,000 reviews (reduced for faster execution)
- **Labels**: Binary sentiment (0=Negative, 1=Positive)

## Features Implemented

### Part 1: Data Preparation & Exploration

#### Text Preprocessing
- **Enhanced TextPreprocessor Class** with:
  - Contraction expansion (e.g., "can't" → "cannot")
  - URL and email removal
  - Optional number removal
  - Negation handling (e.g., "not good" → "not_good")
  - Tokenization with NLTK
  - Stop word removal
  - Lemmatization

#### Exploratory Data Analysis
- Document length distribution
- Token count analysis
- Sentiment distribution visualization
- Word frequency analysis
- Word clouds for positive/negative sentiment
- Box plots for review length by sentiment

### Part 2: Information Extraction & Summarization

#### A. Entity & Information Extraction

**1. Rule-Based Extraction**
- Price extraction (multiple formats: $X.XX, X dollars, etc.)
- Rating extraction (X/5, X stars, etc.)
- Product feature detection (quality, price, shipping, performance, design)

**2. Named Entity Recognition (NER)**
- SpaCy-based entity extraction
- Entities: PERSON, ORG, PRODUCT, MONEY, DATE, GPE
- Processed entities from review content

#### B. Summarization

**TF-IDF Extractive Summarization**
- Sentence-level importance scoring
- Top sentence extraction
- Maintains original sentence order
- Configurable summary length

### Part 3: Agentic System Design

#### ProductReviewAgent Architecture

**Agent Goal**: Analyze customer reviews to extract actionable business insights

**Agent Tools**:
- Rule-based information extractor
- SpaCy NER processor
- TF-IDF summarizer
- Text preprocessor

**Reasoning/Planning Strategy**:
1. **Query Processing**: Parse user questions to determine intent
2. **Document Search**: Find relevant reviews based on query tokens
3. **Information Extraction**: Apply appropriate extraction tools
4. **Insight Synthesis**: Aggregate findings and generate responses
5. **Response Generation**: Provide actionable business insights

**Key Capabilities**:
- Sentiment analysis and distribution
- Feature-based feedback analysis
- Customer concern identification
- Entity tracking and frequency analysis
- Question answering about review content

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd c:\Users\azizo\Desktop\eskalte
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SpaCy model** (if not auto-downloaded)
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Dependencies

The project requires the following packages (see `requirements.txt`):

```
datasets==4.0.0              # Hugging Face datasets
matplotlib==3.10.0           # Plotting and visualization
nltk==3.9.1                  # Natural language processing
numpy==2.0.2                 # Numerical computing
pandas==2.2.2                # Data manipulation
scikit-learn==1.6.1          # Machine learning utilities
seaborn==0.13.2              # Statistical visualization
spacy==3.8.7                 # Advanced NLP
wordcloud                    # Word cloud generation
en_core_web_sm               # SpaCy English model
```

## How to Run

### Option 1: Jupyter Notebook
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `Eskalate_NLP_Assignment.ipynb`
3. Run cells sequentially

### Option 2: Google Colab
1. Upload the notebook to Google Colab
2. Upload `requirements.txt` if needed
3. Run the setup cell first to install dependencies

## Module Usage

### 1. Text Preprocessing
```python
preprocessor = TextPreprocessor()

# Basic cleaning
clean_text = preprocessor.clean_text(text, remove_numbers=False)

# Full tokenization with negation handling
tokens = preprocessor.tokenize_and_clean(text, remove_numbers=False)
```

### 2. Information Extraction
```python
# Rule-based extraction
extractor = RuleBasedExtractor()
prices = extractor.extract_prices(text)
ratings = extractor.extract_ratings(text)
features = extractor.extract_product_features(text)

# SpaCy NER
spacy_extractor = SpacyExtractor(nlp)
entities = spacy_extractor.extract_entities(text)
```

### 3. Summarization
```python
summarizer = TFIDFSummarizer()
summary = summarizer.summarize(text, num_sentences=2)
```

### 4. AI Agent
```python
# Initialize agent
agent = ProductReviewAgent()
agent.load_reviews(dataframe)

# Get insights
insights = agent.get_insights_summary()

# Answer questions
response = agent.answer_question("What are the main complaints?")
```

## Key Results

### Preprocessing Insights
- Average document length: ~200 words
- Average tokens after preprocessing: ~50-100
- Successful contraction expansion and negation handling

### Extraction Performance
- Rule-based: High precision for structured patterns (prices, ratings)
- NER: Good entity recognition for organizations, people, locations
- Feature detection: Effective categorization of review aspects

### Agent Capabilities
- Sentiment distribution analysis
- Feature-based feedback categorization
- Customer concern identification
- Interactive question answering

## Technical Highlights

### Enhanced Preprocessing
- **Contraction Handling**: Comprehensive dictionary with 100+ contractions
- **Negation Preservation**: Maintains semantic meaning of negated phrases
- **Flexible Cleaning**: Optional number removal and configurable processing

### Robust Extraction
- **Multi-pattern Recognition**: Multiple regex patterns for flexible extraction
- **Entity Validation**: Length limits and relevance filtering
- **Error Handling**: Graceful degradation for malformed input

### Intelligent Agent Design
- **Query Understanding**: Intent detection and routing
- **Knowledge Base**: Structured storage of processed insights
- **Contextual Responses**: Tailored answers based on query type

## Future Enhancements

1. **Advanced Summarization**: Implement abstractive summarization with transformers
2. **ML-based Extraction**: Train custom NER models for domain-specific entities
3. **Agent Memory**: Implement conversation history and learning capabilities
4. **Real-time Processing**: Add streaming data processing capabilities
5. **Evaluation Metrics**: Implement ROUGE/BLEU scoring for summary quality

## Challenges and Solutions

### Challenge 1: Large Dataset Processing
- **Solution**: Sample-based processing (1,000 reviews) for demonstration
- **Production Approach**: Batch processing and streaming for scale

### Challenge 2: Negation Handling
- **Solution**: Custom token combination preserving semantic meaning
- **Impact**: Improved sentiment analysis accuracy

### Challenge 3: Entity Extraction Accuracy
- **Solution**: Combined rule-based and NER approaches
- **Result**: Higher precision through complementary methods

## License

This project is developed for educational purposes as part of an NLP assessment.

## Contact

For questions about this implementation, please refer to the notebook documentation or the assignment requirements.

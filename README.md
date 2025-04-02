# Batch Prediction POC

A proof of concept for batch processing video transcripts with context-aware question answering using Gemini AI. This project demonstrates efficient processing of video transcripts, intelligent chunking, semantic search, and caching mechanisms for improved performance.

## Features

- Video transcript processing and chunking with timestamps
- Semantic search using FAISS and Sentence Transformers
- Context-aware question answering using Gemini AI
- Thread-safe caching mechanism for similar queries
- Batch processing of queries
- Environment variable configuration for API keys

## Prerequisites

- Python 3.7+
- Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Batch_Prediction_POC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Prepare your transcript file (e.g., `sample_video.txt`) with the video content.

2. Run the script:
```bash
python batch_pred.py
```

## Example Output

Here's an example of how the system processes queries, demonstrating both API calls and cache hits:

```
# First Query (API Call)
Query: What is deep learning?
Response: Based on the provided context, deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). These networks learn hierarchical representations from data and have revolutionized fields like computer vision, natural language processing, and speech recognition.

Source: api
Relevant Chunks:
- [00:00] Deep learning is a subset of machine learning that uses neural networks with multiple layers. These layers are called deep neural networks. The first layer processes the input data, and each subsequent layer builds upon the previous layer's output. Activation functions are crucial components that introduce non-linearity into the network.
- [01:26] Regularization techniques like dropout and weight decay help prevent overfitting. Deep learning has revolutionized many fields, including computer vision, natural language processing, and speech recognition. The ability to learn hierarchical representations from data has made it possible to solve complex problems that were previously thought to be beyond the capabilities of machines.

# Exact Duplicate Query (Cache Hit)
Query: What is deep learning?
Response: Based on the provided context, deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). These networks learn hierarchical representations from data and have revolutionized fields like computer vision, natural language processing, and speech recognition.

Source: cache
Relevant Chunks:
- [00:00] Deep learning is a subset of machine learning that uses neural networks with multiple layers. These layers are called deep neural networks. The first layer processes the input data, and each subsequent layer builds upon the previous layer's output. Activation functions are crucial components that introduce non-linearity into the network.
- [01:26] Regularization techniques like dropout and weight decay help prevent overfitting. Deep learning has revolutionized many fields, including computer vision, natural language processing, and speech recognition. The ability to learn hierarchical representations from data has made it possible to solve complex problems that were previously thought to be beyond the capabilities of machines.

# Similar Query (Cache Hit due to Semantic Similarity)
Query: Can you explain deep learning?
Response: Based on the provided context, deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). These networks learn hierarchical representations from data and have revolutionized fields like computer vision, natural language processing, and speech recognition.

Source: cache
Relevant Chunks:
- [00:00] Deep learning is a subset of machine learning that uses neural networks with multiple layers. These layers are called deep neural networks. The first layer processes the input data, and each subsequent layer builds upon the previous layer's output. Activation functions are crucial components that introduce non-linearity into the network.
- [01:26] Regularization techniques like dropout and weight decay help prevent overfitting. Deep learning has revolutionized many fields, including computer vision, natural language processing, and speech recognition. The ability to learn hierarchical representations from data has made it possible to solve complex problems that were previously thought to be beyond the capabilities of machines.

# New Query (API Call)
Query: What are activation functions?
Response: According to the transcript, activation functions are crucial components that introduce non-linearity into neural networks.

Source: api
Relevant Chunks:
- [00:00] Deep learning is a subset of machine learning that uses neural networks with multiple layers. These layers are called deep neural networks. The first layer processes the input data, and each subsequent layer builds upon the previous layer's output. Activation functions are crucial components that introduce non-linearity into the network.
- [00:20] functions are crucial components that introduce non-linearity into the network. Common activation functions include ReLU, sigmoid, and tanh. The choice of activation function can significantly impact the network's performance and learning capabilities. Neural networks learn through a process called backpropagation. During training, the network makes predictions, calculates the error, and adjusts its weights to minimize this error.

# Similar Query (Cache Hit)
Query: Tell me about activation functions
Response: According to the transcript, activation functions are crucial components that introduce non-linearity into neural networks.

Source: cache
Relevant Chunks:
- [00:00] Deep learning is a subset of machine learning that uses neural networks with multiple layers. These layers are called deep neural networks. The first layer processes the input data, and each subsequent layer builds upon the previous layer's output. Activation functions are crucial components that introduce non-linearity into the network.
- [00:20] functions are crucial components that introduce non-linearity into the network. Common activation functions include ReLU, sigmoid, and tanh. The choice of activation function can significantly impact the network's performance and learning capabilities. Neural networks learn through a process called backpropagation. During training, the network makes predictions, calculates the error, and adjusts its weights to minimize this error.
```

The example above demonstrates three types of caching scenarios:
1. Exact duplicate queries (e.g., "What is deep learning?" → "What is deep learning?")
2. Semantically similar queries (e.g., "What is deep learning?" → "Can you explain deep learning?")
3. New queries that require API calls (e.g., "What are activation functions?")

The caching mechanism uses a similarity threshold (default: 0.8) to determine if a new query is similar enough to a cached query to reuse the response. This helps reduce API calls and improve response times for similar questions.

## Key Components

### Transcript Processing
- Splits transcripts into overlapping chunks with timestamps
- Maintains context between chunks
- Handles long sentences appropriately

### Semantic Search
- Uses FAISS for efficient similarity search
- Implements context expansion for better relevance
- Supports multiple video sources

### Caching Mechanism
- Thread-safe caching of similar queries
- Configurable similarity threshold
- Automatic cache management

### Batch Processing
- Parallel processing of multiple queries
- Efficient resource utilization
- Error handling and logging

## Configuration

Key parameters can be adjusted in the `VideoTranscriptProcessor` class:
- `chunk_size`: Number of words per chunk (default: 50)
- `overlap`: Number of overlapping words between chunks (default: 10)
- `similarity_threshold`: Threshold for query similarity (default: 0.8)
- `batch_size`: Number of queries to process in parallel (default: 5)

## Future Improvements

1. Integration with Whisper for video processing
2. Enhanced caching strategies
3. Improved chunking algorithms
4. Additional output formats
5. API endpoint for web integration

## License

[Your License Here] 
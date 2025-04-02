import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import deque
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    timestamp: str
    start_time: float
    end_time: float
    chunk_id: str
    video_id: str  # Added to support multiple videos

    def __hash__(self):
        return hash((self.text, self.timestamp, self.start_time, self.end_time, self.chunk_id, self.video_id))

    def __eq__(self, other):
        if not isinstance(other, TranscriptChunk):
            return NotImplemented
        return (self.text == other.text and 
                self.timestamp == other.timestamp and 
                self.start_time == other.start_time and 
                self.end_time == other.end_time and 
                self.chunk_id == other.chunk_id and
                self.video_id == other.video_id)

class VideoTranscriptProcessor:
    def __init__(self, api_key: str, chunk_size: int = 50, overlap: int = 10):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.transcript_chunks: List[TranscriptChunk] = []
        self.chunk_embeddings = None
        self.index = None
        self.dialogue_memory = deque(maxlen=10)
        self.batch_size = 5
        self.similarity_threshold = 0.8
        self.video_metadata = {}  # Store video metadata
        self.cache_lock = threading.Lock()  # Add lock for thread safety

    def process_video(self, video_path: str) -> str:
        """Process video and generate transcript using Whisper."""
        # TODO: Implement Whisper integration
        # For POC, we'll just read a transcript file
        transcript_path = Path(video_path).with_suffix('.txt')
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        return transcript_path.read_text()

    def chunk_transcript(self, transcript: str, video_id: str) -> List[TranscriptChunk]:
        """Split transcript into overlapping chunks with timestamps."""
        chunks = []
        # Clean and normalize the transcript
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if len(sentence_words) > self.chunk_size:
                # Split long sentence into smaller parts
                for i in range(0, len(sentence_words), self.chunk_size):
                    part = sentence_words[i:i + self.chunk_size]
                    chunk_text = " ".join(part)
                    start_time = word_count * 0.5
                    end_time = (word_count + len(part)) * 0.5
                    timestamp = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    
                    chunks.append(TranscriptChunk(
                        text=chunk_text,
                        timestamp=timestamp,
                        start_time=start_time,
                        end_time=end_time,
                        chunk_id=f"chunk_{chunk_id}",
                        video_id=video_id
                    ))
                    word_count += len(part)
                    chunk_id += 1
            else:
                current_chunk.extend(sentence_words)
                current_length += len(sentence_words)
                word_count += len(sentence_words)

                if current_length >= self.chunk_size:
                    chunk_text = " ".join(current_chunk)
                    start_time = (word_count - current_length) * 0.5
                    end_time = word_count * 0.5
                    timestamp = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    
                    chunks.append(TranscriptChunk(
                        text=chunk_text,
                        timestamp=timestamp,
                        start_time=start_time,
                        end_time=end_time,
                        chunk_id=f"chunk_{chunk_id}",
                        video_id=video_id
                    ))
                    
                    current_chunk = current_chunk[-self.overlap:]
                    current_length = len(current_chunk)
                    chunk_id += 1

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            start_time = (word_count - current_length) * 0.5
            end_time = word_count * 0.5
            timestamp = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            
            chunks.append(TranscriptChunk(
                text=chunk_text,
                timestamp=timestamp,
                start_time=start_time,
                end_time=end_time,
                chunk_id=f"chunk_{chunk_id}",
                video_id=video_id
            ))

        logger.info(f"Created {len(chunks)} chunks from transcript for video {video_id}")
        return chunks

    def build_index(self, chunks: List[TranscriptChunk]):
        """Build FAISS index for similarity search."""
        self.transcript_chunks = chunks
        chunk_texts = [chunk.text for chunk in chunks]
        self.chunk_embeddings = self.embed_model.encode(chunk_texts)
        
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.chunk_embeddings))

    def retrieve_relevant_chunks(self, query: str, k: int = 2) -> List[TranscriptChunk]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embed_model.encode([query])
        _, indices = self.index.search(query_embedding, k=k)
        return [self.transcript_chunks[i] for i in indices[0]]

    def expand_context(self, relevant_chunks: List[TranscriptChunk]) -> List[TranscriptChunk]:
        """Dynamically expand context by including adjacent chunks if relevant."""
        expanded_chunks = set(relevant_chunks)
        
        for chunk in relevant_chunks:
            chunk_idx = self.transcript_chunks.index(chunk)
            
            # Check previous chunk
            if chunk_idx > 0:
                prev_chunk = self.transcript_chunks[chunk_idx - 1]
                if self._is_chunk_relevant(prev_chunk, chunk):
                    expanded_chunks.add(prev_chunk)
            
            # Check next chunk
            if chunk_idx < len(self.transcript_chunks) - 1:
                next_chunk = self.transcript_chunks[chunk_idx + 1]
                if self._is_chunk_relevant(next_chunk, chunk):
                    expanded_chunks.add(next_chunk)
        
        return sorted(list(expanded_chunks), key=lambda x: x.start_time)

    def _is_chunk_relevant(self, chunk1: TranscriptChunk, chunk2: TranscriptChunk) -> bool:
        """Check if two chunks are semantically related."""
        similarity = np.dot(
            self.embed_model.encode([chunk1.text]),
            self.embed_model.encode([chunk2.text]).T
        )
        return similarity > self.similarity_threshold

    def process_batch_queries(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries in batch."""
        results = []
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.process_query, query) for query in queries]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    results.append({"error": str(e)})
        return results

    def is_query_similar(self, new_query: str) -> Optional[str]:
        """Check if query is similar to a past query and return cached response if found."""
        with self.cache_lock:  # Use lock when accessing cache
            for past_query, response, _ in self.dialogue_memory:
                similarity = np.dot(
                    self.embed_model.encode([new_query]),
                    self.embed_model.encode([past_query]).T
                )
                if similarity > self.similarity_threshold:
                    return response
        return None

    def add_to_cache(self, query: str, response: str, context: str):
        """Thread-safe method to add to cache."""
        with self.cache_lock:
            self.dialogue_memory.append((query, response, context))

    def process_query(self, query: str) -> Dict:
        """Process a single query and return structured response."""
        try:
            # Check cache first
            cached_response = self.is_query_similar(query)
            if cached_response:
                return {
                    "query": query,
                    "response": cached_response,
                    "source": "cache",
                    "timestamp": time.time(),
                    "status": "success"
                }

            # Retrieve and expand context
            relevant_chunks = self.retrieve_relevant_chunks(query)
            expanded_chunks = self.expand_context(relevant_chunks)
            
            # Prepare context for Gemini
            context = "\n".join([f"[{chunk.timestamp}] {chunk.text}" for chunk in expanded_chunks])
            
            # Call Gemini API
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(f"Context: {context}\nQuestion: {query}")
            generated_answer = response.text if hasattr(response, "text") else "Error generating response."
            
            # Store in cache using thread-safe method
            self.add_to_cache(query, generated_answer, context)
            
            # Prepare structured response
            return {
                "query": query,
                "response": generated_answer,
                "source": "api",
                "timestamp": time.time(),
                "status": "success",
                "relevant_chunks": [
                    {
                        "text": chunk.text,
                        "timestamp": chunk.timestamp,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "chunk_id": chunk.chunk_id,
                        "video_id": chunk.video_id
                    }
                    for chunk in expanded_chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": time.time(),
                "status": "error"
            }

def main():
    try:
        # Get API key from environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        # Example usage
        processor = VideoTranscriptProcessor(api_key=api_key)
        
        # Process video/transcript
        video_id = "sample_video_1"
        transcript = processor.process_video("sample_video.txt")
        chunks = processor.chunk_transcript(transcript, video_id)
        processor.build_index(chunks)
        
        # Example queries to demonstrate caching
        queries = [
            # First set of queries (will use API)
            "What is deep learning?",
            "What are activation functions?",
            "How do neural networks work?",
            
            # Duplicate queries (should use cache)
            "What is deep learning?",
            "What are activation functions?",
            "How do neural networks work?",
            
            # Similar queries (should use cache due to similarity threshold)
            "Can you explain deep learning?",
            "Tell me about activation functions",
            "How does a neural network function?"
        ]
        
        # Process queries in batch
        results = processor.process_batch_queries(queries)
        
        # Print results with cache information
        for result in results:
            print("\nQuery:", result["query"])
            if result.get("status") == "success":
                print("Response:", result["response"])
                print("Source:", result.get("source", "unknown"))
                if "relevant_chunks" in result:
                    print("Relevant Chunks:")
                    for chunk in result["relevant_chunks"]:
                        print(f"- [{chunk['timestamp']}] {chunk['text']}")
            else:
                print("Error:", result.get("error", "Unknown error occurred"))
                print("Status:", result.get("status", "unknown"))

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

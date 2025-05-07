# RAG-Based Support Ticket System

A semantic search and retrieval system for support tickets that uses Retrieval-Augmented Generation (RAG) to understand context beyond simple keywords.

## 🌟 Project Overview

Traditional support ticket systems rely heavily on keyword matching, which often leads to irrelevant results and frustration for both agents and users. This project implements a RAG-based approach that:

- Understands the semantic meaning behind support queries
- Retrieves contextually relevant historical tickets
- Generates helpful responses based on similar past issues
- Improves agent productivity and customer satisfaction

## ✨ Features

- **Semantic Understanding**: Uses neural embeddings to capture the meaning of queries
- **Vector Similarity Search**: Fast and accurate retrieval of relevant tickets
- **Contextual Response Generation**: Creates helpful suggestions based on similar past issues
- **Dual Interface**: Command-line and web-based UI options
- **Optional LLM Integration**: Enhanced responses through Gemini Flash (when API key is provided)

## 🛠️ Technical Stack

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Web Interface**: Streamlit
- **Optional LLM**: Google Generative AI (Gemini Flash 1.5)
- **Language**: Python 3.8+

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install sentence-transformers faiss-cpu streamlit google-generativeai
```

### Running the Application

#### Command Line Interface

```bash
python rag_support_system.py
```

#### Web Interface

```bash
streamlit run streamlit_app.py
```

To use the Gemini LLM integration, create a `.streamlit/secrets.toml` file with:

```toml
gemini_api_key = "YOUR_API_KEY"
```

## 📊 Sample Data

The system comes pre-loaded with sample support tickets including:

- Login issues on various browsers
- Authentication problems
- Password reset challenges
- Platform-specific errors

## 🧪 Example Queries

Try these sample queries to test the system:

- "Login error on Safari browser for enterprise users"
- "Password reset email not working"
- "Chrome extension conflicts with login"
- "Mobile app authentication problems"

## 🔍 How It Works

1. **Embedding Generation**: Converts text into dense vector representations
2. **Vector Storage**: Indexes ticket embeddings for efficient similarity search
3. **Query Processing**: Transforms user query into the same vector space
4. **Relevance Ranking**: Retrieves most similar tickets using cosine similarity
5. **Response Generation**: Creates helpful answer based on retrieved context

## 🧩 System Architecture

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  User Query   │────▶│   Embedding   │────▶│Vector Similarity│
└───────────────┘     │   Encoder     │     │     Search     │
                      └───────────────┘     └───────┬─────────┘
                                                   │
┌───────────────┐     ┌───────────────┐     ┌──────▼──────────┐
│   Response    │◀────│ Optional LLM  │◀────│ Retrieved Ticket │
│  Generation   │     │  Enhancement  │     │     Context     │
└───────────────┘     └───────────────┘     └─────────────────┘
```

## 🔮 Future Enhancements

- **User Feedback Loop**: Implement relevance feedback to improve retrieval quality
- **Metadata Filtering**: Add filtering by ticket attributes (browser, customer type)
- **Persistent Storage**: Move to a production-grade vector database
- **Advanced Indexing**: Implement hybrid search (semantic + keyword)
- **Multilingual Support**: Extend to support multiple languages

## 📝 Design Decisions

### Embedding Model Selection

`sentence-transformers/all-MiniLM-L6-v2` was chosen for:
- Good performance-to-resource ratio
- 384-dimensional embeddings capturing semantic information
- Support for English language queries

For production, consider larger models like `intfloat/multilingual-e5-large-instruct` for improved accuracy.

### Vector Store Choice

FAISS was selected because:
- High-performance similarity search with minimal setup
- Support for various distance metrics
- Scales well for prototyping purposes

For production systems, consider managed services like Pinecone, Weaviate, or Milvus.

### Response Generation Strategy

The system supports two modes:
- Template-based responses (no external dependencies)
- LLM-enhanced responses (when API key is provided)

This dual approach ensures the system works in various environments.

## 📄 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
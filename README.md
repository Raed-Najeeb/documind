#DocuMind

A production-grade RAG pipeline that answers questions about your documents with cited sources, zero hallucinations, and automated evaluation. 

#Evaluation Results

Faithfulness = 100%
Relevance = 84%

(Relevance resulted in 84% cause it included an intentional out-of-domain test, which the system correctly refused to answer.)

Project structure :
documind/
├── src/
│   ├── ingest.py        # PDF text extraction
│   ├── chunk.py         # Text chunking with overlap
│   ├── embed.py         # Embeddings + ChromaDB storage
│   ├── retrieve.py      # Hybrid search (BM25 + Dense) + Re-ranking
│   ├── generate.py      # LLM generation with Groq
│   ├── app.py           # FastAPI web server
│   └── evaluate.py      # LLM-as-judge evaluation suite
├── data/
│   ├── pdfs/            # Your documents
│   ├── eval_dataset.json
│   └── eval_results.json
├── Dockerfile
├── requirements.txt
└── README.md
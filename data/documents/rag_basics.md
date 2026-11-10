# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines information
retrieval with text generation from a language model.

A typical RAG system has these stages:
1. Load source documents.
2. Split documents into chunks.
3. Convert chunks into embeddings.
4. Retrieve relevant chunks for a question.
5. Pass the retrieved context to a language model.
6. Generate an answer grounded in the context.

RAG can help a model answer questions about private or
specialized documents without retraining the model.

RAG does not guarantee correctness. Retrieval can fail, and
a language model can still generate unsupported claims.
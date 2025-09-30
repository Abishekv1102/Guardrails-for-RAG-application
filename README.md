RAG Application with Guardrails

Multi-Format Ingestion: Utilizes helper functions to extract text from diverse documents, including PDF, DOCX, PPTX, and TXT files, using libraries like PyPDF2 and pptx.

RAG Pipeline Stack: Processes text chunks, creates embeddings via Google Generative AI Embeddings, and stores documents in a Chroma vector store for retrieval using a Gemini 1.5 Flash-based RAG chain.

Structured Output Control: Employs a specific system prompt to constrain the LLM's response to a maximum of three sentences and mandate the output format as a JSON object.

Security Integration: The final RAG output is routed through an external Guardrails module for validation, ensuring the answer adheres to predefined safety and structural policies before display.

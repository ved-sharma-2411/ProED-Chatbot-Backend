For each part (example part-600), these 3 core outputs are created:

part-600_sections.json
Parsed legal content by section (clean text + metadata).
This is your structured source record.

part-600_chunks.json
Chunked data used for RAG ingestion (base_chunks, level_chunks, metadata, rules).
This is the main file used for Pinecone ingest.

part-600_manifest.json
Operational metadata: file paths, counts, embedding model used, etc.
This is for tracking/audit, not retrieval.

How Pinecone storage works:

Ingest reads part-600_chunks.json (mainly base_chunks).
During ingest, text is embedded at that time (not pre-embedded in file).
Then vectors are upserted to Pinecone with:
vector id = chunk_id
vector values = embedding
metadata = content, section_ref, node_id, source_url, document_title, etc.

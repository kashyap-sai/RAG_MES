# MES RAG Model - Setup and Execution

This repository contains the code and setup for a Retrieval-Augmented Generation (RAG) system tailored for a Manufacturing Execution System (MES) in the chemical industry. The system allows users to query MES-related documents and receive contextually relevant responses, leveraging a combination of embeddings, FAISS vector database, and a fine-tuned transformer-based model for generation.

### **Project Files Overview**

- `MES_text.txt`: Sample text data related to MES.
- `MES_gmpua.pdf`: A PDF document related to MES processes.
- `Manufacturing Execution System - What is it.mp4`: A video explaining MES (audio extracted).
- `faiss_index.index`: FAISS vector index file for storing embeddings.
- `document_map.pkl`: A pickle file containing document-to-embedding mappings.

---

## **Installation & Setup**

### 1. **Clone the Repository**

Start by cloning the repository to your local machine:

```bash
git clone <repository_url>
cd <repository_folder>

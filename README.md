# ResearchXpert: Advanced Academic Research Assistant
ResearchXpert is an AI-powered academic research assistant designed to streamline the process of querying and summarizing content from academic PDFs. Leveraging state-of-the-art natural language processing (NLP) models, embedding techniques, and an intuitive user interface, ResearchXpert offers a user-friendly tool for students, researchers, and professionals to extract valuable insights from vast amounts of academic literature efficiently.

#Features
* **Automated Document Retrieval**: Upload academic PDFs, and the system processes and organizes the content into manageable chunks.
* **Context-Aware Question Answering**: Ask domain-specific questions, and the system retrieves contextually relevant answers from the uploaded PDFs.
* **Semantic Understanding**: Incorporates advanced embedding models (LLaMA 3.2) for nuanced understanding and precise results.
* **Intuitive** Interface: User-friendly GUI built with Tkinter, making it accessible to both technical and non-technical users.
* **Scalability** and Performance: Handles extensive datasets efficiently and supports deployment in resource-limited environments.
# Technology Stack
* Programming Language: Python
* Frameworks and Libraries:
* LangChain: Chaining NLP tasks for seamless processing.
* Tkinter: GUI development.
* PyPDFLoader: PDF document processing.
* FAISS: Vector store for efficient document retrieval.
* OllamaEmbeddings: Advanced semantic embedding generation.
* Model: LLaMA 3.2 for embeddings and context-aware responses.
* Additional Tools: PIL for image handling within the GUI.
# How It Works
1.PDF Upload: Load any academic PDF into the application through an intuitive file uploader.
2. Document Processing: The system preprocesses the uploaded document, splitting it into smaller, manageable chunks.
3. Embedding Generation: Generates semantic embeddings using the LLaMA 3.2 model to understand document content deeply.
4. Question Answering:
Users input questions through the GUI.
The system retrieves relevant sections using FAISS and provides a context-aware response powered by Ollama's LLM.

# Installation and Usage
## Requirements
* Python 3.8+
* Libraries: Install using `pip install -r requirements.txt`

```
langchain-community
langchain
langchain-text-splitters
langchain-ollama
FAISS
PIL
tkinter
```
# Interface

![quesrtion and answer](https://github.com/user-attachments/assets/77b3827e-b403-447f-b24f-9b72786df2e2)
![first window](https://github.com/user-attachments/assets/400f57ec-34e5-4460-bd4b-99f752f364b1)
![uploaded and chuked](https://github.com/user-attachments/assets/e4056159-fb1a-4d03-981b-0c1b5815b296)
![uploaded and asking question](https://github.com/user-attachments/assets/feaa1278-30c4-4892-a50a-12f56a9ec4bd)

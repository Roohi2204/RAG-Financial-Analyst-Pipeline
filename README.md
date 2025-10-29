# 📊 Business Intelligence Dashboard (RAG Pipeline)



## ⭐️ Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** designed to automate detailed financial reporting and risk analysis. It uses a database of regulatory documents (e.g., 10-K filings, annual reports) to provide context-aware, industry-specific answers, significantly improving reporting efficiency and accuracy for finance professionals.

The RAG pipeline integrates advanced components to deliver structured, high-quality analytical reports.

## 🛠️ Tech Stack & Architecture

* **LLM:** **Google Gemini 2.5 Flash Lite** (via `google-genai`) for generation and sophisticated analysis.
* **Vector Database:** **Chroma** (via `langchain-community`) for persistent storage and efficient similarity search.
* **Embeddings:** **HuggingFace `MiniLM`** (via `sentence-transformers`) for converting text into high-quality vector representations.
* **Data Processing:** **PyPDF2** for text extraction and **LangChain's RecursiveCharacterTextSplitter** for optimal document chunking.
* **Language:** Python

## 🚀 Getting Started

Follow these steps to set up and run the financial report generator locally.

### Prerequisites

* Python 3.8+
* A **Gemini API Key** from Google AI Studio.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd RAG-Financial-Analyst-Pipeline
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Add Data:** Create a folder named `data` in the project root and place your PDF financial documents inside it.
2.  **Configure API Key:** Open `rag_analyst.py` and replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual Gemini API Key.
3.  **Run the script:**
    ```bash
    python rag_analyst.py
    ```
    The system will build the vector database (`chroma_store` folder) and generate a multi-section financial report printed directly to your console.

### Key Features Demonstrated:

* **Custom Prompting:** Uses a highly detailed system prompt to force the LLM into the persona of a "professional financial analyst."
* **Context-Aware Retrieval:** Implements a custom `get_balanced_retrieval` method to ensure context is pulled from **multiple source documents**, preventing single-document bias.
* **Modular Reporting:** The `generate_full_report` function allows easy extension to cover different industries and analytical sections (e.g., Legal Risks, Competitive Landscape).
import os
import PyPDF2
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


class PDFQuestionAnswering:
    def _init_(self, folder_path, gemini_api_key):
        self.folder_path = folder_path
        self.gemini_api_key = gemini_api_key
        self.setup_gemini()
        self.setup_system()

    def setup_gemini(self):
        """Configure the Gemini model with the provided API key."""
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

    def extract_text_from_pdf(self, file_path):
        """Extract text content from a PDF file using PyPDF2."""
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text, True
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return "", False

    def setup_system(self):
        """Prepare the vector store with text chunks from all PDFs in the folder."""
        all_texts = []
        metadata_list = []

        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.folder_path, file_name)
                extracted_text, success = self.extract_text_from_pdf(pdf_path)
                if success:
                    all_texts.append(extracted_text)
                    metadata_list.append({"source": file_name})

        # Split text into overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        doc_chunks = text_splitter.create_documents(all_texts, metadatas=metadata_list)

        # Embed using MiniLM (384-dim vector)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create Chroma vector database
        persist_dir = "chroma_store"
        self.vectorstore = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        self.vectorstore_client = self.vectorstore.as_retriever(search_kwargs={"k": 8})

    def get_balanced_retrieval(self, question):
        """Perform balanced retrieval: one chunk per document if possible."""
        try:
            all_docs = self.vectorstore_client.invoke(question)
            source_wise_docs = {}
            for doc in all_docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in source_wise_docs:
                    source_wise_docs[source] = doc.page_content
            return "\n\n".join(source_wise_docs.values())
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return ""

    def ask_gemini(self, question, context):
        """Send a question and context to Gemini for generation."""
        try:
            prompt = f"""
You are a professional financial analyst assistant. Provide detailed, accurate responses based on the context provided. If context is missing important information (such as financial summary, company background, or legal risks), use your own knowledge to supplement it.

CONTEXT:
{context}

QUESTION:
{question}

Be thorough, formal, and include relevant financial, strategic, and legal insights.
"""
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {e}"

    def generate_full_report(self):
        """Generate a detailed multi-part report for a specific company and industry."""
        industry = "retail"  # You can change this for other industries (e.g., finance, tech)

        sections = {
            "Company Summary": f"""
Please provide a detailed summary about Walmart Inc. based in Arkansas. Include:

- Headquarters and founding details
- Founder’s name and year
- Key business segments
- Global presence
- Annual revenue (FY2023 if possible)
- Major business strategies (e.g., EDLP)
- Any known involvement in litigation, lawsuits, or legal proceedings
""",

            "Industry Risks": f"""
2a. What are common risks impacting the financial performance of companies in the {industry} industry?

2b. What are effective strategies for optimizing financial performance and profit for companies in the {industry} industry?

2c. What are strategies to optimize cash flow for companies in the {industry} industry?

2d. What are methods to accelerate revenue growth in the {industry} industry?
""",

            "Risk of Material Misstatement": f"""
3a. In the {industry} industry, which financial statement line items are considered high risk for material misstatement?

3b. What are typical audit procedures performed for these high-risk items?
""",

            "Economic Conditions": f"""
4a. How might economic conditions impact the financial performance of companies in the {industry} industry?

4b. What economic trends over the past year could affect companies in the {industry} industry?

4c. What strategies help mitigate economic risks in the {industry} industry?

4d. What are typical borrowing rates and terms currently applicable to the {industry} industry?
"""
        }

        final_output = ""

        for section, prompt in sections.items():
            print(f"\n[Generating: {section}]")
            retrieved_text = self.get_balanced_retrieval(prompt)
            answer = self.ask_gemini(prompt, retrieved_text)
            final_output += f"\n\n{section}:\n{'-' * 50}\n{answer}"

        print("\n\nFinal Report:")
        print("=" * 80)
        print(final_output)
        print("=" * 80)


def main():
    folder_path = "data"  # Change this path to your local folder containing PDFs
    gemini_api_key = "AIzaSyCJuNk7cwBmM5D6UVmusW02CYVcJOUXyL0"  # Replace with your Gemini API key
    qa = PDFQuestionAnswering(folder_path, gemini_api_key)
    qa.generate_full_report()


if _name_ == "_main_":
    main()
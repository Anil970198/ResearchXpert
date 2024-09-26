import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# GUI Application
class DocumentQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Q&A with AI")
        self.pdf_file = None
        self.model = "llama3.1"
        self.vectorstore = None

        # Upload PDF button
        self.upload_btn = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf)
        self.upload_btn.pack(pady=10)

        # Question Entry
        self.question_label = tk.Label(self.root, text="Enter your question:")
        self.question_label.pack()

        self.question_entry = tk.Entry(self.root, width=50)
        self.question_entry.pack(pady=5)

        # Ask button
        self.ask_btn = tk.Button(self.root, text="Ask", command=self.ask_question)
        self.ask_btn.pack(pady=10)

        # Answer Display
        self.answer_text = scrolledtext.ScrolledText(self.root, width=70, height=15, wrap=tk.WORD)
        self.answer_text.pack(pady=10)

    def upload_pdf(self):
        # File Dialog to select PDF
        self.pdf_file = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
        if self.pdf_file:
            self.answer_text.insert(tk.END, f"Loaded PDF: {self.pdf_file}\n")
            self.process_pdf()

    def process_pdf(self):
        # Load and split PDF into chunks
        loader = PyPDFLoader(self.pdf_file)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        embeddings = OllamaEmbeddings(model=self.model)
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

        self.answer_text.insert(tk.END, "PDF processing complete. Ready to answer questions.\n")

    def ask_question(self):
        # Get user input
        question = self.question_entry.get()
        if question and self.vectorstore:
            retriever = self.vectorstore.as_retriever()
            # Retrieve context based on the question
            context = retriever.invoke(question)

            # Answer question with the model
            prompt_template = """
            You are an assistant that provides answers to questions based on a given context.
            Answer the question based on the context. If you can't answer, reply 'I don't know'.

            Context: {context}
            Question: {question}
            """
            prompt = PromptTemplate.from_template(prompt_template)
            formatted_prompt = prompt.format(context=context, question=question)

            model = ChatOllama(model=self.model, temperature=0)
            parser = StrOutputParser()
            chain = model | parser
            answer = chain.invoke(formatted_prompt)

            # Display the answer in the text area
            self.answer_text.insert(tk.END, f"Q: {question}\nA: {answer}\n\n")
        else:
            self.answer_text.insert(tk.END, "Please upload a PDF and enter a question.\n")

# Main Function to Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentQAApp(root)
    root.mainloop()


 def ask_question(self):
        if not self.pdf_processed:
            self.answer_text.insert(tk.END, "Please upload and process a PDF first.\n")
            return
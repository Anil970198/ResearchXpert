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



class ResXprt:
    def __init__(self, root):
        self.root = root
        self.root.title("ResearchXpert")
        self.pdf_file = "lesson.pdf"
        self.model = "llama2"
        self.vectorstore = None

        # Upload PDF
        self.upload_btn = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf)
        self.upload_btn.pack(pady=10)

        # Question Entry
        self.question_label = tk.Label(self.root, text="Enter your question:")
        self.question_label.pack()

        self.question_entry = tk.Entry(self.root, width=50)
        self.question_entry.pack(expand=True, fill=tk.BOTH)

        # Ask button
        self.ask_btn = tk.Button(self.root, text="Ask", command=self.ask_question)
        self.ask_btn.pack(pady=10)

        # Answer display
        self.answer_text = scrolledtext.ScrolledText(self.root, width=70, height=15, wrap=tk.WORD)
        self.answer_text.pack(expand=True, fill=tk.BOTH)

    def upload_pdf(self):
        # File dialog to select a file
        self.pdf_file = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
        if self.pdf_file:
            self.answer_text.insert(tk.END, f"Loaded PDF: {self.pdf_file}\n")
            self.process_pdf()

    def process_pdf(self):
        loader = PyPDFLoader(self.pdf_file)
        pages = loader.load()
        self.answer_text.insert(tk.END, "PDF has been loaded successfully.\n\n")

        self.answer_text.insert(tk.END, "Document chunking in progress.\n")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        self.answer_text.insert(tk.END, "Chunks have been prepared.\n\n")

        self.answer_text.insert(tk.END, "Initiating embeddings.\n")
        embeddings = OllamaEmbeddings(model=self.model)
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.answer_text.insert(tk.END, "Embeddings processed.\n\n")

        self.answer_text.insert(tk.END, "PDF processing complete. Ready to answer questions.\n")
        self.answer_text.insert(tk.END, "----------------------------\n")


    def ask_question(self):
        question = self.question_entry.get()
        if question and self.vectorstore:
            retriever = self.vectorstore.as_retriever()
            context = retriever.invoke(question)
            prompt_template = """
                        You are an assistant that provides answers to questions based on a given context.
                        Answer the question based on the context. If you can't answer, reply 'I don't know'.

                        Context: {context}
                        Question: {question}
                        """
            prompt = PromptTemplate.from_template(prompt_template)
            true_prompt = prompt.format(context=context, question=question)

            model = ChatOllama(model=self.model)
            parser = StrOutputParser()
            chain = model | parser
            answer = chain.invoke(true_prompt)

            self.answer_text.insert(tk.END, f"Question: {question}\n Answer: {answer}\n\n\n\n")
        else:
            self.answer_text.insert(tk.END, "Please upload a pdf and enter a question.\n\n\n\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResXprt(root)
    root.mainloop()

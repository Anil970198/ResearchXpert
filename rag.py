import tkinter as tk

PDF_FILE = "lesson.pdf"
MODEL = "llama2"

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(PDF_FILE)
pages = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

chunks = splitter.split_documents(pages)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model= MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()
from langchain_ollama import ChatOllama
model = ChatOllama(model = MODEL)

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()


from langchain.prompts import PromptTemplate
template = """
You are an assistant that provides answers to questions based on
a given context. 

Answer the question based on the context. If you can't answer the
question, reply "I don't know".

Be as concise as possible and go straight to the point.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

from operator import itemgetter
chain = (
    {
        "context" : itemgetter("question") | retriever,
        "question" : itemgetter("question"),
    }
    | prompt
    | model
    | parser
)

print("ok!!")
questions = [
    "summarize transfer learning "
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {chain.invoke({'question': question})}")
    print("*************************\n")
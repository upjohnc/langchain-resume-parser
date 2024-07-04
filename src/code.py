from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.stores import InMemoryStore


def get_model():
    return Ollama(model="llama3")


def load_docs():
    pdf_folder_path = "./docs"
    documents = []
    for file in Path(pdf_folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(file.resolve()))
        doc = loader.load()
        if len(doc[0].page_content) == 0:
            print(
                f"`{file.name}` is not loaded because pypdf doesn't not parse it correctly"
            )
        documents.extend(loader.load())
    return documents


def split_text(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs_split = text_splitter.split_documents(docs)

    return docs_split


# def get_vector_store(docs, embedding):
#     vector_store = FAISS.from_documents(docs, embedding)
#     return vector_store


def get_retriever(docs, embedding):
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store.as_retriever()


# def get_parent_doc_retriever():
# docs, text_splitter = get_docs(chunk_size)
# embeddings = OllamaEmbeddings(model="llama3")
# vector_store = Chroma(
#     collection_name="full_documents", embedding_function=embeddings
# )
# docstore = InMemoryStore()
# retriever = ParentDocumentRetriever(
#     vectorstore=vector_store, docstore=docstore, child_splitter=text_splitter
# )
# retriever.add_documents(docs, ids=None)
# return retriever


def get_prompt():
    prompt_template = """
    As a recruiter, you need to recommend a candidate.If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}

    With the answer include the following sections:
    Candidate:
    Candidate Email Address:
    Why the candidate is suitable:
    ---------
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    query_template = (
        "Given this Job requirements:{job_requirements}"
        " , Please return the name of the candidate"
        " and a summary of less than 200 words of the candidate's suitability for this position:"
    )
    return prompt, query_template


def get_stuff_chain(prompt):
    docs = load_docs()
    docs = split_text(docs)
    chain_type_kwargs = {"prompt": prompt}
    embedding = OllamaEmbeddings(model="llama3")

    retriever = get_retriever(docs, embedding)
    llm = get_model()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa_chain


if __name__ == "__main__":

    prompt, query_template = get_prompt()
    qa_chain = get_stuff_chain(prompt)

    job_description = "post=data engineer, competencies=serverless, snowflake, python"
    query = query_template.format(job_requirements=job_description)
    response = qa_chain.invoke({"query": query})

    print(response["result"])

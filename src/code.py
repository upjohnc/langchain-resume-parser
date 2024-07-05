from pathlib import Path

import click
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStoreRetriever


def get_model() -> BaseLLM:
    return Ollama(model="llama3", temperature=0.2)


def load_docs() -> list[Document]:
    pdf_folder_path = "./docs"
    documents = []
    for file in Path(pdf_folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(file.resolve()))
        doc = loader.load()
        if len(doc[0].page_content) == 0:
            click.echo(
                click.style(
                    f"`{file.name}` is not loaded because pypdf does not parse it correctly\n",
                    fg="bright_blue",
                )
            )
        documents.extend(loader.load())
    return documents


def split_text(docs: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs_split = text_splitter.split_documents(docs)

    return docs_split


def get_retriever(docs: list[Document], embedding: Embeddings) -> VectorStoreRetriever:
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store.as_retriever()


def get_parent_doc_retriever(docs: list[Document], embedding: Embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    vector_store = Chroma(
        collection_name="full_documents", embedding_function=embedding
    )
    docstore = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store, docstore=docstore, child_splitter=text_splitter
    )
    retriever.add_documents(docs, ids=None)
    return retriever


def get_prompt() -> tuple[ChatPromptTemplate, str]:
    prompt_template = """
    As a recruiter, you need to recommend a candidate.If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {input}

    With the answer include the following sections:
    Candidate:
    Candidate Email Address:
    Why the candidate is suitable:
    ---------
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    query_template = (
        "Given this Job requirements:{job_requirements}"
        " , Please return the name of the candidate"
        " and a summary of less than 200 words of the candidate's suitability for this position:"
    )
    return prompt, query_template


def get_stuff_chain(prompt: ChatPromptTemplate) -> Runnable:
    docs = load_docs()
    docs = split_text(docs)
    embedding = OllamaEmbeddings(model="llama3")

    retriever = get_parent_doc_retriever(docs, embedding)
    llm = get_model()
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)

    return retrieval_chain


@click.command
@click.option("-t", "--job_title", help="Title of job for the posting", required=True)
@click.option(
    "-s", "--job_skills", help="List of skills. Comma separated", required=True
)
def main(job_title, job_skills):
    prompt, query_template = get_prompt()
    qa_chain = get_stuff_chain(prompt)

    job_description = f"post={job_title}, competencies={job_skills}"
    query = query_template.format(job_requirements=job_description)
    response = qa_chain.invoke({"input": query})

    click.echo(click.style("Parameters searching for:", fg="bright_green"))
    click.echo(f"Job Title: {job_title}")
    click.echo(f"Skills: {job_skills}")
    click.echo("")

    click.echo(click.style("Response:", fg="bright_green"))
    click.echo(response["answer"])


if __name__ == "__main__":
    main()

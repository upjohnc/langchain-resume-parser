from langchain.chains.question_answering.chain import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

loader = PyPDFLoader("./docs/BrettschneiderResume.pdf")
docs = loader.load_and_split()

model = Ollama(model="llama3")
chain = load_qa_chain(llm=model)

query = "What are the skills of Steve"
response = chain.invoke(input=dict(input_documents=docs, question=query))
print(response["output_text"])

import getpass
import os
import getpass
import os


from langchain import hub

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

OPENAI_API_KEY = "hannah"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
#LANGCHAIN_API_KEY =  "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"

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

example_messages

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})

url_list = ['2023-06-23-agent/', '2023-03-15-prompt-engineering/', '2023-10-25-adv-attack-llm/']
docs = {}
for paths in url_list:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/"+paths,),
        session=session,  # 여기에 session 전달
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs[paths] = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n\n\n"],chunk_size=2000, chunk_overlap=200, length_function=len,
                                                   is_separator_regex=False)
    splits = text_splitter.split_documents(docs[paths])

splits[0]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("hannah",rag_chain.invoke("What is Task Decomposition?"))

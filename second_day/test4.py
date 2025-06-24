import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

# Load, chunk and index the contents of the blog.

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

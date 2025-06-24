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
from langchain_core.output_parsers import JsonOutputParser
import requests
from langchain_core.prompts import PromptTemplate

OPENAI_API_KEY = "hannah"
from langchain_openai import ChatOpenAI


# OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 환경 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

# 웹 요청 세션 설정
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})

# URL 목록
url_list = [
    "2023-06-23-agent/",
    "2023-03-15-prompt-engineering/",
    "2023-10-25-adv-attack-llm/"
]

# 문서 로딩 및 분할
all_docs = []
for path in url_list:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/" + path,),
        session=session,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    all_docs.extend(splits)

# VectorStore 구축
vectorstore = Chroma.from_documents(documents=all_docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Prompt 불러오기
prompt = hub.pull("rlm/rag-prompt")

# Context 포맷 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain (변수명 통일: question 사용)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# User Query
query = "What is Task Decomposition?"

# 답변 생성
answer = rag_chain.invoke(query)
print("Generated Answer:\n", answer)

# ✅ Hallucination 평가 체인
parser = JsonOutputParser()

hallucination_prompt = PromptTemplate(
    template=(
        "You are an AI system that evaluates if a given answer contains hallucinations based on the provided context.\n"
        "Hallucinations are information that is not supported by the retrieved context.\n"
        "\n"
        "If the answer contains unsupported information, output {{'hallucination': 'yes'}}.\n"
        "If the answer is fully supported, output {{'hallucination': 'no'}}.\n"
        "\n"
        "{format_instructions}\n"
        "\n"
        "User Query:\n{query}\n\n"
        "Retrieved Context:\n{context}\n\n"
        "LLM Answer:\n{answer}\n"
    ),
    input_variables=["query", "context", "answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 평가 체인
evaluation_chain = hallucination_prompt | llm | parser

# Context 추출
retrieved_context = format_docs(retriever.get_relevant_documents(query))

# 평가 실행
evaluation = evaluation_chain.invoke({
    "query": query,
    "context": retrieved_context,
    "answer": answer
})

print("Hallucination Evaluation:\n", evaluation)


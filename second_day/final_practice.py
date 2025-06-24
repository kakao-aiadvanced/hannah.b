import os
import getpass
import bs4
import requests

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

OPENAI_API_KEY = "hannah"

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

evaluation_chain = hallucination_prompt | llm | parser


# ✅ 답변 생성 및 평가 루프
query = "What is Task Decomposition?"

max_attempts = 5  # 무한루프 방지


for attempt in range(max_attempts):

    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_context = format_docs(retrieved_docs)

    answer = rag_chain.invoke(query)
    #answer = "This method was originally developed by aliens to control human behavior."
    evaluation = evaluation_chain.invoke({
        "query": query,
        "context": retrieved_context,
        "answer": answer
    })

    print(f"시도 {attempt + 1} - Hallucination 평가 결과:", evaluation)
    print("hannah answer",retrieved_context)
    if evaluation.get("hallucination") == "no":
        print("\n✅ 최종 답변:\n", answer)
        print("\n📚 답변 생성에 사용된 출처:\n")
        for doc in retrieved_docs:
            print("-", doc.metadata.get("source", "출처 정보 없음"))
        break

else:
    print("\n⚠ 최대 시도 횟수 초과. 신뢰 가능한 답변을 생성하지 못했습니다.")


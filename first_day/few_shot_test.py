from openai import OpenAI

# 클라이언트 생성
client = OpenAI()

# Few-Shot Prompt와 추가 질문까지 포함
messages = [
    {"role": "system", "content": "너는 영어를 한국어로 번역하는 번역가야."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "안녕하세요, 잘 지내세요?"},
    {"role": "user", "content": "I love programming."},
    {"role": "assistant", "content": "나는 프로그래밍을 좋아합니다."}
]

# 추가 질문
messages.append({"role": "user", "content": "Where do you live?"})

# API 호출
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# 답변 출력
print(completion.choices[0].message.content)


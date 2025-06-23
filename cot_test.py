from openai import OpenAI

# 클라이언트 생성
client = OpenAI()

# CoT Few-Shot 예시 포함
messages = [
    {"role": "user", "content": "Q: 철수는 사과를 3개 가지고 있었고, 2개를 더 샀습니다. 철수는 지금 사과가 몇 개 있나요? 답을 생각의 흐름과 함께 말해주세요."},
    {"role": "assistant", "content": "먼저, 철수가 가지고 있던 사과는 3개입니다. 그리고 2개를 더 샀으니, 총 3 + 2 = 5개입니다. 답은 5개입니다."},
    
    {"role": "user", "content": "Q: 지영이는 책을 4권 가지고 있었고, 친구가 3권을 더 줬습니다. 지영이는 지금 책이 몇 권 있나요? 답을 생각의 흐름과 함께 말해주세요."},
    {"role": "assistant", "content": "지영이는 원래 4권의 책을 가지고 있었습니다. 친구가 3권을 더 줬으니, 4 + 3 = 7권입니다. 답은 7권입니다."},
]

# 추가 질문 (CoT 유도)
messages.append({
    "role": "user",
    "content": "Q: 민수는 연필을 5자루 가지고 있었고, 4자루를 잃어버렸습니다. 민수는 지금 연필이 몇 자루 있나요? 답을 생각의 흐름과 함께 말해주세요."
})

# API 호출
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# 답변 출력
print(completion.choices[0].message.content)


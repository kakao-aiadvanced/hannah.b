from openai import OpenAI
client = OpenAI()

# Few-shot 예시: 한국어-영어 번역
translation_examples = [
    {
        "role": "system",
        "content": "당신은 한국어를 영어로 번역하는 전문가입니다. 자연스럽고 정확한 영어로 번역해주세요. 문맥을 고려하여 적절한 어조와 스타일을 유지해주세요."
    },
    {
        "role": "user",
        "content": "안녕하세요, 만나서 반갑습니다."
    },
    {
        "role": "assistant",
        "content": "Hello, nice to meet you."
    },
    {
        "role": "user",
        "content": "오늘 날씨가 정말 좋네요."
    },
    {
        "role": "assistant",
        "content": "The weather is really nice today."
    },
    {
        "role": "user",
        "content": "저는 한국 음식을 좋아합니다."
    },
    {
        "role": "assistant",
        "content": "I like Korean food."
    }
]

# 새로운 번역할 문장
new_text = "내일 친구들과 함께 영화를 보러 갈 예정입니다."

# Few-shot 예시들과 새로운 문장을 합쳐서 전송
messages = translation_examples + [{"role": "user", "content": new_text}]

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.1  # 일관된 번역을 위해 낮은 temperature
)

print("=== 번역 Few-shot 결과 ===")
print(f"원문: {new_text}")
print(f"번역: {completion.choices[0].message.content}") 
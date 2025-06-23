from openai import OpenAI
client = OpenAI()

# Few-shot 예시: 감정 분석
sentiment_examples = [
    {
        "role": "system",
        "content": "당신은 텍스트의 감정을 분석하는 전문가입니다. 주어진 문장의 감정을 '긍정', '부정', '중립' 중 하나로 분류하고, 감정 강도(1-5)와 이유를 함께 설명해주세요."
    },
    {
        "role": "user",
        "content": "오늘 날씨가 정말 좋아서 기분이 상쾌해요."
    },
    {
        "role": "assistant",
        "content": "감정: 긍정\n강도: 4/5\n이유: 날씨가 좋고 기분이 상쾌하다는 표현에서 명확한 긍정적 감정이 드러남"
    },
    {
        "role": "user",
        "content": "시험에서 떨어져서 너무 슬퍼요."
    },
    {
        "role": "assistant",
        "content": "감정: 부정\n강도: 4/5\n이유: 시험 실패와 슬픔이라는 표현에서 강한 부정적 감정이 드러남"
    },
    {
        "role": "user",
        "content": "내일 비가 올 것 같아요."
    },
    {
        "role": "assistant",
        "content": "감정: 중립\n강도: 2/5\n이유: 단순한 날씨 예보로 특별한 감정적 색채가 없음"
    }
]

# 새로운 분석할 문장
new_text = "친구와 함께 맛있는 음식을 먹어서 정말 행복했어요!"

# Few-shot 예시들과 새로운 문장을 합쳐서 전송
messages = sentiment_examples + [{"role": "user", "content": new_text}]

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.3  # 일관성 있는 분석을 위해 낮은 temperature
)

print("=== 감정 분석 Few-shot 결과 ===")
print(completion.choices[0].message.content) 
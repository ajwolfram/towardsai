from dotenv import load_dotenv

load_dotenv("../.env")
from openai import OpenAI  # noqa: E402

ENGLISH_TEXT = "Hello, how are you?"


def translate_english_to_french(eng_txt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful translation assistant."},
            {
                "role": "user",
                "content": (
                    "Translate the following English text to French: "
                    f'''"{eng_txt}"'''
                ),
            },
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    french_text = translate_english_to_french(ENGLISH_TEXT)
    print(french_text)

from dotenv import load_dotenv

load_dotenv("../.env")
import cohere


def run_demo():
    co = cohere.Client()
    resp = co.chat(
        chat_history=[
            {"role": "USER", "message": "Who discovered gravity?"},
            {"role": "CHATBOT", "message": "The man who is widely credited with discovering gravity is Sir Isaac Newton"},
        ],
        message="What year was he born?",
        max_tokens=100,  # added to limit the extreme amount of text that is returned if removed
        connectors=[{"id": "web-search"}],
    )

    print(resp.text)


if __name__ == "__main__":
    run_demo()

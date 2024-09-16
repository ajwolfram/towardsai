from dotenv import load_dotenv

load_dotenv("../.env")
from openai import OpenAI  # noqa: E402

prompt = "Describe the movie {movie} using emojis"
examples = [
    {
        "input": "Titanic",
        "output": "🚢🌊💑🧊💥💀🥶🚪🤝👩‍🦰💎💙",
    },
    {
        "input": "The Matrix",
        "output": "⚡💻🧠👨‍💻👨💊☝🛜",
    },
]


def describe_movie_with_emojis(movie):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You know everything there is to know about movies."},
            {"role": "user", "content": prompt.format(movie=examples[0]["input"])},
            {"role": "assistant", "content": examples[0]["output"]},
            {"role": "user", "content": prompt.format(movie=examples[1]["input"])},
            {"role": "assistant", "content": examples[1]["output"]},
            {"role": "user", "content": prompt.format(movie=movie)},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    movie = "Raising Arizona"
    movie_emojis = describe_movie_with_emojis(movie)
    print(movie_emojis)

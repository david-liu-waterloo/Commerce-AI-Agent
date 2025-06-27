from openai import OpenAI
from dotenv import load_dotenv

# TODO: 
# - get commercial website data set
#   - check out this data set: https://huggingface.co/datasets/milistu/AMAZON-Products-2023
#   - also this data set: https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset
# - integrate langchain conversation support
# - need vectorization (check out ChromaDB)
# - design UI in streamlit (must support text and image input)

load_dotenv() # to load environmental variables

def main():
    client = OpenAI()

    while True:
        query = input("Ask ChatGPT: ")

        if query.lower() in ["bye", "goodbye"]:
            break

        response = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content" : query}]
            )

        print(response.output_text)

if __name__ == "__main__":
    main()
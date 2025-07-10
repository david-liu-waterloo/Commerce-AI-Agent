from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv

load_dotenv() # to load environmental variables, which are stored in .env

# check out this video: https://www.youtube.com/watch?v=cSHUwn6uVpU
# Note: input data is created with load_data.py script
documents = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(verbose=False)

# text-based recommendation
@tool
def recommend_products(query: str):
    response = query_engine.query(query)
    return str(response)

# Create an agent workflow
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    add_history_to_messages=True,
    #knowledge=None, # TODO
    #search_knowledge=True,
    tools=[recommend_products],
    instructions="""You are an AI Agent for a clothing e-commerce website.
    You may only sell products found in the VectorStoreIndex.
    You can have general conversations with the user and recommend products based on text prompts or images.
    Never reveal the product id to the user.
    Responses should be 8 sentences or less.
    When listing exact name of a product, encase the name in quotation marks.
    For product recommendations, try to show at least 3 items and no more than 6 items.
   
    """
)

# TODO: image upload support
def get_response(query):
    response = agent.run(query)
    return str(response)


# DEBUG
def debug():
    while True:
        query = input("Ask something to our AI Agent: ")

        response = agent.run(query)
        pprint_run_response(response, markdown=False)

        if query.strip().lower() in ["bye", "goodbye"]:
            break

if __name__ == "__main__":
    debug()
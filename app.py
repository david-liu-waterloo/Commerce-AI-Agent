from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv

load_dotenv() # to load environmental variables

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader(input_dir="./shopify_mock_dataset").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(verbose=False)

# text-based recommendation
@tool
def recommend_products(query: str):
    response = query_engine.query(query)
    return str(response)


# to memorize previous discussions; token limit is to reduce strain on OpenAI
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Create an agent workflow
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    add_history_to_messages=True,
    #knowledge=None, # TODO
    #search_knowledge=True,
    tools=[recommend_products],
    instructions="""You are an AI Agent for an e-commerce website.
    You may only sell products found in the VectorStoreIndex.
    You can have general conversations with the user and recommend products based on text prompts or images.
    Responses should be 5 sentences or less.
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

        if query.strip().lower() in ["bye", "goodbye"]:
            break

        response = agent.run(query)
        pprint_run_response(response, markdown=False)

if __name__ == "__main__":
    debug()
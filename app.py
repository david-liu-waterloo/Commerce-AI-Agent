import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from dotenv import load_dotenv

# TODO: 
# - image input to AI
# - design UI in streamlit (must support text and image input)

load_dotenv() # to load environmental variables

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader(input_dir="./shopify_mock_dataset").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(verbose=False)

# text-based recommendation function
def recommend(query: str) -> str:
    response = query_engine.query(query)
    return str(response)

tools=[
    recommend
]

# Create an agent workflow
agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are an AI Agent for an e-commerce website.
    This website only sells the products in this VectorStoreIndex.
    You can have general conversations with the user (while remembering context) and recommend products based on text prompts or images.
    Responses should be 5 sentences or less.""",
)
# to memorize previous discussions
ctx = Context(agent)

async def main():
    # Run the agent
    while True:
        query = input("Say something to ChatGPT: ")

        if query in ["bye", "goodbye"]:
            break

        response = await agent.run(query, ctx=ctx)
        print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
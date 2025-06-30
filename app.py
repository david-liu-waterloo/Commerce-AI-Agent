from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv

# TODO: 
# - image input to AI
# - design UI in streamlit (must support text and image input)

load_dotenv() # to load environmental variables

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader(input_dir="./shopify_mock_dataset").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(verbose=False)

# text-based recommendation
tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="ProductQA",
        description="Answer questions about product details or recommendations."
    ),
)


# to memorize previous discussions; token limit is to reduce strain on OpenAI
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Create an agent workflow
agent = OpenAIAgent.from_tools(
    tools=[tool],
    memory=memory,
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=False,
    system_prompt="""You are an AI Agent for an e-commerce website.
    This website only sells the products in this VectorStoreIndex.
    You can have general conversations with the user (while remembering context) and recommend products based on text prompts or images.
    Responses should be 5 sentences or less.""",
)

def get_response(text):
    response = agent.chat(text)
    return response

# DEBUG
# def main():
#     while True:
#         query = input("[DEBUG] Say something to ChatGPT: ")

#         if query in ["bye", "goodbye"]:
#             break

#         response = agent.chat(query)
#         print(response)

# if __name__ == "__main__":
#     main()
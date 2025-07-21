from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from agno.agent import Agent
from agno.tools import tool
from agno.media import Image as AgnoImage
from agno.models.openai import OpenAIChat
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from dotenv import load_dotenv

TEMP_IMAGE_FILE = "./input_image.jpg"

load_dotenv() # to load environmental variables, which are stored in .env

# *** IMAGE SEARCH ***
image_model = SentenceTransformer("clip-ViT-B-32")
images = pd.read_csv("./data/data.csv")["image"]
image_embeddings = image_model.encode([img for img in images])

# *** TEXT-BASED RECOMMENDATIONS ***
# Create embeddings for RAG using LlamaIndex (Note: input data is created with load_data.py script; depends on global state unfortunately)
documents = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=6)

# *** LLM TOOLS ***
# text-based recommendation tool
@tool(
        description="text-based product recommender",
        instructions="Recommend between 3-6 products from the VectorStoreIndex that match user demands."
)
def recommend_products(query: str):
    nodes = retriever.retrieve(query)
    return "\n".join([n.node.text for n in nodes])

# image-based product search
@tool(
        description="image-based product search",
        instructions="Find the top 5 most similar products based on input image. Always use this tool when an image input is provided, and never use this tool if no image is provided."
)
def image_search():
    # TODO: fix input_image param
    query_image = PILImage.open(TEMP_IMAGE_FILE)
    #query_image = PILImage.fromarray(BytesIO(image_bytes), mode="rgba")
    query_embedding = image_model.encode([query_image])
    results = util.semantic_search(query_embedding, image_embeddings, top_k=5)[0]
    return results

# *** AGENT WORKFLOW ***
# (Note: agno Agent already handles general conversations)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an AI Agent for a fashion e-commerce website that sells clothing and accessories to men and women.",
    add_history_to_messages=True,
    num_history_responses=5, # WARNING: AI Agent only remembers past 5 messages in session (NO LONG-TERM MEMORY, which requires persistent storage)
    tools=[recommend_products, image_search],
    tool_choice="auto",
    reasoning=True,
    show_tool_calls=True,
    stream=False,
    instructions="""You may only sell and recommend products from the VectorStoreIndex.
    You can have general conversations with the user and recommend products based on text prompts or images.
    Product recommendations should include at least 3 options and no more than 6 options.
    Only use tool "image_search" if an image is provided. 
    When an image is provided, use tool "image_search" to identify matching products, then use tool "recommend_products" to answer questions using metadata from those products.
    When listing the exact name of a product, encase the name in quotation marks.
    Responses should be 8 sentences or less.
    Never reveal the product id to the user.
    """
)

def send_to_ai_agent(query) -> dict:
    # image_param_input = None
    # if query.files:
    #     raw_bytes = query.files[0].read()
    #     encoded = base64.b64encode(raw_bytes).decode("utf-8")
    #     image_param_input = {"image_base64": encoded}

    # get image if attached
    image_param_input = None
    if query.files:
        image_data = query.files[0].read()
        with open(TEMP_IMAGE_FILE, 'wb') as f: # save this image anywhere, such as "/tmp" or equivalent
            f.write(image_data)
        image_param_input = [AgnoImage(filepath=TEMP_IMAGE_FILE)] #, content=image_data)]

    # ask AI agent!
    try:
        response = agent.run(
            message=query.text,
            images=image_param_input
            #images=image_param_input # for the LLM to comment on in case of error in LLM tools
            #input=image_bytes_input
        )
    finally:
        if os.path.exists(TEMP_IMAGE_FILE):
            os.remove(TEMP_IMAGE_FILE)

    useful_data = {
        "text": response.content,
        "images": response.images
    }

    return useful_data


# DEBUG
def debug():
    while True:
        query = input("Ask something to our AI Agent: ")

        agent.print_response(query)

        if query.strip().lower() in ["bye", "goodbye"]:
            break

if __name__ == "__main__":
    debug()
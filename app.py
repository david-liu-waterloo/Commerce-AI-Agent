from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from agno.agent import Agent
from agno.tools import tool
from agno.media import Image as AMImage
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

load_dotenv() # to load environmental variables, which are stored in .env

# *** IMAGE SEARCH ***
image_model = SentenceTransformer("clip-ViT-B-32")
images = pd.read_csv("./data/data.csv")["image"]
image_embeddings = image_model.encode([img for img in images])

# *** TEXT-BASED RECOMMENDATIONS ***
# Create embeddings for RAG using LlamaIndex (Note: input data is created with load_data.py script; depends on global state unfortunately)
documents = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(verbose=False)

# *** LLM TOOLS ***
# text-based recommendation tool
@tool(
        description="Recommend products from VectorStoreIndex",
        instructions="Recommend between 3-6 products from the VectorStoreIndex that match user demands."
)
def recommend_products(query: str):
    response = query_engine.query(query)
    return str(response)

# image-based product search
@tool(
        description="Find the top 5 most similar products based on input image",
        instructions="Use this tool to find visually similar products given a product image. Always use this tool when an image input is provided."
)
def image_search(image_bytes: bytes):
    #query_image = PILImage.frombytes(mode="rgba", size=(128, 128), data=image_bytes)
    # TODO: fix input_image param
    query_image = PILImage.open(BytesIO(image_bytes))
    #query_image = PILImage.fromarray(BytesIO(image_bytes), mode="rgba")
    query_embedding = image_model.encode([query_image])

    # Compute cosine similarity
    results = util.semantic_search(query_embedding, image_embeddings, top_k=5)[0]
    return results

# *** AGENT WORKFLOW ***
# (Note: agno Agent already handles general conversations)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    add_history_to_messages=True,
    num_history_responses=5, # WARNING: AI Agent only remembers past 5 messages in session (NO LONG-TERM MEMORY, which requires persistent storage)
    tools=[], #[recommend_products, image_search],
    instructions="""You are an AI Agent for a fashion e-commerce website that sells clothing and accessories to men and women.
    You may only sell and recommend products from the VectorStoreIndex.
    You can have general conversations with the user and recommend products based on text prompts or images.
    Product recommendations should include at least 3 options and no more than 6 options.
    When listing the exact name of a product, encase the name in quotation marks.
    Responses should be 8 sentences or less.
    Never reveal the product id to the user.
    """
)

def send_to_ai_agent(query) -> dict:
    # get image URL if image was attached
    image_bytes_input = None
    if query.files:
        image_bytes_input = {"image_bytes": query.files[0].getvalue()}
    
    # image = PILImage.open(BytesIO(image_bytes))
    # print("PIL: ", image)
    # return { "text": query.text, "files": image if image_bytes else None }
    
    # ask AI agent!
    response = agent.run(
        message=query.text,
        input=image_bytes_input
    )

    useful_data = {
        "text": response.content,
        "images": response.images
    }

    return useful_data


# # DEBUG
# def debug():
#     while True:
#         query = input("Ask something to our AI Agent: ")

#         response = agent.run(query)
#         pprint_run_response(response, markdown=False)

#         if query.strip().lower() in ["bye", "goodbye"]:
#             break

# if __name__ == "__main__":
#     debug()
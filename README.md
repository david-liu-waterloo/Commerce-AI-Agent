# Commerce-AI-Agent
This repo contains an AI agent for an e-commerce site, based on OpenAI's gpt-4o model. This app primarily uses LlamaIndex to create embeddings for Retrieval-Augmented Generation (RAG), agno for AI agents, and sentence-transformers for image-based searches. To use this app, simply type in the text query in the chat input field and possibly attach one .jpg image, and the AI agent will provide a response based on the embeddings.

## Features
- General Conversation: the agent can respond to icebreaker questions in a professional manner, and features a sliding window memory for the past 5 messages.
- Text-Based Product Recommendation: ask the agent for product recommendations, and the agent will recommend around 3-6 products from the RAG index. 
- Image-Based Product Search: users can upload a single .jpg image and the agent will try to find the associated product or a group of similar products from the RAG index.
- Frontend User Interface: built in streamlit.

## Setup
1. Download this Repo: use git clone in your chosen directory.
2. Set up your OpenAI API Key: obtain an API key by signing up to OpenAI, then put the key inside .env file (OPENAI_API_KEY=sk-proj...).
3. Ensure you have Python 3.8 or Higher: ensure Python is installed, and that the version is high enough.
4. Set up Virtual Environment: create and start a virtual environment, then install required packages with: `pip install -r requirements.txt`
5. Generate Dataset: create mock dataset (may take a while): `python load_data.py`
6. Deploy with Streamlit: start up the app: `streamlit run frontend.py`
7. Use App: Enter text queries in the chat input field and click on the paperclip to attach a .jpg image.

## Overview of Files
- app.py: contains the backend logic, including initialization of the AI agent, RAG index, and image embeddings.
- frontend.py: contains all frontend logic for the User Interface, mainly streamlit functions.
- load_data.py: helper script to create a local csv dataset

## Agent API (Tool Calls)
- recommend_products(query): searches the RAG index and finds a maximum of 6 unique products related to the input query.
- image_search(): takes an input image, encodes it, and searches for similar images within the image_embeddings. The results are passed to recommend_products to fetch the associated product data.

## The Technology Stack
The frontend (User-Interface) is designed entirely using streamlit (https://streamlit.io/) due to streamlit's ease of use, ease of deployment, and ability to support file uploads for image searches.

For the backend, the AI agent is created using agno (https://pypi.org/project/agno/1.1.1/), a lightweight framework specialized for AI agents. The agno agent handles general conversation, supports reasoning, maintains short-term session memory for the past 5 messages, and runs the agent. Text-based product recommendations and image-based product searches are implemented in their respective custom-built tools, recommend_products and image_search. Overall, the agno agent is the backbone of the backend.

Note: when an image input is provided, the backend saves the image to a temporary file first, then the image_search tool opens the image and processes all bytes. The image data can be passed to the agno agent agent.run() directly; however, I have found that sending the raw image data to the agno agent consumes a heavy number of tokens and eventually resorted to the temporary file method due to budget issues during image tests. I think the temporary file method is an ugly solution, but it is adequate for the demonstration, and I am out of budget for a better solution. Images are saved in the working directory of this project, but in a real-world setting they can be stored in a temporary directory such as "/tmp" on Linux or "%userprofile%\AppData\Local\Temp" on Windows.

The AI agent can work with specific e-commerce datasets since it supports Retrieval Augmented Generation (RAG). All RAG functionality is provided by LlamaIndex (https://www.llamaindex.ai/). Assuming all e-commerce product data is stored in csv format, LlamaIndex reads the csv file using SimpleDirectoryReader(), creates the knowledge base/embeddings using VectorStoreIndex, and uses a retriever to get items from the embeddings. (As of now, the embeddings are only in memory, but persistent storage can be added in a future update.) To access the RAG index/embeddings, the agno AI agent uses the product_search tool, which queries the embeddings and returns up to 6 records related to the user input. The RAG is an essential feature as it grounds LLM responses to an external knowledge base, allowing the AI agent to provide far more accurate, detailed, and nuanced responses in relation to the e-commerce website's inventory.

Image-based product search is implemented with multiple libraries. All image objects are handled using the Pillow library (https://pillow.readthedocs.io/en/stable/reference/Image.html). There are two main components: an in-memory store containing embeddings of all images, and an image_search tool that takes an input image and finds up to 6 similar images. A combination of Sentence Transformers (https://pypi.org/project/sentence-transformers/) and Pandas (https://pandas.pydata.org/) are used to implement these components. For the in-memory store, the Pandas library is used to extract images from the dataset, while the Sentence Transformers library converts images to vector embeddings and stores the embeddings in an image index. For the image searches, an image_search tool is provided to the agno agent that converts an input image into a vector embedding, then uses a semantic search based on cosine similarity to find similar images from the image index. These findings are then passed into the recommend_products tool to retrieve each image's associated text data (i.e.: name, category, price, etc.) from the RAG embeddings.

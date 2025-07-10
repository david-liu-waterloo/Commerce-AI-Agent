from datasets import load_dataset

dataset = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=False)
dataset = dataset.remove_columns(["image"]) # images stored separately

# data (for text-based product recommendations)
open("./data/data.csv", 'w')
dataset.to_csv("./data/data.csv")

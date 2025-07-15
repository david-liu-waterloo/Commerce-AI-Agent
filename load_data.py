from datasets import load_dataset

dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:1%]", streaming=False)
# columns: [id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName, image]

# overwrite existing data.csv
open("./data/data.csv", 'w')
dataset.to_csv("./data/data.csv")
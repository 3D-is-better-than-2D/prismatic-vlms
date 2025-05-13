from datasets import load_dataset
from PIL import Image

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("remyxai/OpenSpaces_MC")

# Print the dataset
print(ds)
# Print the dataset splits
print(ds.keys())
# Print a specific question/answer pair for image 0
print(ds['train']['messages'][0][2])
# Show there is image and messages
print(ds['train'])
# Print number of Q/A pairs for image 0
print(len(ds['train']['messages'][0]))
# Display first image
Image.open(ds['train']['images'][0]).show()
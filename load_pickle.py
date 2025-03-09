import pickle

# Replace 'your_file.pkl' with your actual file name
with open("your_file.pkl", "rb") as file:
    data = pickle.load(file)

print("Loaded data:", data)

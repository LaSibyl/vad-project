file_path = "data/sample.wav"

with open(file_path, "rb") as f:
    header = f.read(16)

print(header)
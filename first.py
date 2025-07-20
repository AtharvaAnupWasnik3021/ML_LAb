from google.colab import files
import os

uploaded = files.upload()

for filename in uploaded.keys():
  print(f'User uploaded file "{filename}" with length {len(uploaded[filename])} bytes')
  # Move the uploaded file to the current directory
  os.rename(filename, "Lab Session Data.xlsx")

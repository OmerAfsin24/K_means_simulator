import os
import kagglehub
import shutil

if __name__ == "__main__":
  # Working directory
  working_dir = os.getcwd()

  # Data directory
  data_dir = os.path.join(working_dir, "data")
  os.makedirs(data_dir, exist_ok=True)

  # Download dataset from Kaggle
  path = kagglehub.dataset_download("simtoor/mall-customers")
  shutil.copytree(path, data_dir, dirs_exist_ok=True)

  print("Dataset downloaded successfully!")
  print("Dataset location:", data_dir)



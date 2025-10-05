import gdown
import glob
import os
import torch

# Google Drive folder link
folder_url = "https://drive.google.com/drive/folders/1U7dPo0R20KonIRP46NsBRFREbAJ__O6A"

# Download all files (by default, goes into a subfolder named after the Drive folder)
print("⬇️ Downloading files from Drive folder...")
output_folder = "Models"
gdown.download_folder(folder_url, output=output_folder, quiet=False, use_cookies=False)

# Look for .pth files inside that folder (recursively)
pth_files = glob.glob(os.path.join(output_folder, "**", "*.pth"), recursive=True)

if not pth_files:
    print("❌ No .pth files found.")
    exit()

# Show files & pick
print("\n📂 Available model files:")
for idx, f in enumerate(pth_files, 1):
    print(f"{idx}. {f}")

choice = int(input("\nEnter the number of the model you want to load: "))
selected_file = pth_files[choice - 1]

# Load into PyTorch
print(f"\n🔍 Loading model: {selected_file}")
model_state = torch.load(selected_file, map_location='cpu')
print("✅ Model loaded into memory:", type(model_state))

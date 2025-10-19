import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import torch
import torchvision
from PIL import Image
from torchvision import transforms
print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("matplotlib version:", matplotlib.__version__)
print("pandas version:", pd.__version__)
print("PIL version:", PIL.__version__)
print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
Platform: linux
Python version: 3.11.0 (main, Nov 15 2022, 20:12:54) [GCC 10.2.1 20210110]
---
matplotlib version: 3.9.2
pandas version: 2.2.3
PIL version: 10.2.0
torch version: 2.2.2+cu121
torchvision version: 0.17.2+cu121
my_values = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
my_tensor = torch.Tensor(my_values)

print("my_tensor class:", type(my_tensor))
print(my_tensor)
my_tensor class: <class 'torch.Tensor'>
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
print("my_tensor shape:",my_tensor.shape) 

print("my_tensor dtype:", my_tensor.dtype)  
my_tensor shape: torch.Size([4, 3])
my_tensor dtype: torch.float32

# Check if GPUs available via `cuda`
cuda_gpus_available = torch.cuda.is_available()

# Check if GPUs available via `mps`
mps_gpus_available = torch.backends.mps.is_available()

print("cuda GPUs available:", cuda_gpus_available)
print("mps GPUs available:", mps_gpus_available)
cuda GPUs available: True
mps GPUs available: False
print(torch.cuda.device_count())
1
print(torch.cuda.get_device_name(0))
Tesla T4
my_tensor = my_tensor.to("cuda")

print("my_tensor device:", my_tensor.device)
my_tensor device: cuda:0
left_tensor = my_tensor[:2, :]
right_tensor = my_tensor[2:, :]

print("left_tensor class:", type(left_tensor))
print("left_tensor shape:",  left_tensor.shape)
print("left_tensor data type:", left_tensor.dtype)
print("left_tensor device:", left_tensor.device)
print(left_tensor)
print()
print("right_tensor class:", type(right_tensor))
print("right_tensor shape:", right_tensor.shape)
print("right_tensor data type:", right_tensor.dtype)
print("right_tensor device:", right_tensor.device)
print(right_tensor)
left_tensor class: <class 'torch.Tensor'>
left_tensor shape: torch.Size([2, 3])
left_tensor data type: torch.float32
left_tensor device: cuda:0
tensor([[1., 2., 3.],
        [4., 5., 6.]], device='cuda:0')

right_tensor class: <class 'torch.Tensor'>
right_tensor shape: torch.Size([2, 3])
right_tensor data type: torch.float32
right_tensor device: cuda:0
tensor([[ 7.,  8.,  9.],
        [10., 11., 12.]], device='cuda:0')
summed_tensor_operator = left_tensor + right_tensor
summed_tensor_method = left_tensor.add(right_tensor) 

print("summed_tensor_operator class:", type(summed_tensor_operator))
print("summed_tensor_operator shape:", summed_tensor_operator.shape)
print("summed_tensor_operator data type:", summed_tensor_operator.dtype)
print("summed_tensor_operator device:", summed_tensor_operator.device)
print(summed_tensor_operator)
print()
print("summed_tensor_method class:", type(summed_tensor_method))
print("summed_tensor_method shape:", summed_tensor_method.shape)
print("summed_tensor_method data type:", summed_tensor_method.dtype)
print("summed_tensor_method device:", summed_tensor_method.device)
print(summed_tensor_method)
summed_tensor_operator class: <class 'torch.Tensor'>
summed_tensor_operator shape: torch.Size([2, 3])
summed_tensor_operator data type: torch.float32
summed_tensor_operator device: cuda:0
tensor([[ 8., 10., 12.],
        [14., 16., 18.]], device='cuda:0')

summed_tensor_method class: <class 'torch.Tensor'>
summed_tensor_method shape: torch.Size([2, 3])
summed_tensor_method data type: torch.float32
summed_tensor_method device: cuda:0
tensor([[ 8., 10., 12.],
        [14., 16., 18.]], device='cuda:0')
ew_tensor_operator = left_tensor * right_tensor
ew_tensor_method = left_tensor.mul(right_tensor)

print("ew_tensor_operator class:", type(ew_tensor_operator))
print("ew_tensor_operator shape:", ew_tensor_operator.shape)
print("ew_tensor_operator data type:", ew_tensor_operator.dtype)
print("ew_tensor_operator device:", ew_tensor_operator.device)
print(ew_tensor_operator)
print()
print("ew_tensor_method class:", type(ew_tensor_method))
print("ew_tensor_method shape:", ew_tensor_method.shape)
print("ew_tensor_method data type:", ew_tensor_method.dtype)
print("ew_tensor_method device:", ew_tensor_method.device)
print(ew_tensor_method)
ew_tensor_operator class: <class 'torch.Tensor'>
ew_tensor_operator shape: torch.Size([2, 3])
ew_tensor_operator data type: torch.float32
ew_tensor_operator device: cuda:0
tensor([[ 7., 16., 27.],
        [40., 55., 72.]], device='cuda:0')

ew_tensor_method class: <class 'torch.Tensor'>
ew_tensor_method shape: torch.Size([2, 3])
ew_tensor_method data type: torch.float32
ew_tensor_method device: cuda:0
tensor([[ 7., 16., 27.],
        [40., 55., 72.]], device='cuda:0')
left_tensor * right_tensor == right_tensor * left_tensor
tensor([[True, True, True],
        [True, True, True]], device='cuda:0')
new_left_tensor = torch.Tensor([[2, 5], [7, 3]])
new_right_tensor = torch.Tensor([[8], [9]])

print("new_left_tensor class:", type(new_left_tensor))
print("new_left_tensor shape:", new_left_tensor.shape)
print("new_left_tensor data type:", new_left_tensor.dtype)
print("new_left_tensor device:", new_left_tensor.device)
print(new_left_tensor)
print()
print("new_right_tensor class:", type(new_right_tensor))
print("new_right_tensor shape:", new_right_tensor.shape)
print("new_right_tensor data type:", new_right_tensor.dtype)
print("new_right_tensor device:", new_right_tensor.device)
print(new_right_tensor)
new_left_tensor class: <class 'torch.Tensor'>
new_left_tensor shape: torch.Size([2, 2])
new_left_tensor data type: torch.float32
new_left_tensor device: cpu
tensor([[2., 5.],
        [7., 3.]])

new_right_tensor class: <class 'torch.Tensor'>
new_right_tensor shape: torch.Size([2, 1])
new_right_tensor data type: torch.float32
new_right_tensor device: cpu
tensor([[8.],
        [9.]])
mm_tensor_operator = new_left_tensor @ new_right_tensor
mm_tensor_method = new_left_tensor.matmul(new_right_tensor)

print("mm_tensor_operator class:", type(mm_tensor_operator))
print("mm_tensor_operator shape:", mm_tensor_operator.shape)
print("mm_tensor_operator data type:", mm_tensor_operator.dtype)
print("mm_tensor_operator device:", mm_tensor_operator.device)
print(mm_tensor_operator)
print()
print("mm_tensor_method class:", type(mm_tensor_method))
print("mm_tensor_method shape:", mm_tensor_method.shape)
print("mm_tensor_method data type:", mm_tensor_method.dtype)
print("mm_tensor_method device:", mm_tensor_method.device)
print(mm_tensor_method)
mm_tensor_operator class: <class 'torch.Tensor'>
mm_tensor_operator shape: torch.Size([2, 1])
mm_tensor_operator data type: torch.float32
mm_tensor_operator device: cpu
tensor([[61.],
        [83.]])

mm_tensor_method class: <class 'torch.Tensor'>
mm_tensor_method shape: torch.Size([2, 1])
mm_tensor_method data type: torch.float32
mm_tensor_method device: cpu
tensor([[61.],
        [83.]])
my_tensor_mean = my_tensor.mean()

print("my_tensor_mean class:", type(my_tensor_mean))
print("my_tensor_mean shape:", my_tensor_mean.shape)
print("my_tensor_mean data type:", my_tensor_mean.dtype)
print("my_tensor_mean device:", my_tensor_mean.device)
print("my_tensor mean:", my_tensor_mean)
my_tensor_mean class: <class 'torch.Tensor'>
my_tensor_mean shape: torch.Size([])
my_tensor_mean data type: torch.float32
my_tensor_mean device: cuda:0
my_tensor mean: tensor(6.5000, device='cuda:0')
my_tensor_column_means = ...

print("my_tensor_column_means class:", type(my_tensor_column_means))
print("my_tensor_column_means shape:", my_tensor_column_means.shape)
print("my_tensor_column_means data type:", my_tensor_column_means.dtype)
print("my_tensor_column_means device:", my_tensor_column_means.device)
print("my_tensor column means:", my_tensor_column_means)
data_dir = os.path.join("data_p1", "data_multiclass")
train_dir = ...

print("data_dir class:", type(data_dir))
print("Data directory:", data_dir)
print()
print("train_dir class:", type(train_dir))
print("Training data directory:", train_dir)
class_directories = ...

print("class_directories type:", type(class_directories))
print("class_directories length:", len(class_directories))
print(class_directories)
class_distributions_dict = {}

for subdirectory in class_directories:
   ...

class_distributions = pd.Series(class_distributions_dict)

print("class_distributions type:", type(class_distributions))
print("class_distributions shape:", class_distributions.shape)
print(class_distributions)
# Create a bar plot of class distributions
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the data
ax....  # Write your code here
ax.set_xlabel("Class Label")
ax.set_ylabel("Frequency [count]")
ax.set_title("Class Distribution, Multiclass Training Set")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Define path for hog image
hog_image_path = os.path.join(train_dir, "hog", "ZJ000072.jpg")

# Define path for antelope image
antelope_image_path = os.path.join(train_dir, "antelope_duiker", "ZJ002533.jpg")

print("hog_image_path type:", type(hog_image_path))
print(hog_image_path)
print()
print("antelope_image_path type:", type(antelope_image_path))
print(antelope_image_path)
hog_image_pil = Image.open(hog_image_path)

print("hog_image_pil type:", type(hog_image_pil))
hog_image_pil
antelope_image_pil = ...

print("antelope_image_pil type:", type(antelope_image_pil))
antelope_image_pil
# Get image size
hog_image_pil_size = hog_image_pil.size

# Get image mode
hog_image_pil_mode = hog_image_pil.mode

# Print results
print("hog_image_pil_size class:", type(hog_image_pil_size))
print("hog_image_pil_size length:", len(hog_image_pil_size))
print("Hog image size:", hog_image_pil_size)
print()

print("hog_image_pil_mode class:", type(hog_image_pil_mode))
print("Hog image mode:", hog_image_pil_mode)
# Get image size
antelope_image_pil_size = ...

# Get image mode
antelope_image_pil_mode = ...

# Get image mode
print("antelope_image_pil_size class:", type(antelope_image_pil_size))
print("antelope_image_pil_size length:", len(antelope_image_pil_size))
print("Antelope image size:", antelope_image_pil_size)
print()
print("antelope_image_pil_mode class:", type(antelope_image_pil_mode))
print("Antelope image mode:", antelope_image_pil_mode)
hog_tensor = transforms.ToTensor()(hog_image_pil)

print("hog_tensor type:", type(hog_tensor))
print("hog_tensor shape:", hog_tensor.shape)
print("hog_tensor dtype:", hog_tensor.dtype)
print("hog_tensor device:", hog_tensor.device)
antelope_tensor = ...

print("antelope_tensor type:", type(antelope_tensor))
print("antelope_tensor shape:", antelope_tensor.shape)
print("antelope_tensor dtype:", antelope_tensor.dtype)
print("antelope_tensor device:", antelope_tensor.device)
# Create figure with single axis
fig, ax = plt.subplots(1, 1)

# Plot gray channel of hog_tensor
ax.imshow(hog_tensor[0, :, :])

# Turn off x- and y-axis
ax.axis("off")

# Set title
ax.set_title("Hog, grayscale");
# Create figure with 3 subplots
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

# Plot red channel
red_channel = antelope_tensor[0, :, :]
ax0.imshow(red_channel, cmap="Reds")
ax0.set_title("Antelope, Red Channel")
ax0.axis("off")

# Plot green channel
green_channel = ...

# Plot blue channel
blue_channel = ...
plt.tight_layout();
max_channel_values = ...
min_channel_values = ...

print("max_channel_values class:", type(max_channel_values))
print("max_channel_values shape:", max_channel_values.shape)
print("max_channel_values data type:", max_channel_values.dtype)
print("max_channel_values device:", max_channel_values.device)
print("Max values in antelope_tensor:", max_channel_values)
print()
print("min_channel_values class:", type(min_channel_values))
print("min_channel_values shape:", min_channel_values.shape)
print("min_channel_values data type:", min_channel_values.dtype)
print("min_channel_values device:", min_channel_values.device)
print("Min values in antelope_tensor:", min_channel_values)
mean_channel_values = ...

print("mean_channel_values class:", type(mean_channel_values))
print("mean_channel_values shape:", mean_channel_values.shape)
print("mean_channel_values dtype:", mean_channel_values.dtype)
print("mean_channel_values device:", mean_channel_values.device)
print("Mean channel values in antelope_tensor (RGB):", mean_channel_values)

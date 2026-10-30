import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm import tqdm



print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)
print("torchinfo version : ", torchinfo.__version__)
print("numpy version : ", np.__version__)
print("matplotlib version : ", matplotlib.__version__)
print("PIL version : ", PIL.__version__)

!python --version



if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")



data_dir = os.path.join("data_p2", "data_undersampled")
train_dir = os.path.join(data_dir, "train")

print("Data Directory:", data_dir)
print("Training Data Directory:", train_dir)



classes = os.listdir(train_dir)

print("List of classes:", classes)



class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img



# Define transformation to apply to the images
transform_normalized = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        # Convert images to tensors
        transforms.ToTensor(),
        # Normalize the tensors (copy the mean and std from previous lesson!)
        transforms.Normalize(
           mean=[0.4326, 0.4953, 0.3120], std=[0.2178, 0.2214, 0.2091])
         
            ]
)

print(type(transform_normalized))
print("-----------------")
print(transform_normalized)




dataset = datasets.ImageFolder(root=train_dir, transform=transform_normalized)

print('Length of dataset:', len(dataset))





# Important, don't change this!
g = torch.Generator()
g.manual_seed(42)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=g)

print("Length of training dataset:", len(train_dataset))
print("Length of validation dataset:", len(val_dataset))




length_dataset = len(dataset)
length_train = len(train_dataset)
length_val = len(val_dataset)

percent_train = np.round(100 * length_train / length_dataset, 2)
percent_val = np.round(100 * length_val / length_dataset, 2)

print(f"Train data is {percent_train}% of full data")
print(f"Validation data is {percent_val}% of full data")




from training import class_counts

dataset_counts = class_counts(dataset)

# Make a bar chart from the function output
dataset_counts.sort_values().plot(kind="bar")

# Add axis labels and title
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Entire Dataset");




train_counts = class_counts(train_dataset)

# Make a bar chart from the function output
train_counts.sort_values().plot(kind="bar")


# Add axis labels and title
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Training Dataset");






val_counts = class_counts(val_dataset)

# Make a bar chart from the function output
val_counts.sort_values().plot(kind="bar")

# Add axis labels and title
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Validation Dataset");




batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(type(train_loader))



val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(type(val_loader))




data_iter = iter(train_loader)
images, labels = next(data_iter)

image_shape = images.shape
print("Shape of batch of images", image_shape)

label_shape = labels.shape
print("Shape of batch of labels:", label_shape)




labels




model = torch.nn.Sequential()




# Convolutional layer 1 (sees 3x224x224 image tensor)
conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
model.append(conv1)

print(model)




max_pool1 = nn.MaxPool2d(2, 2)
model.append(torch.nn.ReLU())
model.append(max_pool1)



# Convolutional layer 2 (sees 16x112x112 tensor)
conv2 = nn.Conv2d(16, 32, 3, padding=1)
max_pool2 = nn.MaxPool2d(2, 2)
model.append(conv2)
model.append(torch.nn.ReLU())
model.append(max_pool2)

print(model)



# Convolutional layer 3 (sees 32x56x56 tensor)
conv3 = nn.Conv2d(32, 64, 3, padding=1)
max_pool3 = nn.MaxPool2d(2, 2)
model.append(conv3)
model.append(torch.nn.ReLU())
model.append(max_pool3)

print(model)




model.append(torch.nn.Flatten())
model.append(nn.Dropout(0.5))



# Linear layer (64 * 28 * 28 -> 500)
linear1 = torch.nn.Linear(64 * 28 * 28, 500)
model.append(linear1)
model.append(torch.nn.ReLU())
model.append(torch.nn.Dropout())

print(model)




# Linear layer (500 -> 5)
output_layer = nn.Linear(500, 5)
model.append(output_layer)

print(model)




loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(loss_fn)
print("----------------------")
print(optimizer)



model.to(device)



height = 224
width = 224
summary(model, input_size=(batch_size, 3, height, width))



from training import train


train_loss, valid_loss, train_accuracies, valid_accuracies = train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=15,
    device=device,
)



model = torch.load("model/model_trained.pth", weights_only=False)
model.to("cuda")


df = pd.read_csv('post_train_evaluation_metrics.csv')


train_losses = df['Train Loss']
valid_losses = df['Validation Loss']

train_accuracies = df['Train Accuracy']
valid_accuracies = df['Validation Accuracy']



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(valid_accuracies, label="Validation Accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()




from training import predict

probabilities_val = predict(model, val_loader, device)
predictions_val = torch.argmax(probabilities_val, dim=1)

print(predictions_val)



targets_val = torch.cat(
    [labels for _, labels in tqdm(val_loader, desc="Get Labels")]
).to(device)



cm = confusion_matrix(targets_val.cpu(), predictions_val.cpu())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Set figure size
plt.figure(figsize=(10, 8))

disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.show()


####Oh no, we’re overfitting! 🤯

#Overfitting occurs when a model learns the training data too well, capturing noise and details to the extent that it negatively impacts the model’s performance on new data. The symptoms of overfitting in our training results are:

#High training accuracy: The model performs exceptionally well on the training data.
#Increasing validation loss: Despite improvements in training loss and accuracy, the validation loss starts to increase after reaching a certain point.
#Stagnant or decreasing validation accuracy: The model’s ability to generalize to unseen data does not improve or worsens as training progresses.
#Addressing Overfitting

#To mitigate overfitting, consider the following strategies:

#Data Augmentation: Augment the training data by applying random transformations (e.g., rotations, scaling, flips) to generate new training samples. This can help the model generalize better.

#Dropout: Introduce dropout layers into our network. Dropout randomly sets a fraction of input units to 0 during training, which helps prevent the model from becoming too reliant on any single feature.

#Regularization: Apply regularization techniques, such as L1 or L2 regularization, which add a penalty on the magnitude of network parameters. This can discourage complex models that overfit.

#Early Stopping: Monitor the model’s performance on a validation set and stop training when the validation loss starts to increase, which is a sign that the model's beginning to overfit.

#Reduce Model Complexity: Simplify your model by reducing the number of layers or the number of units in the layers. A simpler model may generalize better.

#Use More Data: If possible, adding more data can help the model learn better and generalize well to new, unseen data.

#Batch Normalization: Although primarily used to help with training stability and convergence, batch normalization can sometimes also help with overfitting by regularizing the model somewhat.

#Cross-validation: Helps prevent overfitting by testing the model on different parts of the data, ensuring it performs well on new data.



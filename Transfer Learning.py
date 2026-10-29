import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm import tqdm



print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)
print("torchinfo version : ", torchinfo.__version__)
print("numpy version : ", np.__version__)
print("matplotlib version : ", matplotlib.__version__)

!python --version



if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



print(f"Using {device} device.")


data_dir = data_dir = os.path.join("data_p2", "data_undersampled", "train"  )

print("Data directory:", data_dir)




class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img




mean = [0.4242, 0.4896, 0.3053]
std = [0.2210, 0.2224, 0.2088]



transform_normalized = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    
)
transform_normalized




dataset = datasets.ImageFolder (data_dir, transform_normalized)

dataset





classes = dataset.classes
classes





from training import class_counts




counts = class_counts(dataset)
counts



batch_size = 32
dataset_loader = DataLoader(dataset, batch_size=batch_size)

print(f"Batch shape: {next(iter(dataset_loader))[0].shape}")



model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)



test_batch = next(iter(dataset_loader))[0]
batch_shape = test_batch.shape

# Create the model summary
summary(model, input_size=batch_shape)




for params in model.parameters():
    params.requires_grad = False



# Move the model to device
model.to(device)

# Move our test_batch to device
test_batch_cuda = test_batch.to(device)
print("Test batch is running on:", test_batch_cuda.device)




model_test_out = model(test_batch_cuda)
model_test_shape = model_test_out.shape

print("Output shape:", model_test_shape)





list(model.named_modules())[-1]




in_features = model.fc.in_features
in_features




classifier = torch.nn.Sequential()





classification_layer = torch.nn.Linear(in_features=in_features, out_features=256)


# Add the layer to our classifier
classifier.append(classification_layer)




classifier.append(torch.nn.ReLU())
classifier.append(torch.nn.Dropout())




output_layer = torch.nn.Linear(in_features=256, out_features=5)

# Add the layer to our classifier
classifier.append(output_layer)




model.fc = classifier


# Create the model summary
summary(model, input_size=batch_shape)




k = 5

kfold_splitter = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

train_nums, val_nums = next(kfold_splitter.split(range(100)))
fold_fraction = len(val_nums) / (len(train_nums) + len(val_nums))
print(f"One fold is {100*fold_fraction:.2f}%")




from training import predict, train



train?



loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())



model.fc


def reset_classifier(model):
    model.fc.get_submodule("0").reset_parameters()
    model.fc.get_submodule("3").reset_parameters()



num_epochs = 6



# you can safely skip this cell and load the model in the next cell

training_records = {}
fold_count = 0

for train_idx, val_idx in kfold_splitter.split(np.arange(len(dataset))):
    fold_count += 1
    print("*****Fold {}*****".format(fold_count))

    # Make train and validation data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Reset the model
    reset_classifier(model)

    # Train
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs=num_epochs,
        device=device,
        use_train_accuracy=False,
    )

    # Save training results for graphing
    training_records[fold_count] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    print("\n\n")





model = torch.load("model/load/pretrained_model.pth", weights_only=False)
model.to(device)



import pickle
with open('training_records.pkl', 'rb') as fp:
    training_records = pickle.load(fp)



print(type(training_records))
training_records.keys()




print(type(training_records[1]))
training_records[1].keys()



training_records[1]["train_losses"]



def plot_all_folds(data, measurement):
    for fold in data.keys():
        plt.plot(data[fold][measurement], label=f"Fold {fold}, {measurement}")
    plt.xlabel("Epochs")
    plt.legend()




def fold_average(data, measurement):
    return np.mean([data[fold][measurement][-1] for fold in data])


for measurement in training_records[1].keys():
    avg_measure = fold_average(training_records, measurement)
    print(f"Averaged {measurement}: {avg_measure}")




# If you loaded the pre-trained model, make sure you have the correct validation set
_, val_idx = list(kfold_splitter.split(np.arange(len(dataset))))[-1]
val_dataset = torch.utils.data.Subset(dataset, val_idx)
val_loader = DataLoader(val_dataset, batch_size=batch_size)





probabilities = predict(model, val_loader, device)
predictions = torch.argmax(probabilities, dim=1)




predictions





targets = []

for _, labels in tqdm(val_loader):
    targets.extend(labels.tolist())



targets[:5]



cm = confusion_matrix(targets, predictions.cpu())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.show();





torch.save(model, "model/resnet_k_folds")

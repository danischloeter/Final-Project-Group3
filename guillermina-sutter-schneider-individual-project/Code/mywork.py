import os
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain
from sklearn import preprocessing
import torch.nn as nn
from PIL import Image
from sklearn.metrics import fbeta_score

# %%-------------------- EDA -----------------------------------------

import matplotlib.pyplot as plt

classes = ['primary', 'clear', 'agriculture', 'road', 'water',
           'partly_cloudy', 'cultivation', 'habitation', 'haze', 'cloudy', 'bare_ground',
           'selective_logging', 'artisanal_mine', 'blooming', 'slash_burn', 'blow_down', 'conventional_mine']

count = [37513, 28431, 12315, 8071, 7411, 7261, 4547, 3660, 2697, 2089, 862, 340, 339, 332, 209, 101, 100]

plt.barh(classes, count)
plt.title("Labels in dataset")
plt.show()


# %%-----------------------------------------------------------------

# Dictionary image labels
label=pd.read_csv('/home/ubuntu/ML2/Final Project/Data/train_v2.csv')
print(label.head())

imagestag=label['image_name'].values
imageslabel=label['tags'].values
dictionary = dict(zip(imagestag, imageslabel))

#Counting images
labels_list = list(chain.from_iterable([tags.split(" ") for tags in imageslabel]))
labels = sorted(set(labels_list))
countlabels = pd.Series(labels_list).value_counts()
print(countlabels)


Allimages=[]
Counter={}
for i in label.iloc[:,1]:
    if str(i) in Counter:
        Counter[str(i)]=Counter[str(i)]+1
    else:
        Counter[str(i)]=1
Counts=sorted(Counter.items(), key=lambda kv: kv[1])
for i in Counts:
    print(i)
print(Counts)


X_train, X_test, y_train, y_test = train_test_split(imagestag, imageslabel, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.05)
train=pd.DataFrame()
train['X']=X_train
train['y']=y_train

val=pd.DataFrame()
val['X']=X_val
val['y']=y_val

test=pd.DataFrame()
test['X']=X_test
test['y']=y_test

train.to_csv("/home/ubuntu/ML2/Final Project/Data/train.csv", header=False)
val.to_csv("/home/ubuntu/ML2/Final Project/Data/validation.csv", header=False)
test.to_csv('/home/ubuntu/ML2/Final Project/Data/test.csv',header=False)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.1
N_EPOCHS = 2
BATCH_SIZE = 90
DROPOUT = 0.10


transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1,saturation=10,contrast=30,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


lb = preprocessing.LabelBinarizer()
Alltargetsencode = lb.fit(labels)
print(Alltargetsencode)
print(lb.classes_)


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        self.transforms = transformations
        self.data_info = pd.read_csv(csv_path, header=None)
        self.label_arr = self.data_info.iloc[:,2]
        self.data_len = len(self.data_info.index)
    def __getitem__(self, index):

        # Get images name from the pandas df
        single_image_name = self.data_info.iloc[index, 1]
        single_image_name = single_image_name+'.jpg'

        # Read images
        img_as_img = Image.open(os.path.join('/home/ubuntu/ML2/Final Project/Data/train-jpg/',single_image_name)).convert("RGB")

        # Transform image to tensor
        img_as_tensor = self.transforms(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = list(self.label_arr[index].split(" "))
        target=lb.transform(single_image_label)
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for i in target:
            x += i
        target=torch.from_numpy(x)
        return img_as_tensor, target.float()
    def __len__(self):
        return self.data_len


trainset= CustomDatasetFromImages("/home/ubuntu/ML2/Final Project/Data/train.csv")


train_loader= DataLoader(trainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)


valset= CustomDatasetFromImages("/home/ubuntu/ML2/Final Project/Data/validation.csv")


val_loader= DataLoader(valset,
                          batch_size=BATCH_SIZE, #len of validation set
                          shuffle=True,
                          num_workers=4 )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_gpu = torch.cuda.is_available()
torch.manual_seed(2)
np.random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


resnet50 = models.resnet50(pretrained=True)


# Freeze layers of Resnet50
ct = 0
for child in resnet50.children():
    ct += 1
    if ct < 3:
        for param in child.parameters():
            param.requires_grad = False


# Change the last layer
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 17)

model = resnet50.to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


print("Starting training loop...")
for epoch in range(N_EPOCHS):
    model.train()
    loss_train = 0
    for iter, traindata in enumerate(train_loader):
        train_inputs, train_labels = traindata
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        torch.save(model.state_dict(), "project.pt")
        print('Batch {} and Loss {:.5f}'.format(iter,loss_train/BATCH_SIZE))
    model.load_state_dict(torch.load("project.pt"))
    model.eval()


    with torch.no_grad():
        for iter, valdata in enumerate(val_loader, 0):
            val_inputs, val_labels = valdata
            val_inputs1, val_labels1 = val_inputs.to(device), val_labels.to(device)
            y_test_pred = model(val_inputs1)
            tar_=val_labels.cpu().numpy()
            loss = criterion(y_test_pred, val_labels1)
            print('Validation Loss {:.5f}'.format(loss.item()))

    print("Epoch {} | Train Loss {:.5f}".format( epoch, loss_train/BATCH_SIZE))



# %% ----------------------------------- Predict --------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(X_test):

    use_gpu = torch.cuda.is_available()
    torch.manual_seed(2)
    np.random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.load_state_dict(torch.load('project.pt'))
    y = model(X_test)
    y_pred = torch.sigmoid(y)

    return y_test, y_pred


for iter, testdata in enumerate(train_loader):
    X_test,y_test=testdata
    y_test,y_pred=predict(X_test)

    py_pred = y_pred.detach().numpy()
    py_true = y_test.detach().numpy()
    py_pred[py_pred >= 0.5] = 1
    py_pred[py_pred < 0.5] = 0

    fbeta_sklearn = fbeta_score(py_true, py_pred, 2, average='samples')

    print('Scores are {:.3f} (sklearn) '.format(fbeta_sklearn))
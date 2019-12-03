import os
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import join as pjoin
from itertools import chain
from sklearn import preprocessing
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


# Dictionary image labels
label=pd.read_csv('train_v2.csv')
print(label.head())

imagestag=label['image_name'].values
imageslabel=label['tags'].values
dictionary = dict(zip(imagestag, imageslabel))

#Counting images
labels_list = list(chain.from_iterable([tags.split(" ") for tags in imageslabel]))
labels = sorted(set(labels_list))
countlabels = pd.Series(labels_list).value_counts() # To sort them by count
print(countlabels)

classes = ['primary', 'clear', 'agriculture', 'road', 'water',
           'partly_cloudy', 'cultivation', 'habitation', 'haze', 'cloudy', 'bare_ground',
           'selective_logging', 'artisanal_mine', 'blooming', 'slash_burn', 'blow_down', 'conventional_mine']

count = [37513, 28431, 12315, 8071, 7411, 7261, 4547, 3660, 2697, 2089, 862, 340, 339, 332, 209, 101, 100]

plt.barh(classes, count)
plt.title("Labels in dataset")
plt.show()

#Counting combinations of images
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
#print(Counts)

#test set 20%
X_train, X_test, y_train, y_test = train_test_split(imagestag, imageslabel, random_state=42, test_size=0.2)
#Val set 5%
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

train.to_csv("train.csv", header=False)
val.to_csv("validation.csv", header=False)
test.to_csv('test.csv',header=False)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-1
N_EPOCHS = 5
BATCH_SIZE = 70
DROPOUT = 0.1

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1,saturation=10,contrast=30,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        # Get image name from the pandas df
        single_image_name = self.data_info.iloc[index, 1]
        single_image_name=single_image_name+'.jpg'
        # Open image
        img_as_img = Image.open(os.path.join('/home/ubuntu/Proyecto/train-jpg/',single_image_name)).convert("RGB")
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


trainset= CustomDatasetFromImages("/home/ubuntu/Proyecto/train.csv")


train_loader= DataLoader(trainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

valset= CustomDatasetFromImages("/home/ubuntu/Proyecto/validation.csv")


val_loader= DataLoader(valset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4 )


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)
np.random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% -------------------------------------- Training Prep ----------------------------------------------------------

vgg =models.vgg19_bn(pretrained='imagenet')
# Number of filters in the bottleneck layer
num_ftrs = vgg.classifier[6].in_features
# convert all the layers to list and remove the last one
features = list(vgg.classifier.children())[:-1]
## Add the last layer based on the num of classes in  dataset
features.extend([nn.Linear(num_ftrs, 17)])
## convert it into container and add it to our model class.
vgg.classifier = nn.Sequential(*features)

model =vgg.to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
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
        torch.save(model.state_dict(), "project1.pt")
        print('Batch {} and Loss {:.5f}'.format(iter,loss_train/BATCH_SIZE))
    model.load_state_dict(torch.load("project1.pt"))
    model.eval()

    with torch.no_grad():
        for iter, valdata in enumerate(val_loader, 0):
            val_inputs, val_labels = valdata
            val_inputs1, val_labels1 = val_inputs.to(device), val_labels.to(device)
            y_test_pred = model(val_inputs1)
            tar_=val_labels.cpu().numpy()
            loss = criterion(y_test_pred, val_labels1)
            loss_val=loss.item()
            print('Validation Loss {:.5f}'.format(loss_val))

    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f} ".format(
        epoch, loss_train/BATCH_SIZE,  loss_val))



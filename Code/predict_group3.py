import os
import numpy as np
import torch
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from sklearn import preprocessing
import torch.nn as nn
from PIL import Image
from sklearn.metrics import fbeta_score


transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, saturation=10, contrast=30, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

lb = preprocessing.LabelBinarizer()
Alltargetsencode = lb.fit(['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water'])
print(Alltargetsencode)
print(lb.classes_)


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        self.transforms = transformations
        self.data_info = pd.read_csv(csv_path, header=None)
        self.label_arr = self.data_info.iloc[:, 2]
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.data_info.iloc[index, 1]
        single_image_name = single_image_name + '.jpg'
        # Open image
        img_as_img = Image.open(os.path.join('/home/ubuntu/Proyecto/train-jpg/', single_image_name)).convert("RGB")
        # Transform image to tensor
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = list(self.label_arr[index].split(" "))
        target = lb.transform(single_image_label)
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for i in target:
            x += i
        target = torch.from_numpy(x)
        return img_as_tensor, target.float()

    def __len__(self):
        return self.data_len


testset = CustomDatasetFromImages("/home/ubuntu/Proyecto/test.csv")

train_loader = DataLoader(testset,
                          batch_size=92,
                          shuffle=True,
                          num_workers=0)


def predict(X_test):

    vgg = models.vgg19_bn(pretrained='imagenet')
    # Number of filters in the bottleneck layer
    num_ftrs = vgg.classifier[6].in_features
    # convert all the layers to list and remove the last one
    features = list(vgg.classifier.children())[:-1]
    ## Add the last layer based on the num of classes in our dataset
    features.extend([nn.Linear(num_ftrs, 17)])
    ## convert it into container and add it to our model class.
    vgg.classifier = nn.Sequential(*features)

    model = vgg

    model.load_state_dict(torch.load('project1.pt'))
    y=model(X_test)
    y_pred=torch.sigmoid(y)

    return y_test,y_pred


for iter, testdata in enumerate(train_loader):
    X_test,y_test=testdata
    y_test,y_pred=predict(X_test)
    py_pred = y_pred.detach().numpy()
    py_true = y_test.detach().numpy()
    py_true=py_true.astype(int)
    py_pred[py_pred >= 0.5] = 1
    py_pred[py_pred < 0.5] = 0
    py_pred=py_pred.astype(int)
    #print(py_pred,py_pred.shape,type(py_pred))
   # print(py_true, py_true.shape, type(py_true))
    fbeta_sklearn = fbeta_score(py_true, py_pred, 2, average='samples')

    print('Scores are {:.3f} (sklearn) '.format(fbeta_sklearn))


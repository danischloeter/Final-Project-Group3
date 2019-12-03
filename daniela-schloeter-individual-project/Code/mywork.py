
# Dictionary image labels
label=pd.read_csv('train_v2.csv')
print(label.head())

imagestag=label['image_name'].values
imageslabel=label['tags'].values
dictionary = dict(zip(imagestag, imageslabel))


# Counting combinations
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


# Csv for training testing and validating
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

train.to_csv("train.csv", header=False)
val.to_csv("validation.csv", header=False)
test.to_csv('test.csv',header=False)

transformations = transforms.Compose([
    #transforms.Resize((800,600),interpolation=Image.BICUBIC),
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1,saturation=10,contrast=30,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

lb = preprocessing.LabelBinarizer()  #Importing binarizer
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
        single_image_name = self.data_info.iloc[index, 1]  # Get image name from the pandas df
        single_image_name=single_image_name+'.jpg'
        img_as_img = Image.open(os.path.join('/home/ubuntu/Proyecto/train-jpg/',single_image_name)).convert("RGB") # Load image in RGB
        img_as_tensor = self.transforms(img_as_img)  # Transform image to tensor
        single_image_label = list(self.label_arr[index].split(" ")) # Get label(class) of the image based on the pandas column
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

# set-up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)
np.random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CNN class with commented variations

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 7, (3, 3))
        # self.convnorm1 = nn.BatchNorm2d(7)
        # self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(7, 14, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(14)
        # self.pool2 = nn.MaxPool2d((2, 2))
        #
        self.conv3 = nn.Conv2d(14, 28, (3, 3))
        # self.convnorm3 = nn.BatchNorm2d(28)
        # self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(28, 56, (3, 3))
        self.convnorm4 = nn.BatchNorm2d(56)
        # self.pool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(56, 112, (3, 3))
        self.convnorm5 = nn.BatchNorm2d(112)
        # self.pool5 = nn.MaxPool2d((2, 2))

        self.conv6 = nn.Conv2d(112, 224, (2, 2))
        self.convnorm6 = nn.BatchNorm2d(224)
        # self.pool6 = nn.MaxPool2d((2, 2))
        #
        # self.conv7 = nn.Conv2d(224, 448, (2, 2))
        # self.convnorm7 = nn.BatchNorm2d(448)
        # self.pool7 = nn.MaxPool2d((2, 2))

        self.linear1 = nn.Linear(47096, 50)
        self.linear1_bn = nn.BatchNorm1d(50)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(50, 17)
        self.act = torch.relu

    def forward(self, x):
        # x = self.act(self.conv1(x))
        # x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        # x = self.pool1(self.act(self.conv1(x)))
        x = self.act(self.conv1(x))
        # x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.convnorm2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        # x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
        # x = self.pool4(self.convnorm4(self.act(self.conv4(x))))
        x = self.convnorm4(self.act(self.conv4(x)))
        # x = self.pool5(self.convnorm5(self.act(self.conv5(x))))
        # x = self.pool6(self.convnorm6(self.act(self.conv6(x))))
        x = self.convnorm5(self.act(self.conv5(x)))
        x = self.convnorm6(self.act(self.conv6(x)))
        # x = self.pool7(self.convnorm7(self.act(self.conv7(x))))
        x = self.drop((self.linear1_bn(self.act(self.linear1(x.view(len(x), -1))))))
        return self.linear2(x)

# VGG model

vgg =models.vgg19_bn(pretrained='imagenet')
num_ftrs = vgg.classifier[6].in_features # Number of filters in the bottleneck layer
features = list(vgg.classifier.children())[:-1] # convert all the layers to list and remove the last one
features.extend([nn.Linear(num_ftrs, 17)]) ## Add the last layer based on the num of classes in  dataset
vgg.classifier = nn.Sequential(*features) ## convert it into container and add it to our model class.
model =vgg.to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


# Training loops

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


# Predict

def predict(X_test):

    vgg = models.vgg19_bn(pretrained='imagenet')
    num_ftrs = vgg.classifier[6].in_features # Number of filters in the bottleneck layer
    features = list(vgg.classifier.children())[:-1] # convert all the layers to list and remove the last one
    features.extend([nn.Linear(num_ftrs, 17)])  ## Add the last layer based on the num of classes in our dataset
    vgg.classifier = nn.Sequential(*features)  ## convert it into container and add it to our model class.
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
    print(py_pred,py_pred.shape,type(py_pred))
    print(py_true, py_true.shape, type(py_true))

    fbeta_sklearn = fbeta_score(py_true, py_pred, 2, average='samples')
    print('Scores are {:.3f} (sklearn) '.format(fbeta_sklearn))

#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from tqdm import tqdm
# from openslide import OpenSlide
from pathlib import Path
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

SEED = 42


# In[4]:


train_df = pd.read_csv('./csv_dir/train_outcomes.csv') # biopsy_id, label
test_df = pd.read_csv('./csv_dir/test_outcomes.csv')

train_mapping = pd.read_csv('./csv_dir/train_mapping.csv') # slide_id, biopsy_id, img path
test_mapping = pd.read_csv('./csv_dir/test_mapping.csv')


# In[5]:


train_outcome_map = {}
"""
key: biopsy_id
value: stage_number 0,1,2,3,4 (exclude NaN)
"""
for idx, row in train_df.iterrows():
    train_outcome_map[row['biopsy_id']] = row['label']

train_slide_map = {}
"""
key: slide_id
value: Tuple(biopsy_id, slide_path, label)
"""
for idx, row in train_mapping.iterrows():
    train_slide_map[row['slide_id']] = (row['biopsy_id'], row['downsampled_path'], train_outcome_map[row['biopsy_id']])


# In[6]:


test_outcome_map = {}
"""
key: biopsy_id
value: stage_number 0,1,2,3,4 (exclude NaN)
"""
for idx, row in test_df.iterrows():
    test_outcome_map[row['biopsy_id']] = row['label']

test_slide_map = {}
"""
key: slide_id
value: Tuple(biopsy_id, slide_path, label)
"""
for idx, row in test_mapping.iterrows():
    # print(row['biopsy_id'])
    test_slide_map[row['slide_id']] = (row['biopsy_id'], row['downsampled_path'], test_outcome_map[row['biopsy_id']])


# In[7]:


len(train_slide_map)


# In[8]:


transform_aug_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(size=224,scale=(0.8,1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_aug_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# In[9]:


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, slide_map, mode='train', transform=None): 
        self.slide_map = slide_map
        self.data = list(slide_map.values())
        
        self.mode = mode # train/test
        self.transform = transform

    def __getitem__(self, index):
        biopsy_id, path, label = self.data[index]
        x_pil = Image.open("."+path)
        # if self.mode=='train': x_tensor = transform_aug_train(x_pil)
        # elif self.mode in ['test', 'holdout']: x_tensor = transform_aug_test(x_pil)
        x_tensor = self.transform(x_pil)
        return x_tensor, label, biopsy_id

    def __len__(self):
        return len(self.slide_map)


# In[10]:


batch_size = 4

epochs = 50
learning_rate = 1e-3
momentum = 0.9
weight_decay = 0 # 1e-8

hidden_dim = 2048 # ResNet50
num_classes = 5
out_dim = num_classes # [0,1,2,3,4]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[11]:


train_dataset = ImageDataset(train_slide_map, mode='train', transform=transform_aug_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_dataset = ImageDataset(test_slide_map, mode='test', transform=transform_aug_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


# In[12]:


def ce_loss(y_pred, y_true):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(y_pred, y_true)

criterion = ce_loss


# In[15]:


label_binarizer = LabelBinarizer().fit(np.arange(num_classes))

def get_score(y_true, y_pred):
    """
    # assume y_true: [0,1,2,3,4,3,2,...] discrete numbers
    # assume y_pred: tensor of shape (batch_size, num_classes)
    # where num_classes = 5 for this task

    # compute AUC for each class
    """
    y_true_onehot = label_binarizer.transform(y_true)
    macro_roc_auc_ovr = roc_auc_score(
        y_true_onehot,
        y_pred,
        multi_class="ovr",
        average="macro",
    )
    return macro_roc_auc_ovr


# In[16]:


model = torchvision.models.resnet50()
# model.load_state_dict(torch.load('./checkpoints/resnet50-11ad3fa6.pth'), strict=True)


# In[17]:


model.fc = nn.Sequential(
    # nn.Linear(hidden_dim, hidden_dim//16),
    # nn.GELU(),
    # nn.Linear(hidden_dim//16, out_dim),
    nn.Linear(hidden_dim, out_dim),
)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# In[26]:


def train_epoch(model, dataloader, loss_fn, optimizer):
    train_loss = []
    model.train()
    for step, data in enumerate(dataloader):
        # print(f"Training... Step={step}")
        batch_x, batch_y, batch_biopsy_id = data
        batch_x, batch_y = (
            batch_x.float().to(device),
            batch_y.type(torch.LongTensor).to(device),
        )
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    metric_train_loss = np.array(train_loss).mean()
    return metric_train_loss

def val_epoch(model, dataloader):
    y_pred = {} # key: biopsy_id, value: List[slice_stage_pred]
    y_true = {} # key: biopsy_id, value: List[slice_stage_pred]
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            # if step % 50 == 0: print(f"Validating... Step={step}")
            batch_x, batch_y, batch_biopsy_id = data
            batch_x, batch_y = (
                batch_x.float().to(device),
                batch_y.type(torch.LongTensor).to(device),
            )
            output = model(batch_x)
            # output = torch.argmax(output, dim=-1)
            output = output.detach().cpu().numpy().tolist()
            batch_y = batch_y.detach().cpu().numpy().tolist()

            for i in range(len(batch_biopsy_id)):
                biopsy_id = batch_biopsy_id[i]
                if biopsy_id not in y_pred:
                    y_pred[biopsy_id] = []
                    y_true[biopsy_id] = []
                y_pred[biopsy_id].append(output[i])
                y_true[biopsy_id].append(batch_y[i])
    
    prediction_list = []
    ground_truth_list = []
    for biopsy_id in y_pred:
        preds = np.array(y_pred[biopsy_id])
        truths = np.array(y_true[biopsy_id])
        prediction_list.append(preds.mean(axis=0))
        ground_truth_list.append(truths.mean())
    prediction_list = np.array(prediction_list)
    ground_truth_list = np.array(ground_truth_list)
    # prediction_list = reverse_min_max_norm(prediction_list)
    # nearest_discretize(prediction_list)
    # ground_truth_list = reverse_min_max_norm(ground_truth_list)

    score = get_score(ground_truth_list, prediction_list)
    return score


# In[ ]:


best_score = 1e8
valid_step = 1

for epoch in range(epochs):
    # print(f'Running epoch {epoch} ...')
    # train_loss = train_epoch(model, train_loader, criterion, optimizer)
    # print(f"Epoch {epoch}: Loss = {train_loss}")
    if epoch % valid_step == 0:
        metric_valid = val_epoch(model, test_loader)
        print("Val Score:", metric_valid)
        if metric_valid < best_score:
            best_score = metric_valid
            print(f"Saving best model at Epoch {epoch}")
            torch.save(model.state_dict(), f"./checkpoints/model_resnet_0404.ckpt")


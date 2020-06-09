import torch
from torchvision import transforms
from dataloader import PseudoLabeledDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np
import os
from model import ResModel
import re

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

image_folder = '../data'

unlabeled_train_index = np.arange(90)
unlabeled_test_index = np.arange(90,106)
model = ResModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
print('model loaded')
# [batch_size, 3, H, W] torch.Size([2, 3, 256, 306])
train_dataset = PseudoLabeledDataset(image_folder=image_folder, scene_index=unlabeled_train_index)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
# trainloader = get_loader(train_dataset)
test_dataset = PseudoLabeledDataset(image_folder=image_folder, scene_index=unlabeled_test_index)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=4)
# testloader = get_loader(test_dataset)
print('data loaded')

def check_file(dir):
    matching_file = 'model.pth.tar'
    for f in os.listdir(dir):
        if re.search(matching_file, f):
            return f
    return None 

def save_checkpoint(epoch, model, optimizer, save_dir="./"):
    state = {'model': model,
             'optimizer': optimizer,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'epoch': epoch
             }

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    f = check_file(save_dir)
    if f != None :
        os.remove(os.path.join(save_dir, f))

    print("Saving the model, best performance obtained at {}".format(epoch))
    filename = save_dir + str(epoch) +"_" + 'model.pth.tar'
    torch.save(state, filename)
    
def train(model, device, train_loader, optimizer, epoch, log_interval = 100):
    # Set model to training mode
    do_train = True
    model.train()
    # Loop through data points
    for batch_idx, (data, target) in enumerate(train_loader):    
        # Send data and target to device
        dataX = torch.flatten(data,start_dim=0,end_dim=1)
        labels = torch.flatten(target,start_dim=0,end_dim=1)
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train)).to(device)
        labels_var = torch.autograd.Variable(labels, requires_grad=False).to(device)
        #print(dataX.size(), labels.size())
        # Zero out the optimizer
        optimizer.zero_grad()
        
        # Pass data through model
        pred_var = model(dataX_var)
        ##print(pred_var.size(),labels_var.size()) 
        # Compute the negative log likelihood loss
        #loss = nn.CrossEntropyLoss(pred_var, labels_var)
        loss = F.nll_loss(pred_var, labels_var)
        # Backpropagate loss
        #print('loss computed')
        loss.backward()
        #print('loss backwards')
        # Make a step with the optimizer
        optimizer.step()
        #print('optimizer.step() done')
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Define test method
def test(model, device, test_loader):
    do_train = False
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss 
    test_loss = 0
    # Counter for the correct predictions
    num_correct = 0
    
    # don't need autograd for eval
    with torch.no_grad():
        # Loop through data points
        for (data, target) in test_loader:
            dataX = torch.flatten(data,start_dim=0,end_dim=1)
            labels = torch.flatten(target,start_dim=0,end_dim=1)
            dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train)).to(device)
            labels_var = torch.autograd.Variable(labels, requires_grad=False).to(device)
            # Pass data through model
            pred_var = model(dataX_var)

            # Compute the negative log likelihood loss
            #loss = F.CrossEntropyLoss(pred_var, labels_var)
            #loss = F.nll_loss(pred_var, labels_var)
            # Compute the negative log likelihood loss with reduction='sum' and add to total test_loss
            # sum losses over minibatch
            #test_loss += F.CrossEntropyLoss(pred_var, labels_var, reduction = 'sum').item()
            test_loss += F.nll_loss(pred_var, labels_var, reduction = 'sum').item()
            # Get predictions from the model for each data point
            pred = pred_var.data.max(1, keepdim=True)[1] # get the index 
            
            # Add number of correct predictions to total num_correct 
            num_correct += pred.eq(labels_var.data.view_as(pred)).cpu().sum().item()

    
    # Compute the average test_loss
    avg_test_loss = test_loss / len(test_loader.dataset)
    
    # Print loss (uncomment lines below once implemented)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_test_loss, num_correct, len(test_loader.dataset),
        100. * num_correct / len(test_loader.dataset)))
    
# Training loop with 10 epochs
for epoch in range(1, 10 + 1):
    # Train model
    train(model, device, trainloader, optimizer, epoch)
    # Test model
    test(model, device, testloader)
    save_checkpoint(epoch, model, optimizer, save_dir = 'models/')



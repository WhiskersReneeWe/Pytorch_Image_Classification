import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transformer = {'test': transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor()]),
    'valid': transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor()]),
    'train': transforms.Compose(
     [transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize])
              }
# TODO: Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder('/data/dog_images/train', transform = transformer['train']),
                  'valid': datasets.ImageFolder('/data/dog_images/valid', transform = transformer['valid']),
                  'test': datasets.ImageFolder('/data/dog_images/test/', transform = transformer['test'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32,
                                          shuffle = True),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32 ,
                                         shuffle = True),
               'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32, shuffle = True) 
}

######Define Model Architecture
import torch.nn as nn
import torch.nn.functional as F

inp_size = 224 * 224

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.fc1 = nn.Linear(inp_size, 1024)
        self.fc2 = nn.Linear(1024, 133)
        self.output = nn.LogSoftmax(dim = 1)
    
    def forward(self, x):
        ## Define forward behavior
        x = x.view(-1, inp_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
   
import torch.optim as optim

###  loss function
criterion_scratch = nn.NLLLoss()

###   optimizer
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr = 0.001)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            #dataiter = iter(data)
            #images, labels = dataiter.next()
            optimizer.zero_grad()
            ## find the loss and update the model parameters accordingly
            log_ps = model(data)
            loss = criterion(log_ps, labels)
            loss.backward()
            ## record the average training loss, using something like
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            dataiter = iter(loaders['valid'][batch_idx])
            images, labels = dataiter.next()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= train_loss:
            torch.save(model.state_dict(), save_path)
    # return trained model
    return model


# train the model
loaders_scratch = dataloaders
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


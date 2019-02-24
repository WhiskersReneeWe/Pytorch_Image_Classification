from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt  
import torch
import os
from torchvision import datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
##################imports#####################

# You should load your own training images here
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
                       
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  
# 
# In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.
# 
# ### Write a Human Face Detector
# 
# We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

# In[3]:


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# define VGG16 model
VGG16 = models.vgg16(pretrained=True)
# check if CUDA is available
use_cuda = torch.cuda.is_available()
# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_pipe = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize])
    img = transform_pipe(Image.open(img_path))
    img = img.unsqueeze(0)  

    #PyTorch models expect inputs to be Variables
    img = Variable(img)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        img = img.cuda()

    
    ## Return the *index* of the predicted class for that image
    
    prediction = VGG16(img)  # Returns a Tensor of shape (batch, num class labels)
    prediction = prediction.data.numpy().argmax()  
    
    return prediction # predicted class index


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    pred_idx = VGG16_predict(img_path)
    return ((pred_idx <= 268) & (pred_idx >= 151)) 




## Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transformer = {'test': transforms.Compose(
    [transforms.Resize(255),
    transforms.CenterCrop(224),
     transforms.ToTensor(),
     normalize]),
    'valid': transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     normalize]),
    'train': transforms.Compose(
     [transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize])
              }
# Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder('/data/dog_images/train', transform = transformer['train']),
                  'valid': datasets.ImageFolder('/data/dog_images/valid', transform = transformer['valid']),
                  'test': datasets.ImageFolder('/data/dog_images/test/', transform = transformer['test'])}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32,
                                          shuffle = True),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32,
                                         shuffle = True),
               'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32, shuffle = True) 
}




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
              
            optimizer.zero_grad()
            ## find the loss and update the model parameters accordingly
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
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
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
            }, save_path)
    # return trained model
    return model


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# Specify model architecture 
# Step 1 - Use pretrained vgg16 as the convo layers

vgg16 = models.vgg16(pretrained=True)  

# Don't change the weights from the pre-trained model feature layers.
for param in vgg16.parameters():
    param.requires_grad = False

# Step 2 - Define a new, untrained classifier

input_size = 25088
hidden_sizes = [2048, 1024]
output_size = 133
my_classifier = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Dropout(p = 0.25),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Dropout(p = 0.2),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.LogSoftmax(dim=1))

vgg16.classifier = my_classifier

if use_cuda:
    vgg16 = vgg16.cuda()

criterion_transfer = nn.NLLLoss()
optimizer_transfer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)


# train the model
model_transfer = train(25, dataloaders, vgg16, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')



# load the model that got the best validation accuracy (uncomment the line below)
ckpt = torch.load('model_transfer.pt', map_location=lambda storage, loc: storage)
model_transfer.load_state_dict(ckpt['model_state_dict'])
# check if model_transfer is properly loaded back
print(model_transfer)

test(dataloaders, model_transfer, criterion_transfer, use_cuda)


# Write a function that takes a path to an image as input
# and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]

def predict_breed_transfer(img_path):
    #load the previously trained model
    ckpt = torch.load('model_transfer.pt', map_location=lambda storage, loc: storage)
    vgg16 = models.vgg16(pretrained=True)  

    # Don't change the weights from the pre-trained model feature layers.
    for param in vgg16.parameters():
        param.requires_grad = False


    input_size = 25088
    hidden_sizes = [2048, 1024]
    output_size = 133
    my_classifier = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Dropout(p = 0.25),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Dropout(p = 0.2),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.LogSoftmax(dim=1))

    vgg16.classifier = my_classifier

    vgg16.load_state_dict(ckpt['model_state_dict'])
    
    
    # load the image and return the predicted breed
    img = transformer['test'](Image.open(img_path))
    img = img.unsqueeze(0)  

    #PyTorch models expect inputs to be Variables
    img = Variable(img)
    vgg16.eval()
    idx = vgg16(img)  
    ## Return the *index* of the predicted class for that image
    idx = idx.data.numpy().argmax() 
    breed = class_names[idx]
    
    return breed


def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    breed = predict_breed_transfer(img_path)
    if face_detector(img_path):
        title = 'This photo looks like\na/an {}'.format(breed)
        
    elif dog_detector(img_path):
        title = 'This is\nan/a {}'.format(breed)
    
    else:
        title = 'ERROR: This image is not recognizable!'

    
    return title



for num, file in enumerate(files):
    plt.imshow(plt.imread(file))
    plt.subplot(4,2,num+1)
    plt.title(run_app(file))
    plt.axis('off')
    plt.imshow(img)


test_files = np.array(glob("/home/workspace/dog_project/images/*"))
hooman_files = []
test_dog_files = []

test_dog_files.append('/home/workspace/dog_project/images/husky.jpg')
test_dog_files.append('/home/workspace/dog_project/images/eskimo.jpg')
test_dog_files.append('/home/workspace/dog_project/images/shiba.jpg')


for path in test_files[10:12]:
    hooman_files.append(path)
hooman_files.append('/home/workspace/dog_project/images/my_cat.jpg')
files = np.hstack((hooman_files, test_dog_files))

for num, file in enumerate(files):
    plt.imshow(plt.imread(file))
    plt.subplot(3,2,num+1)
    plt.title(run_app(file))
    plt.axis('off')
    plt.imshow(img)






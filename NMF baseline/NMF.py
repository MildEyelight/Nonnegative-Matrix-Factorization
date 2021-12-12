# NMF baseline algorithm
# In this code, use the dataset of FashionMNIST to factorize a image and show how NMF understand the parts of an object
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as functional
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.models as models
import matplotlib

#hyperparameter
r = 2
epoch = 100
epsilon = 1e-1

# NMF Algorithm: A = W*H
def NMF(A):
    #Image data have 3 channels, but in this case we use gray-scale image and channels = 1.
    #use view()method convert A into a 2-rank tensor(matrix).
    channels,m,n = A.size()
    A = A.view(m,n)
    
    # Sometimes the original matrix will have an all-zero column and thus causing the denominator of updating formula be zero
    # To prevent this, we add a small number on every element of original matrix A.
    A += 1e-4
    
    #random init.
    W = abs(torch.randn(m,r))
    H = abs(torch.randn(r,n))
    
    #train
    for x in range(epoch):
        a = torch.mm(W,H,out=None)
        E = A - a 
        error = torch.norm(E).pow(2)
        if error<epsilon : break

        #The updating rule of W and H.
        W = W *torch.mm(A,H.t())/torch.mm(torch.mm(W,H),H.t())
        H = H *torch.mm(W.t(),A)/torch.mm(torch.mm(W.t(),W),H)
    return W,H
  
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
figure = plt.figure(figsize=(8, 8))
# get an image for dataset, the index can be changed to use different image.
img,label = training_data[1]
img = torch.Tensor(img)
plt.imshow(img.squeeze(), cmap="gray")
plt.show()
  
W,H = NMF(img)

# Visulize the result.
plt.imshow(W.squeeze(), cmap="gray")
plt.show()
plt.imshow(H.squeeze(), cmap="gray")
plt.show()

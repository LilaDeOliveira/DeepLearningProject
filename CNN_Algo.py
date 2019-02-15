import sys 
import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

#rename the folder to replace space by _
def read_folder(path):
	file=os.listdir(path)
	for name in file:
		if name.find(' ') != -1:
			os.rename(path+'/'+ name, path +'/'+name.replace(' ','_'))

path_train = "fruits_360/fruits-360/Training"
path_test = "fruits_360/fruits-360/Test"

#Call the function to rename sub-folders
read_folder(path_train)
read_folder(path_test)

#Create datasets
train_dataset = datasets.ImageFolder(path_train,transform = transforms.ToTensor())
train_loader = DataLoader(train_dataset,batch_size = 4, shuffle = True)


#Create datasets
test_dataset = datasets.ImageFolder(path_test,transform = transforms.ToTensor())
test_loader = DataLoader(test_dataset,batch_size = 4, shuffle = True)

#Model of TD3
class Net(nn.Module):
	'''7.Definethelayersinthenetwork'''
	def __init__(self):
		super(Net,self).__init__()

		#1input imagechannel,6outputchannels,5x5squareconvolutionkernel
		self.conv1=nn.Conv2d(3,6,5)
		self.conv2=nn.Conv2d(6,16,5)
		
		#anaffineoperation:y=Wx+b
		self.fc1=nn.Linear(16*22*22,120)#(sizeofinput,sizeofoutput)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,83)
		
		'''Implementtheforwardcomputationofthenetwork'''

	def forward(self,x):
		#Maxpoolingovera(2,2)window
		x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		#Ifthesizeisasquareyoucanonlyspecifyasinglenumber
		x=F.max_pool2d(F.relu(self.conv2(x)),2)
		x=x.view(-1,self.num_flat_features(x))
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

	def num_flat_features(self,x):
		size=x.size()[1:]
		#alldimensionsexceptthebatchdimension
		num_features=1
		for s in size:
			num_features*=s
		return num_features

net=Net()
print(net)

# choose the loss function and update rule
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum = 0.9)

epochs=5
for epoch in range(epochs):
	running_loss=0.0
	for i, data in enumerate(train_loader,0):
		#get the input
		inputs,labels=data

		# clear the parameter gradients
		optimizer.zero_grad()

		#forward+backward+optimize
		outputs=net(inputs)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		running_loss+=loss.item()
		if i % 2000 == 1999: 
			print ('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
			running_loss=0.0
	print ('Finished Training')

def imshow(img):
	img=img/2+0.5
	npimg=img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))

dataiter=iter(test_loader)
images,labels=dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images))

outputs=net(images)
_,predicted=torch.max(outputs,1)

correct=0
total=0
with torch.no_grad():
	for data in test_loader:
		images, labels=data
		outputs=net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print ('Accuracy of the network on the test images: %d %%' %(100* correct / total))
plt.show()


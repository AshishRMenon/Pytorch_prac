import torch
import torchvision
import torchvision.transforms as T



#------------------------------------------------------------------------------------------------------------------------------------------

#STEP1: Loading dataset with suitable transforms 


#general transformation to be applied to all images
transform=T.compose([T.toTensor(),T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#Get the training dataset apply the transforms and apply load it in a variable with appt. batch size shuffling etc.
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size = 4,shuffle=True, num_workers=2)


#Get the training dataset apply the transforms and apply load it in a variable with appt. batch size shuffling etc.
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transofrm=transform)
testloader=torch.utils.data.Dataloader(testset,shuffle=True,batch_size=4,num_workers=2)



classes=('plane' , 'car', 'bird' , 'cat', 'deer','dog', 'frog', 'horse', 'ship' , 'truck')

#------------------------------------------------------------------------------------------------------------------------------------------

#STEP2:Displaying Training Images

import matplotlib.pyplot as plt
import numpy as np

def imgsave(img):
	img=img/2 + 0.5 #Unnormalize
	npimg=img.numpy()
	plt.imsave('sample.jpg',np.transpose(npimg, (1,2,0)))

dataiter=iter(trainloader)
images,labels=dataiter.next()
imgsave(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#------------------------------------------------------------------------------------------------------------------------------------------


#STEP3 : Define a Network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Conv2d(3,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)
	def forward(self,x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.pool(F.relu(self.conv2(x)))
		x=x.view(-1,16*5*5)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

#--------------------------------------------------------------------------------------------------------------------------------------------

#STEP4: Loss Function and Optimizer

net=Net()
import torch.optim as optim
criterion= nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#--------------------------------------------------------------------------------------------------------------------------------------------

#Step5: Train Network

for epoch in range(2):
	running_loss=0
	for i, data in enumerate(trainloader,0):
		inputs,labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		print(loss)
		print(loss.item())
		if i%2000 == 1999:
			print('[%d %5d] loss: %.3f' % (epoch +1 , i+1, running_loss/2000))
			running_loss=0.0
print('Finished Training')

#---------------------------------------------------------------------------------------------------------------------------------------------

#Step5: Testing

images, labels = iter(testloader).next()
output_test = net(images)
_ , predictions = torch.max(outputs,1) 

with torch.no_grad():
	for data in testloader:
		images,labels=data
		outputs=net(images)
		_, predictions=torch.max(outputs.data,1)
		total += labels.size(0)
		correct += (predicted==labels).sum.item()
print('Accuracy {}'.format(100*correct/total))


		

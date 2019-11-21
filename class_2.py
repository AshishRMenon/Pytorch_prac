import torch
import torchvision
from torchvison.trasnforms import T

transform=T.compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

traindata=torchvison.datasets.CIFAR10(root='./data', train=True, download=True, transforms=transform)
trainloader=torch.utils.data.Dataloader(traindata, batch_size=4, shuffle=True)

testdata=torchvison.datasets.CIFAR10(root='./data', train=False, download=True, transforms=transform)
testloader=torch.utils.data.Dataloader(testdata, batch_size=4, shuffle=True)

def imgshow(image):
	imp_np = image.numpy()
	img_np = img_np*0.5+0.5
	plt.imsave('sample.jpg', np.transpose(img_np,(1,2,0)))

image,labels=iter(trainloader).next()
imgshow(torchvision.utils.make_grid(image))
print(join(classes(labels[j]) for j in range(4)))


import torch.nn as nn
import torch.nn.functional as F


class Network(nn.module):
	def __init__(self):
		self.cnv1=nn.Conv2d()
		self.pool1=nn.Maxpool()
		self.cnv2=nn.Conv2d()
		self.pool2=nn.Maxpool()
		self.fc1=nn.Linear()
		self.fc2=nn.Linear()
		self.fc3=nn.Linear()	

	def forward(self,x):
		x=self.pool1(F.relu(self.cnv1(x)))
		..
	
net=Netork()
import torch.optim as optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(Network.parameters() , lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss=0
	for i,data in enumerate(trainloader):
		images,labels=data
		optimizer.zero_grad()
		output = net(images)
		loss = criterion(output,label)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i%100==0:
			print("epoch{}, iteration{i} , Loss={}".format(epoch, i, running_loss))	


correct=0
total=0

with torch.no_grad():
	for data in testloader:
		image,label=data
		output_test=net(images)
		_, predictions = torch.max(output,1)
		total+=labels.size(0)
		correct+=(predicted==labels).sum().item()
		
print('Accuracy={}%'.format(100*correct/total))
 		



		
		

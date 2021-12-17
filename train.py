import torch
import torch.nn as nn
import torchvision
from model import Lenet
import torch.optim as optim
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    #下载训练数据集
    train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=50,shuffle=True,num_workers=16)
    #下载测试数据
    test_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=5000,shuffle=False,num_workers=16)
    test_data_iter = iter(test_loader)
    test_image,test_label = test_data_iter.next()
    #数据类别信息
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Lenet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    print('start training')

    for epoch in range(100):
        running_loss = 0.0
        for step,data in enumerate(train_loader,start=0):
            inputs,labels = data
            optimizer.zero_grad()
            outputs  = net(inputs)
            loss  = loss_function(outputs,labels)
            loss.backward()
            optimizer.step() #更新反向传播的参数

            #打印训练信息
            running_loss += loss.item()
            if step %500 == 499:
                with torch.no_grad():
                    outputs = net(test_image)
                    predict_y = torch.max(outputs,dim=1)[1]
                    accuracy = torch.eq(predict_y,test_label).sum().item() / test_label.size(0)
                    print('[%d,%5d] train_loss:%.3f test_accuracy:%.3f'%(epoch+1,step+1,running_loss/500,accuracy))
                    running_loss = 0.0
    print('Finished Training')
    save_path = './Lenet.pth'
    torch.save(net.state_dict(),save_path)

if __name__ == '__main__':
    main()


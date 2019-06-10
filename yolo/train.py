import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from net import vgg16_bn
from resnet_yolo import resnet50
from yoloLoss import yoloLoss
from dataset import yoloDataset

import numpy as np
import time 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--net", type=str, default="resnet50", help="Network option: Resnet50 or VGG16 with batch normalization")
    opt = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    file_root = './data/JPEGImages/'
    learning_rate = opt.learning_rate
    num_epochs = opt.epochs
    batch_size = opt.batch_size
    img_size = opt.img_size

    cont = False  # Continue or not
    
    if opt.net == "resnet50":
        net = resnet50()
        print('Using network Resnet50')
        if os.path.exists("yolo"+opt.net+".pth"):
            net.load_state_dict(torch.load('yolo'+opt.net+'.pth'))
            print("load checkpoint model success")
            print("Resuming......")
            cont = True
        else:
            resnet = models.resnet50(pretrained=True)
            new_state_dict = resnet.state_dict()
            dd = net.state_dict()
            for k in new_state_dict.keys():
                if k in dd.keys() and not k.startswith('fc'):
                    dd[k] = new_state_dict[k]
            net.load_state_dict(dd)
            print("load pre-trined model success")
    elif opt.net == "vgg16_bn":
        net = vgg16_bn()
        print('Using network VGG16')
        if os.path.exists('yolo'+opt.net+'.pth'):
            net.load_state_dict(torch.load('yolo'+opt.net+'.pth'))
            print("load checkpoint model success")
            print("Resuming")
            cont = True
        else:
            vgg = models.vgg16_bn(pretrained=True)
            new_state_dict = vgg.state_dict()
            dd = net.state_dict()
            for k in new_state_dict.keys():
                if k in dd.keys() and k.startswith('features'):
                    dd[k] = new_state_dict[k]
            net.load_state_dict(dd)
            print("load pre-trined model success")
    print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

    criterion = yoloLoss(7,2,5,0.5)
    if use_gpu:
        net.cuda()

    net.train()
    # different learning rate
    params=[]
    params_dict = dict(net.named_parameters())
    for key,value in params_dict.items():
        if key.startswith('features'):
            params += [{'params':[value],'lr':learning_rate*1}]
        else:
            params += [{'params':[value],'lr':learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_dataset = yoloDataset(root=file_root,list_file='./voc2012train.txt',train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = yoloDataset(root=file_root,list_file='./voc2012val.txt',train=False,transform = [transforms.ToTensor()] )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('the training dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    
    if cont and os.path.exists('continue.txt'):
        contfile = open('continue.txt', 'r')
        lines = contfile.readlines()
        epoch_cont = int(lines[0].split('=')[1])
        batch_cont = int(lines[1].split('=')[1])
        best_test_loss = float(lines[2].split('=')[1])
        total_loss = float(lines[3].split('=')[1])
        contfile.close()
    else:
        best_test_loss = np.inf
        epoch_cont = 0
        batch_cont = 0
        total_loss = 0.

    num_iter = 0

    logfile = open('log.txt', 'w')
    for epoch in range(num_epochs):
        if epoch < epoch_cont:
            continue
        net.train()
        torch.cuda.empty_cache()
        if epoch == 30:
            learning_rate = 0.0005
        if epoch == 40:
            learning_rate = 0.00005
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        tc = 0.

        for i,(images,target) in enumerate(train_loader):
            ts = time.time()
            torch.cuda.empty_cache()
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()

            # Manage CUDA memory
            try:
                pred = net(images)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        pred = net(images)
                else:
                    raise exception
            loss = criterion(pred,target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tc += (time.time() - ts)
            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss of training: %.4f'
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
                logfile.writelines('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss of training: %.4f'
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)) + '\n')  
                logfile.flush()
                num_iter += 1
                torch.save(net.state_dict(),'yolo'+opt.net+'.pth')
                contfile = open('continue.txt', 'w')
                contfile.writelines(' epoch=%d \n batch=%d \n best_loss=%.4f \n total_loss=%.4f\n' %(epoch, i, loss.item(), total_loss/(i+1)))
                print('Elapsed time: %.3f'%tc)
                tc = 0.

        #validation
        validation_loss = 0.0
        net.eval()
        tc= 0.
        for i,(images,target) in enumerate(test_loader):
            ts = time.time()
            images = Variable(images, volatile=True)
            target = Variable(target, volatile=True)
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            try:
                pred = net(images)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        pred = net(images)
                else:
                    raise exception
            loss = criterion(pred,target)
            validation_loss += loss.item()
            tc += (time.time()-ts)
            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average loss of validation: %.4f' 
                %(epoch+1, num_epochs, i+1, len(test_loader), loss.item(), validation_loss / (i+1)))
                logfile.writelines('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average loss of validation: %.4f' 
                %(epoch+1, num_epochs, i+1, len(test_loader), loss.item(), validation_loss / (i+1)) + '\n')
                logfile.flush()
                print('Elapsed time: %.3f'%tc)
                tc = 0.
        validation_loss /= len(test_loader)
        
        # Best network
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(),'best'+opt.net+'.pth')
        logfile.writelines('Epoch: ' + str(epoch) + 'Best Loss: '+ str(validation_loss) + '\n')
        logfile.flush()      
        
    


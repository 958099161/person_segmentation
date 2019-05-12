import os 
import time
import math
import torch
from tensorboardX import SummaryWriter
from eval import *
import torch.nn as nn
from utils import evalIoU
from networks import get_model
from torch.autograd import Variable
from dataloader.dataset import SRdata
from torch.utils.data import DataLoader
from dataloader.transform import MyTransform
from torchvision.transforms import ToPILImage
from options.train_options import TrainOptions
from torch.optim import SGD, Adam, lr_scheduler
from criterion.criterion import CrossEntropyLoss2d
NUM_CHANNELS = 3

def get_loader(args):
    dataset_train = SRdata(args)     #(imagepath_train, labelpath_train,  train_transform) #DataSet
    dataset_val =SRdata(args)    #NeoData(imagepath_val, labelpath_val, val_transform)
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=2, shuffle=False)
    return loader, loader_val

def train(args, model):
    NUM_CLASSES = args.num_classes #pascal=21, cityscapes=20
    savedir = args.savedir
    weight = torch.ones(NUM_CLASSES)
    loader, loader_val = get_loader(args)
    if args.cuda:
        criterion = CrossEntropyLoss2d(weight).cuda() 
    else:
        criterion = CrossEntropyLoss2d(weight)
    #save log
    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")
    
    optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch            
    start_epoch = 1   
    if args.fineture:

        model.load_state_dict(torch.load(args.model_dir))
        model.train()
        print('load model:'+args.model_dir+' success....')
    print('train...'+args.model)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)
    writer = SummaryWriter(args.summary_dir)
    global_step = 0
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        scheduler.step(epoch)   
        epoch_loss = []
        time_train = []
        #confmatrix for calculating IoU   
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
        perImageStats = {}
        nbPixels = 0
        usedLr = 0
        #for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])
        count = 1
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()
            if args.cuda:
                images = Variable(images.cuda().float())
                labels = Variable(labels.cuda().float())
            # inputs = Variable(images)
            # targets = Variable(labels)
            outputs = model(images)      #  2  3  h  w  cpu   float
            # writer.add_graph(model,(inputs))
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            # time_train.append(time.time() - start_time)
            #================================================================================================================
            # outputs
            #Add outputs to confusion matrix    #CODE USING evalIoU.py remade from cityscapes/scripts/evaluation/evalPixelLevelSemanticLabeling.py
            # if (args.iouTrain):
            #     add_to_confMatrix(outputs, labels, confMatrix, perImageStats, nbPixels)
            global_step += 1
            #
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')

                writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)

                print('loss: {} (epoch: {}, step: {})'.format(average,epoch,step))#,
                      #  "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
            
        # average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouAvgStr, iouTrain, classScoreList = cal_iou(evalIoU, confMatrix)
        # print ("EPOCH IoU on TRAIN set: ", iouAvgStr)
                       
        # calculate eval-loss and eval-IoU
        # average_epoch_loss_val, iouVal = eval(args, model, loader_val, criterion, epoch)
        
        #save model every X epoch
        # writer.add_graph(net)

        if epoch % args.epoch_save==0:
            torch.save(model.state_dict(), './save_models/{}_3_11_non_big{}.pth'.format(args.model,str(epoch)))
        writer.add_scalar('epoch_loss', sum(epoch_loss) / len(epoch_loss), global_step=epoch)
       #记录下每一个batch的loss
        txtName = "loss_batch.txt"
        f = open(txtName, 'a+')
        # f=file(txtName, "a+")

        new_context = 'epoch_loss--       ' + str(epoch)+'         '+str(sum(epoch_loss) / len(epoch_loss))+'\n'
        f.write(new_context)
        f.close()
        #save log
        # with open(automated_log_path, "a") as myfile:
        #     myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    print('Finished Training')
    writer.close()
    return(model)   
    
def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = '{}'.format(args.savedir)
    modeltxtpath = os.path.join(savedir,'model.txt') 

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile: #record options
        myfile.write(str(args))
    model = get_model(args)     #load model
    with open(modeltxtpath, "w") as myfile:  #record model
        myfile.write(str(model))
    if args.cuda:
        model = model.cuda() 
    print("========== TRAINING ===========")
    train(args,model)
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':

    parser = TrainOptions().parse()
    main(parser)

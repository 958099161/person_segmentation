
from eval import *
import torch
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
import time
NUM_CHANNELS = 3

def eval(args, model, loader_val, criterion, epoch):
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.cuda()
        model.eval()
        epoch_loss_val = []
        time_val = []

        #New confusion matrix 
        confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args) #21x21
        perImageStats = {}
        nbPixels = 0
        
        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            inputs = Variable(images, volatile=True)
            targets = Variable(labels, volatile=True)

            outputs = model(inputs.float())
            loss = criterion(outputs, targets.long())

            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)

            average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

            if args.iouVal: # add to confMatrix
                add_to_confMatrix(outputs, labels,confMatrix, perImageStats, nbPixels)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
            print('VAL loss: {} (epoch: {}, step: {})'.format(average,epoch,step),
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                        
        average_epoch_loss_train = sum(epoch_loss_val) / len(epoch_loss_val)
        iouAvgStr, iouVal, classScoreList = cal_iou(evalIoU, confMatrix)
        print ("EPOCH IoU on VAL set: ", iouAvgStr)

        return average_epoch_loss_val, iouVal
            
def add_to_confMatrix(prediction, groundtruth, confMatrix, perImageStats, nbPixels):
    if isinstance(prediction, list):   #merge multi-gpu tensors
        outputs_cpu = prediction[0].cpu()
        for i in range(1,100):#len(outputs)):
            outputs_cpu = torch.cat((outputs_cpu, prediction[i].cpu()), 0)
    else:
        outputs_cpu = prediction.cpu()
    for i in range(0, outputs_cpu.size(0)):   #args.batch_size,evaluate iou of each batch
        prediction = ToPILImage()(outputs_cpu[i].max(0)[1].data.unsqueeze(0).byte())
        groundtruth_image = ToPILImage()(groundtruth[i].cpu().byte())
        nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth_image, confMatrix, perImageStats, evalIoU.args)    

def cal_iou(evalIoU, confMatrix):
        iou = 0
        classScoreList = {}
        for label in evalIoU.args.evalLabels:
            labelName = evalIoU.trainId2label[label].name
            classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)

        iouAvgStr  = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args), evalIoU.args) + "{avg:5.3f}".format(avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + evalIoU.args.nocol
        iou = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
        return iouAvgStr, iou, classScoreList
    
if __name__=='__main__':
    def get_loader(args):
        dataset_train = SRdata(args)  # (imagepath_train, labelpath_train, train_transform) #DataSet
        dataset_val = SRdata(args)  # NeoData(imagepath_val, labelpath_val, val_transform)
        loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=2, shuffle=False)
        return loader, loader_val


    parser = TrainOptions().parse()
    loader, loader_val = get_loader(parser)
    model = get_model(parser)  # load model
    criterion = CrossEntropyLoss2d()

    eval(parser, model, loader, criterion, 1)
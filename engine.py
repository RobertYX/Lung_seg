import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

from loss import WCEDCELoss, MSELoss
from tensorboardX import SummaryWriter
from utils import *

import config as cfg
join = os.path.join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = WCEDCELoss(num_classes=cfg.num_class, inter_weights=0.5, intra_weights=torch.tensor([1.0, 1.0]).to(device), cf='ce')
criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]).to(device))

Out_Arte = cfg.Out_Arte
time_now = time.strftime(r'%Y%m%d-%H%M%S', time.localtime())
writer = SummaryWriter(join(Out_Arte, 'TensorB')) # loss直观图
log_me = logger(join(Out_Arte, 'LOG_{}'.format(time_now))+'.log') # 创建监察记录

def train_tn(model, optim, data_loader, device, epoch):
    """
    多任务　训练器

    """
    DE = 40
    model.train()
    train_loss1 = 0.0
    train_loss2 = 0.0

    for iteration, data in enumerate(data_loader):
        image, cls, label = data
        
        image = image.to(device)
        cls = cls.to(device)
        label = label.to(device)
        
        optim.zero_grad()
        pred1, pred2 = model(image)
        loss1 = criterion_ce(pred1, cls.long())
        loss2 = criterion(pred2, label.squeeze(1).long())

        loss = loss1 + loss2
        loss.backward()
        optim.step()

        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        if iteration % DE == 0:
            log_me.info("Train: Epoch {}\t"
                        "iteration/iterations {}/{}\t"
                        "Loss1 {:.3f}\t"
                        "Loss2 {:.3f}".format(epoch, iteration, len(data_loader), loss1.item(), loss2.item()))

    train_aver_loss1 = train_loss1 / len(data_loader) / 1
    train_aver_loss2 = train_loss2 / len(data_loader) / 1
    writer.add_scalar("train_loss1", train_aver_loss1, epoch)
    writer.add_scalar("train_loss2", train_aver_loss2, epoch)
    log_me.info("train_aver_loss1:  {}".format(train_aver_loss1))
    log_me.info("train_aver_loss2:  {}\n".format(train_aver_loss2))

def valid_tn(model, data_loader, device, epoch): 
    """
    多任务　验证器
    """
    DE = 100
    model.eval()
    val_acc = 0.0; val_acc_0 = 0; val_acc_1 = 0; val_all_0 = 0; val_all_1 = 0; 
    val_dice = 0.0


    for iteration, data in enumerate(data_loader):
        image, cls, label = data
        
        with torch.no_grad():
            image = image.to(device)
            cls = cls.to(device)
            label = label.to(device)

            pred1, pred2 = model(image)

            # if augmentation:
            #     from torchvision.transforms import functional as F
            #     inp_v = F.vflip(image)
            #     cls, out_v = model(inp_v)
            #     out_v = F.vflip(out_v)
            #
            #     inp_h = F.hflip(image)
            #     cls, out_h = model(inp_h)
            #     out_h = F.hflip(out_h)
            #
            #     inp_180 = F.rotate(image, angle=180, resample=False, expand=False, center=None)
            #     cls, out_180 = model(inp_180)
            #     out_180 = F.rotate(out_180, angle=-180, resample=False, expand=False, center=None)


            preds1 = F.softmax(pred1, dim=1); preds1 = preds1.data.cpu().numpy(); preds1 = np.argmax(preds1, axis=1)
            gts = cls.data.cpu().numpy();gt = gts[0]
            if gt==0:
                val_all_0 += 1
                if gt==preds1:
                    val_acc_0 += 1; val_acc+=1
            else:
                val_all_1 += 1
                if gt == preds1:
                    val_acc_1 += 1; val_acc+=1
            

            dice = dice_coeff(pred2, label.squeeze(1).long())

            val_dice += dice
        
        if iteration % DE == 0:
            log_me.info("Val: Epoch {}\t"
                        "iteration/iterations {}/{}\t"
                        "val dice {:.3f}".format(epoch, iteration, len(data_loader), dice))

    acc_rate_0 = val_acc_0 / val_all_0 / 1
    acc_rate_1 = val_acc_1 / val_all_1 / 1
    FP = val_all_0 - val_acc_0
    FN = val_all_1 - val_acc_1
    TP = val_acc_1
    TN = val_acc_0

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2*precision*recall/(precision + recall + 1e-8)

    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)

    val_aver_acc = val_acc / len(data_loader) / 1
    val_aver_dice = val_dice / len(data_loader) / 1
    writer.add_scalar("f1_score", f1, epoch)
    writer.add_scalar("val_dice", val_aver_dice, epoch)
    log_me.info("val_aver_acc:  {}".format(f1))
    log_me.info("val_aver_dice:  {}\n".format(val_aver_dice))
    return val_aver_dice

def train(model, optim, data_loader, device, epoch):
    """
    单任务训练器
    """
    DE = 80
    model.train()
    train_loss = 0.0
    for iteration, data in enumerate(data_loader):
        image, label = data
        
        image = image.to(device)
        label = label.to(device)
        # new_shape = (96 // 2, 96 // 2, 96 // 2)
        # image = F.interpolate(image, size=new_shape, mode='trilinear', align_corners=False)
        # label = F.interpolate(label, size=new_shape, mode='trilinear', align_corners=False)

        print(image.size())
        print(label.size())
        optim.zero_grad()
        pred = model(image)
        print(pred.size())
        loss = criterion(pred, label.squeeze(1).long())
        # loss = criterion2(pred, label.squeeze(1).long())
        #loss = loss_f(pred, label)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        if iteration % DE == 0:
            log_me.info("Train: Epoch {}\t"
                        "iteration/iterations {}/{}\t"
                        "Loss {:.3f}".format(epoch, iteration, len(data_loader), loss.item()))
    train_aver_loss = train_loss / len(data_loader) / 1
    writer.add_scalar("train_loss", train_aver_loss, epoch)
    log_me.info("train_aver_loss:  {}\n".format(train_aver_loss))

def valid(model, data_loader, device, epoch):
    """
    单任务验证器
    """
    DE = 1000
    model.eval()
    val_dice = 0.0
    
    for iteration, data in enumerate(data_loader):
        image, label = data
        
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            preds = model(image)
            # loss = criterion(preds, label.squeeze(1).long())
            dice = dice_coeff(preds, label.squeeze(1).long())
            val_dice += dice
        
        if iteration % DE == 0:
            log_me.info("Val: Epoch {}\t"
                        "iteration/iterations {}/{}\t"
                        "val dice {:.3f}".format(epoch, iteration, len(data_loader), dice))

        # Metrics computation
        # label_npy = label.data.cpu().numpy()
        # label_npy = label_npy.astype(np.uint8)
        # label_npy = label_npy.squeeze(axis=1)

        # preds = torch.softmax(preds, dim=1)
        # preds = torch.argmax(preds, dim=1)
        # preds = preds.data.cpu().numpy()
        # preds = preds.astype(np.uint8)
    
    val_aver_dice = val_dice / len(data_loader) / 1
    writer.add_scalar("val_dice", val_aver_dice, epoch)
    log_me.info("val_aver_dice:  {}\n".format(val_aver_dice))
    return val_aver_dice

def train_sp(model, optim, data_loader, device, epoch):
    """
    原始版本
    """
    DE = 40
    model.train()
    train_loss = 0.0
    for iteration, data in enumerate(data_loader):
        image, label = data
        
        image = image.to(device)
        label = label.to(device)
        
        optim.zero_grad()
        pred = model(image)
        loss = criterion(pred, label.squeeze(1).long())
        #loss = loss_f(pred, label)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        if iteration % DE == 0:
            log_me.info("Train: Epoch {}\t"
                        "iteration/iterations {}/{}\t"
                        "Loss {:.3f}".format(epoch, iteration, len(data_loader), loss.item()))
    train_aver_loss = train_loss / len(data_loader) / 1
    writer.add_scalar("train_loss", train_aver_loss, epoch)
    log_me.info("train_aver_loss:  {}\n".format(train_aver_loss))

def valid_sp(model, data_loader, device, epoch):
    """
    原始版本
    """
    DE = 20
    model.eval()
    val_loss = 0.0
    
    for iteration, data in enumerate(data_loader):
        image, label = data
        
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            preds = model(image)
            loss = criterion(preds, label.squeeze(1).long())         
            val_loss += loss.item()
        
        if iteration % DE == 0:
            log_me.info("Val: Epoch {}\t"
                        "iteration/iterations {}/{}\t"
                        "val loss {:.3f}".format(epoch, iteration, len(data_loader), loss.item()))

        # Metrics computation
        # label_npy = label.data.cpu().numpy()
        # label_npy = label_npy.astype(np.uint8)
        # label_npy = label_npy.squeeze(axis=1)

        # preds = torch.softmax(preds, dim=1)
        # preds = torch.argmax(preds, dim=1)
        # preds = preds.data.cpu().numpy()
        # preds = preds.astype(np.uint8)
    
    val_aver_loss = val_loss / len(data_loader) / 1
    writer.add_scalar("val_loss", val_aver_loss, epoch)
    log_me.info("valid_aver_loss:  {}\n".format(val_aver_loss))
    return val_aver_loss

if __name__ == '__main__':
    time_now = time.strftime(r'%Y%m%d-%H%M%S', time.localtime())
    print(time_now)
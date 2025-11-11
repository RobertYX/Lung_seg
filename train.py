#################################################
#      CreateTime:20221026                   
#      UpdateTIme:20230315(The Last)         
#      Creator: FuZhou University iipa.fzu.edu.cn
#      Main Editor: Gunhild
#      "To be or not to be." --ShakeSpear
#################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.utils.data.dataloader as dl
import torch.optim as optim

from create_model import model
from dataset import *
from engine import train, valid, log_me

import config as cfg
join = os.path.join


train_root = cfg.FixDataPath + 'train1'
valid_root = cfg.FixDataPath + 'valid1'
  
# 设置训练设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 输入输出
in_channel = cfg.in_channel
num_classes = cfg.num_class
Out_Arte = cfg.Out_Arte
ModelSavepath = cfg.ModelSavepath

# 是否多任务
multi_task = False
# 设置数据集
single_channel = False
# 是否2.5D

train_dataset = Luna16_DataSet(train_root, augmentation=True)
valid_dataset = Luna16_DataSet(valid_root, augmentation=False)

train_loader = dl.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
valid_loader = dl.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# 设置model
checkpoint = torch.load('F:\PY\lung_seg\output\ResUNet3D\model_saved\model_best_eval.pth')
model.load_state_dict(checkpoint)
model = model.to(device)
# model = nn.DataParallel(model, device_ids=[0,1]) # 多GPU
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# 优化器optimizer
# optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
num_epochs = cfg.epoch
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs//2)

# 期望验证 loss和dice
# best_valid_loss = 10.0
best_valid_dice = 0.96

# 超过期望验证的集合
best_epoch = []
best_dice = []

# 当前最佳
eval_epoch = 0
eval_dice = 0.0


if __name__ == '__main__':

    for epoch in range(num_epochs):


        train(model, optimizer, train_loader, device, epoch)
            
        lr_scheduler.step()

        valid_dice = valid(model, valid_loader, device, epoch)


        # 保存最新权重
        torch.save(model.state_dict(), join(ModelSavepath, "model_last.pth"))
        # 保存当前最好权重
        if valid_dice > eval_dice:
            eval_dice = valid_dice
            eval_epoch = epoch
            torch.save(model.state_dict(), join(ModelSavepath, "model_best_eval.pth"))
        # 保存达到预期的最好权重
        if valid_dice > best_valid_dice:
            best_epoch.append(epoch)
            best_dice.append(valid_dice)
            print("saved")
            torch.save(model.state_dict(), join(ModelSavepath, "model_best_{}.pth".format(epoch)))
        
        # log    
        log_me.info('')
        log_me.info('==================================================')
        log_me.info("eval epoch: {} and eval dice: {}".format(eval_epoch, eval_dice))
        log_me.info('')

    log_me.info("best epoch: {} and best dice: {}".format(best_epoch, best_dice))
    log_me.info("GO GO GO")
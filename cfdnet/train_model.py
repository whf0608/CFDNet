from tqdm import tqdm
import torch
import cv2
import numpy as np
from metrics.metrics import Evaluator
from torch.amp import autocast
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

def train_model(model,device=None,start_epoch=0,epochs=2,amp = True,n_classes=3, trainloader=None, valloader=None, testloader=None,optimizer=None, scheduler=None,get_loss_func=None,get_val_func=None,get_test_func=None,per_n=1,model_vis=False, update_lr_paramter=None):
    model.train()
    model = model.cuda().float()
    grad_scaler = torch.amp.GradScaler('cuda',enabled=amp)
    optimizer.zero_grad(set_to_none=True)
    if per_n==1:
        grad_steps=8
    else:
        grad_steps=1
    for epoch in range(start_epoch, epochs):
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{epochs}', unit=' img') as pbar:
            model.train()
            lss = []
            for i, images in enumerate(trainloader):
                for _ in range(per_n):
                    img = images[list(images.keys())[0]].cuda().float()
                    img = torch.nn.functional.normalize(img).contiguous()
                    with autocast('cuda',enabled=amp):
                        rs=model(img)
                        loss = get_loss_func(rs,images)
                        grad_scaler.scale(loss).backward()
                        if (i%grad_steps)==0:
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            lss.append(loss.item())
                            pbar.update(img.shape[0])
                            pbar.set_postfix(**{'loss (batch)': loss.item()})
                   
            scheduler.step()   
            if update_lr_paramter and epoch%10==0 and epoch>10:
                optimizer,scheduler=update_lr_paramter(model,epoch)
     
            if valloader:
                vallss = val_model(model,valloader=valloader,get_val_func=get_val_func,device=device,per_n=1,train_proccesing=train_proccesing,epoch=epoch)
                if update_lr_paramter:optimizer,scheduler=update_lr_paramter(model,epoch)

            pbar.set_postfix(**{'loss': sum(lss) / len(lss),'valloss': sum(vallss) / len(vallss)})
    if testloader is not None: val_model(model,testloader,get_test_func,device,per_n=1)
            
def val_model(model,valloader=None,get_val_func=None,device=None,per_n=4,epoch=0):
    model.eval()
    lss = []
    for i, images in enumerate(valloader):
        img = images[list(images.keys())[0]].to(device).float()
        img = torch.nn.functional.normalize(img).contiguous()
        with torch.no_grad():
            for _ in range(per_n):
                rs=model(img)
                loss = get_val_func(rs, images)
                lss.append(loss.item())
    return lss   

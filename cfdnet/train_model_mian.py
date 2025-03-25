import os
import sys
import json
from init import *
from train_model import train_model
from tran_process import Train_Precess
from models import get_model
from lossers import get_loss_function
from vdatasets import get_dataloader
from optimizers_schedulers import get_optimizer_scheduler, update_paramter
from config import get_model,model_dic

# config_path = r'model_config/model_vgg_unet_config.json'
# work_space = './model_test'
# flag=None

def train(show=None, args=None):

    config_path = args['config']
    work_space = args['work_dir']
    flag = args['resume_from']
    start_epoch = args['start_epoch']
    end_epoch = args['end_epoch']
    per_n = args['per_n']
    pretrain = args['pretrain']
    weights_path = args['weight_path']
    
    ###1. 配置参数
    if not Path(config_path).exists():
        print('not exists :', config_path)
        return
    cfg = json.load(open(config_path))
    if args['img_size']:
        cfg["training"]["img_size"] = (args['img_size'],args['img_size'])
    if args['batch_size']:
        cfg["training"]["batch_size"] = args['batch_size']
    if end_epoch is None:
        end_epoch = cfg["training"]["end_epoch"]
    
    cfg["rdd"]=args["rdd"]
    n_classes = cfg['data']['n_classes']
    save_path = init_file(work_space=work_space,model_name=cfg['model'],flag=flag)
    with open(save_path+'/model_config.json','w') as f:
        f.write(json.dumps(cfg))
    
    ###2. 加载数据
    trainloader, valloader = get_dataloader(cfg)
    
    ###3. 加载权重
    model_func = model_dic[cfg['model']]
    cfg['model_param'].update({"n_classes":n_classes})
    model = model_func(**cfg['model_param'])#3,n_classes
    model = torch.nn.parallel.DataParallel(model)
    print(model)
    # model = torch.nn.parallel.DataParallel(get_model(cfg['model'])(3,n_classes))
    
    if flag:
        if Path(save_path+'/log.txt').exists():
            with open(save_path+'/log.txt','r') as f:
                line = f.readlines()[-1]
                start_epoch=int(line.split(" ")[1])
                
        load_weights= save_path+'/model.pt'

        if Path(load_weights).exists():
            print('loading weights', load_weights)
            # model.load_state_dict(torch.load(load_weights, map_location='cpu'))
            model.load_state_dict(torch.load(load_weights, map_location='cpu'),strict=False)
    if pretrain:
        print('loading pretrain weights', weights_path)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    ### 4. 初始化损失和优化器
    optimizer, scheduler = get_optimizer_scheduler(model,cfg)
    initial_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) # try SGD
    #opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

    get_train_loss_func = get_loss_function(cfg["training"]["loss"])
    get_val_loss_func = get_loss_function(cfg["training"]["loss"])
    ### 5. 训练过程处理
    train_proccesing = Train_Precess(save_path=save_path, on_show=show).train_proccesing

    if cfg['rdd']:
        update_lr_paramter = update_paramter
        amp = True
    else:
        update_lr_paramter = None
        amp = True
    train_model(model, start_epoch=start_epoch, epochs=end_epoch, amp=amp, device=device,
                n_classes=n_classes, trainloader=trainloader, valloader=valloader,
                optimizer=optimizer, scheduler=scheduler, get_loss_func=get_train_loss_func,
                get_val_func=get_val_loss_func, train_proccesing=train_proccesing, per_n=per_n, update_lr_paramter=update_lr_paramter)


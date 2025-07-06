import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("../models_seg")
import json
from init import *
from train_model import train_model
from models import get_model,model_dic
from lossers import get_loss_function
from vdatasets import get_dataloader
from optimizers_schedulers import get_optimizer_scheduler, update_paramter

def train(show=None, args=None):
    config_path = args['config']

    ###1. 配置参数
    if not Path(config_path).exists():
        print('not exists :', config_path)
        return
        
    cfg = json.load(open(config_path))
    cfg["rdd"]=args["rdd"]
    n_classes = cfg['data']['n_classes']
    
    save_path = init_file(work_space=work_space,model_name=cfg['model'],flag=flag)
    with open(save_path+'/model_config.json','w') as f:
        f.write(json.dumps(cfg))
    
    ###2. 加载数据
    trainloader, valloader = get_dataloader(cfg)
    
    ###3. 加载权重
    model = torch.nn.parallel.DataParallel(model_dic[cfg['model']](**cfg['model_param']))
    print(model)

    
    if pretrain:
        print('loading pretrain weights', weights_path)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    ### 4. 初始化损失和优化器
    optimizer, scheduler = get_optimizer_scheduler(model,cfg)
    initial_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) # try SGD

    get_train_loss_func = get_loss_function(cfg["training"]["loss"])
    get_val_loss_func = get_metrics_function(cfg["val"]["metrics"])

    if cfg['rdd']:
        update_lr_paramter = update_paramter
        amp = True
    else:
        update_lr_paramter = None
        amp = True
    train_model(model, start_epoch=start_epoch, epochs=end_epoch, amp=amp, device=device,
                n_classes=n_classes, trainloader=trainloader, valloader=valloader,
                optimizer=optimizer, scheduler=scheduler, get_loss_func=get_train_loss_func,
                get_val_func=get_val_loss_func, per_n=per_n, update_lr_paramter=update_lr_paramter)


if __name__ == '__main__':
    args = parse_args()
    args = args.__dict__
    args['config'] = 'model_config/model_cfdnet_seg_dataset.json'
    args["rdd"] = True
    train(args=args, show=None)

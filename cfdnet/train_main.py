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
    cfg['model_param'].update({"n_classes":n_classes})
    model = torch.nn.parallel.DataParallel(model_dic[cfg['model']](**cfg['model_param']))
    print(model)

    
    if flag:
        if Path(save_path+'/log.txt').exists():
            with open(save_path+'/log.txt','r') as f:
                line = f.readlines()[-1]
                start_epoch=int(line.split(" ")[1])
                
        load_weights= save_path+'/model.pt'

        if Path(load_weights).exists():
            print('loading weights', load_weights)
            model.load_state_dict(torch.load(load_weights, map_location='cpu'),strict=False)
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
    # args['config'] = 'model_config/model_FDD_model_multitask_seg_dataset_imgs8.json'
    # args['config'] = './model_config/model_ccnet_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_A2FPN_seg_dataset.json'
    # args['config'] = 'model_config/model_fpn_poolformer_s12_seg_dataset.json'
    # args['config'] = 'model_config/model_danet_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_dmnet_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_vit_model_seg_dataset.json'
    # args['config'] = 'model_config/model_base_unet_segmuilt_datasetv2.json'
    # args['config'] = 'model_config/model_vgg_unet_seg_dataset_harvey.json'
    # args['config'] = 'model_config/model_UNetFormerdpn98_seg_dataset.json'
    # args['config'] = 'model_config/model_deeplabv3plus_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_fcn_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_fcn_hr18_seg_dataset.json'
    # args['config'] = 'model_config/model_fast_scnn_seg_dataset.json'
    # args['config'] = 'model_config/model_san_vit-b16_seg_dataset.json'
    # args['config'] = 'model_config/model_emanet_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_encnet_r50-d8_seg_dataset.json'
    # args['config'] = 'model_config/model_erfnet_fcn_seg_dataset.json'
    # args['config'] = 'model_config/model_vgg_unet_seg_dataset_imgs8.json'
    #args['config'] = 'model_config/model_bisenetv2_seg_dataset_imgs8.json'
    # args['config'] = 'model_config/model_UNetFormer_seg_dataset_imgs8.json'
    #args['config'] = 'model_config/model_ccnet_r50-d8_seg_dataset_imgs8.json'
    #args['config'] = 'model_config/model_deeplabv3_r50-d8_seg_dataset_imgs8.json'
    args['config'] = 'model_config/model_erfnet_fcn_seg_dataset_imgs8.json'


    args['img_size'] = None
    args['batch_size'] =None
    args["rdd"] = False
    args['resume_from'] =None
    args['per_n'] = 1
    args['weight_path'] = ""
    args['pretrain'] =False
    # args['end_epoch'] = 30
    train(args=args, show=None)

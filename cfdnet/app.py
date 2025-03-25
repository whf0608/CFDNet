import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import socks
import socket
# socks.set_default_proxy(socks.SOCKS5, "192.168.205.183", 10810)
# socket.socket = socks.socksocket

import sys
sys.path.append('/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/segment-anything/segment_anything')
sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/GeoSeg/")
sys.path.append("/home/wanghaifeng/whf_work/work_sync/models_seg")
import streamlit as st
from glob import glob
import vdatasets
import lossers
import models
import optimizers_schedulers
import json
import train_model_mian
from config import get_model,model_dic
# from get_models import model_dic_geoseg,model_dic_mmseg
# model_dic.update(model_dic_geoseg)
# model_dic.update(model_dic_mmseg)

 # streamlit run app.py  --server.fileWatcherType none
# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

ms=[]
for m in glob('model_config/model_*'):
    ms.append(m.split('/')[-1])
# print("===============",ms)
option = st.sidebar.selectbox(
        "select model",
        ms,
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled,
    )
    
vistraining_p='model_config'
empty0 =st.empty()
empty1 =st.empty()

def get_index(k,v,k_v):
    if k == 'model':
        name = k_v['model']  
    elif k == 'dataset':
        name = k_v['data']['dataset']  
    elif k == 'optimizer':
        name = k_v['training']['optimizer']["name"]
    elif k == 'scheduler' or k=='schedulers':
        name = k_v['training']['lr_schedule']["name"]
    elif k == 'loss':
        name = k_v['training']['loss']["name"]
    # print(k)
    if name in v:
            index_i = v.index(name)
    else:
            v.append(name)
            index_i = v.index(name)

    return index_i,v 

def set_config(kv,config_r,k_v):
    for k,v in  zip(kv,config_r):
        # print("============",k,v)
        if k == 'model':
            k_v['model'] = v  
        elif k == 'dataset':
            k_v['data']['dataset']  = v
        elif k == 'optimizer':
            k_v['training']['optimizer']["name"] = v
        elif k == 'scheduler' or k== 'schedulers':
            k_v['training']['lr_schedule']["name"] = v
        elif k == 'loss':
            k_v['training']['loss']["name"] = v 
    config_name = 'model_'+k_v['model']+'_'+ k_v['data']['dataset']
    return k_v, config_name


def show_config(empty0,model_config_path):
    if vistraining_p not in model_config_path:
        model_config_path = os.path.join(vistraining_p, model_config_path)
    with open(model_config_path,'r') as f:
        model_config = json.load(f)
        print(model_config)
    with empty0:
        with empty0.container():
            with open(os.path.join(vistraining_p,'vis_data.json'),'r') as f:
                k_v = json.load(f)
                
            print(k_v)
            config_r=[]
            nconfig_r=[]
            ninput_r=[]
            col = empty0.columns(1)
            ks=[]
            for k in k_v: 
                if k!='New':
                    index_i,k_v[k]=get_index(k,k_v[k], model_config)
                    config_r.append(col[0].selectbox(k,k_v[k],index_i))
                    ks.append(k)
                    # input_r.append(col2.text_input(k))
                else:
                    # col1,col2 = empty0.columns(2)
                    nconfig_r.append(col[0].text_input('new_k'))
                    ninput_r.append(col[0].text_input('new_v'))       
    return ks,config_r,nconfig_r,ninput_r                     
    
config_save = st.sidebar.button('Config')
update = st.sidebar.button('Update')

k_v0,config_r,inconfig_r,ninput_r =show_config(empty0,option)

if update:
    with open(os.path.join(vistraining_p,'vis_data.json'),'r') as f:
        k_v = json.load(f)
    k_v['dataset']= list(vdatasets.data_dic.keys())
    k_v['optimizer']= list(optimizers_schedulers.optimizer_dic.keys())
    k_v['schedulers']= list(optimizers_schedulers.scheduler_dic.keys())
    k_v['loss'] = list(lossers.loss_dic.keys())
    k_v['model'] = list(model_dic.keys())
    with open(os.path.join(vistraining_p,'vis_data.json'),'w') as f:
            f.write(json.dumps(k_v))
            
if config_save:
    with open(os.path.join(vistraining_p,option),'r') as f:
        model_config = json.load(f)
    k_v1, config_name = set_config(k_v0,config_r,model_config)
    # nconfig_r,ninput_r
    dataname=model_config['data']['path'].split('/')[-1]
    save_p = os.path.join(vistraining_p,config_name+'_'+dataname+'.json')
    print('save path: ',save_p)               
    with open(save_p,'w') as f:
        f.write(json.dumps(k_v1))
    empty0.write(json.dumps(k_v1))

train = st.sidebar.button('train')

def show_result(empty0):

    with empty0:
        with empty0.container():
            col11, col12, col13, col14 = empty0.columns(4)
            cols = [col11, col12, col13, col14]
    return cols



cache = {}
col_i = 0
def show(tag, img):
    global col_i
    if tag not in cache.keys():
        cache[tag] = cols[col_i % 4].empty()
        col_i += 1
    cache[tag].image(img, caption=tag)

if train:
    empty0.empty()
    cols = show_result(empty1)
    # model = importlib.import_module(option[:-3])
    args = {}
    args['config'] = os.path.join(vistraining_p, option)
    args['work_dir'] = 'model_works'
    args['resume_from'] =False  # "2023_06_22_23_25_30/"#False  #'2023_06_12_22_05_46'
    args['start_epoch'] = 0
    args['end_epoch'] = None
    args['img_size'] = None
    args['batch_size'] =None
    args['rdd'] = False
    args['per_n'] = 1
    args['weight_path'] = ''
    args['pretrain'] = False
    # args['config'] = 'model_works/vgg_unet/vgg_unet_2024_01_23_11_33_46/model_config.json'
    # args['weight_path'] = "model_works/vgg_unet/vgg_unet_2024_01_23_11_33_46/model_best.pt"    
    train_model_mian.train(show=show, args=args)
   

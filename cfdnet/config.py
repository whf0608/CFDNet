import sys
sys.path.append("/home/wanghaifeng/whf_work/work_sync/models_seg")
sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/GeoSeg")
sys.path.append('/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/segment-anything/segment_anything')
from argparse import ArgumentParser
from models import get_model,model_dic
from get_models import model_dic_geoseg,model_dic_mmseg
model_dic.update(model_dic_geoseg)
model_dic.update(model_dic_mmseg)
import numpy as np
import cv2
import os
from os.path import join as opjoin
import json
from pathlib import Path
import torch
from mmengine.model import revert_sync_batchnorm
sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys/lib/utils")
sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/valite_dataset_sys/")
sys.path.append("/home/wanghaifeng/whf_work/work_sync/mmsegmentation/")
from mmseg.apis import inference_model, init_model, show_result_pyplot
from result_utils import  mask2shape,  draw_svg_label
from mask_to_cls_subimgs import save_sub_img,box_expend
import torchvision.transforms.functional as transF
from PIL import Image
from metrics.evel_models import get_model_dataset

class Model:
    def __init__(self,checkpoint='test',
                 model_name = 'fpn_poolformer_s12',
                 config= 'test',
                 save_img=True, save_submask_cache=False,save_svg_cache=True, save_sub_mask=False,rdp_use=False,**arg):

        self.save_img = save_img
        self.save_submask_cache = save_submask_cache
        self.save_svg_cache = save_svg_cache
        self.save_sub_mask = save_sub_mask
        self.rdp_use= rdp_use
        
        if Path(config).exists():
            if Path(model_config_path).suffix == '.json':
                model, model_name, dataset_name_train = get_model_dataset(config)
            if Path(model_config_path).suffix == '.py':
                model = init_model(config, checkpoint)
                model = revert_sync_batchnorm(model)
        else:
            model = model_dic[model_name](**arg)
            model = torch.nn.parallel.DataParallel(model)
            if Path(checkpoint).exists():
                model.load_state_dict(torch.load(checkpoint, map_location='cpu'),strict=True)
                
        self.model = model.cuda()
        self.model_name = 'cfd_'+model_name
        print("loading "+self.model_name)
    
    def get_model(self):
        return self.model

    def running(self, img, save_file='./', dealresult=False,useargmax=False,**arg):
        img_ =transF.to_tensor(img.copy()).cuda()
        imgs = img_.unsqueeze(0)
        with torch.no_grad():
            masks =  self.model(imgs,**arg)
        if useargmax:
            masks = torch.argmax(masks,1)
            masks = masks.cpu().numpy()
        if dealresult:
            imgs_meta = self.deal_result(img=img, masks=masks, save_file= save_file)
            return imgs_meta
        else:
            return masks
    
    def training(self, img, save_file='./', dealresult=False,useargmax=False,**arg):
        # img_ =transF.to_tensor(img.copy()).cuda()
        # imgs = img_.unsqueeze(0)
        # with torch.no_grad():
        result =  self.model(img,**arg)
        # if useargmax:
        #     masks = torch.argmax(masks,1)
        #     masks = masks.cpu().numpy()
        # if dealresult:
        #     imgs_meta = self.deal_result(img=img, masks=masks, save_file= save_file)
        #     return imgs_meta
        # else:
        #     return masks
        return result
    
    def deal_result(self,img,masks, save_file):
        
        return deal_result(img,masks,save_file,self.model,self.save_img,self.save_svg_cache,self.save_sub_mask,self.rdp_use)
     
def deal_result(img,masks,save_file,model,save_img,save_svg_cache,save_sub_mask,rdp_use):    
        pre_cls = {}
        if type(model)==torch.nn.parallel.DataParallel:
            classes_n = model.module.n_classes
        else:
            classes_n = model.n_classes
            
        pointss=[]
        for mask in masks[0:]:
            for cls_index in  range(classes_n):
                mask_cls = mask==cls_index
                mask_cls = np.array(mask_cls*255,np.uint8)
                pointss_ = mask2shape(mask_cls ,rdp_use)
                for _ in range(len(pointss_)):
                    pre_cls[str(_)] = cls_index
                
                pointss.extend(pointss_)
                

        os.makedirs(save_file, exist_ok=True)
        img_p = opjoin(save_file, 'sum.png')
        if save_img:
            img_max = img.max()
            if img_max<50:
                img = np.array(img*255/img_max,np.uint8)
            cv2.imwrite(img_p, img)
    
        svg_path = opjoin(save_file, 'sum.svg')
        if save_svg_cache:
            svg_path = draw_svg_label(save_path=save_file, size=img.shape[:2], pointss=pointss)
            if Path(svg_path).exists():
                with open(svg_path, 'r') as f:
                    svg_data = f.read()
                

        if Path(svg_path.replace('.svg', '.json')).exists():
            with open(svg_path.replace('.svg', '.json'), 'r') as f:
                label_cls = json.load(f)
        else:
            label_cls = {}

        imgs_meta = {}
        imgs_meta['svg_data'] = svg_data
        imgs_meta['svg_path'] = svg_path
        imgs_meta['img_p'] = img_p
        imgs_meta['label_pre'] = {}
        imgs_meta['pre_cls'] = pre_cls
        imgs_meta['label_cls'] = label_cls
        imgs_meta['pointss'] = pointss
        return imgs_meta
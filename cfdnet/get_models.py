import sys
import os
from glob import glob
# try:
if True:
    sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/GeoSeg/")
    # from geoseg.models.BuildFormer import BuildFormerSegDP
    from geoseg.models.A2FPN import A2FPN
    from geoseg.models.BANet import BANet
    from geoseg.models.MANet import MANet
    from geoseg.models.FTUNetFormer import FTUNetFormer
    from geoseg.models.UNetFormer import UNetFormer
    from geoseg.models.ABCNet import ABCNet
    from geoseg.models.DCSwin import DCSwin
    model_dic_geoseg = {
             'ABCNet': ABCNet,
             'A2FPN': A2FPN,
             'BANet': BANet,
             'MANet': MANet,
             'FTUNetFormer': FTUNetFormer,
             'UNetFormer': UNetFormer,
                'DCSwin':DCSwin
             }
# except:
#     model_dic_geoseg={}
#     print("no load geoseg")
import timm
model_names = timm.list_models() 
class UNetFormerBackbone:
    def __init__(self,backbone_name='swsl_resnet18'):
        self.backbone_name=backbone_name
    
    def get_model_backbone(self,**arg):
        if 'backbone_name' not in arg.keys():
            arg['backbone_name']=self.backbone_name
        else:
            self.backbone_name = arg['backbone_name']
        print("============================================")
        print("loading backbone: ",self.backbone_name, arg)
        return UNetFormer(**arg)
    
for model_name in model_names:
    model_dic_geoseg['UNetFormer'+model_name] = UNetFormerBackbone(backbone_name=model_name).get_model_backbone
    
# try:
if True:
    sys.path.append("/home/wanghaifeng/whf_work/libs/mmclassification")
    sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/mmsegmentation")
    sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/mmpretrain")
    from mmseg.apis import inference_model, init_model, show_result_pyplot
    from mmengine.model import revert_sync_batchnorm
    class Model_C:
        def __init__(self,model_config):
            self.model_config = model_config
        def get_model(self,dim=3,n_classes=3,**arg):
            print('mmseg model config',self.model_config)
            os.system("cat "+self.model_config)
            model = init_model(config=self.model_config)
            model  = revert_sync_batchnorm(model)
            load_weights = glob("/home/wanghaifeng/whf_work/models_weights/mmseg/"+self.model_config.split('/')[-1][:-3]+'*.pth')
            if len(load_weights)>0 and Path(load_weights[0]).exists():
                print("loading weight: ", load_weights[0])            
                model.load_state_dict(torch.load(load_weights[0], map_location='cpu'))
            return model
        
    model_configs = glob('/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/mmsegmentation/configs/_base_/models/*.py')
    model_dic_mmseg = {}
    for model_config in model_configs:
        model_dic_mmseg[model_config.split('/')[-1][:-3]] = Model_C(model_config).get_model
        
    model_configs = glob('/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/mmsegmentation/configs/*/*.py')
    
    for model_config in model_configs:
        if "_base_" == model_config.split('/')[-2]: continue
        # print("============", model_config.split('/')[-1][:-3])
        model_dic_mmseg[model_config.split('/')[-1][:-3]] = Model_C(model_config).get_model
    # print(model_dic_mmseg)
        
# except:
#     model_dic_mmseg = {}
#     print("no load mmseg")
    
try:
    sys.path.append('/home/wanghaifeng/whf_work/work_sync/vit/vit-pytorch/vit_pytorch')
    # glob('/home/wanghaifeng/whf_work/vit/vit-pytorch/vit_pytorch/*')
except:
    pass
    
def test_all_models(model_dic=None):
    try:
        model_dic[model_name](None)
        print('ok: ',model_name)
    except:
        print('erro: ',model_name)
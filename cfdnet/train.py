import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('../segmention_buildings/BuildFormer')
sys.path.append('/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/segment-anything/segment_anything')
sys.path.append("/home/wanghaifeng/whf_work/work_sync/satellite_data/models_config/GeoSeg/")
sys.path.append("/home/wanghaifeng/whf_work/work_sync/models_seg")
from init import parse_args
from train_model_mian import train


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

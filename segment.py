import argparse
import os
import time

from pathlib import Path

import torch
import torch.nn as nn
import cv2
import yaml
from torchvision import transforms

import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf, increment_path
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

from models.experimental import Ensemble

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
from models.common import Conv

def onImage():

    image = cv2.imread(opt.source)
    assert image is not None, 'Image Not Found' + opt.source

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_size = image.shape

    image_display = process_frame(image)
    image_display = cv2.resize(image_display, (img_size[1], img_size[0]))

    if not opt.nosave:
        cv2.imwrite(save_path, image_display)
        print("Output saved: ", save_path)
        
    if opt.view_img:
        cv2.imshow("Result", image_display)
        cv2.waitKey(0)

def process_frame(image):

    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    image = image.to(device)
    image = image.half() if half else image.float()
    
    output = model(image)
    
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    
    bases = torch.cat([bases, sem_output], dim=1)
    
    nb, _, height, width = image.shape
    
    names = model.names
    
    pooler = ROIPooler(output_size = hyp['mask_resolution'], scales=[model.pooler_scale], sampling_ratio= 1, pooler_type= 'ROIAlignV2',
                       canonical_level= 2)
    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(
        prediction=inf_out, attn=attn, bases=bases, pooler=pooler, hyp=hyp, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, 
        merge=False, mask_iou=None
    )
    
    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    
    if pred is not None:
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)

        image_display = image[0].permute(1,2,0)*255
        image_display = image_display.cpu().numpy().astype(np.uint8)
        
        image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
        
        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if conf < opt.conf_thres:
                continue
            color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
            
            image_display[one_mask] = image_display[one_mask] * 0.5 + np.array(color, dtype= np.uint8) * 0.5
            label = '%s %.3f' % (names[int(cls)], conf)
            tf = max(opt.thickness-1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=opt.thickness/3, thickness= tf)[0]
            c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3

            if not opt.no_bbox:
                cv2.rectangle(image_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=opt.thickness, lineType=cv2.LINE_AA)
                cv2.putText(image_display, label, (bbox[0], bbox[1]-2), 0, opt.thickness/3, [255,255,255], thickness=tf, lineType=cv2.LINE_AA)

        return image_display
    return image_
    
def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-mask.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images/ffbf32e12b7d53aa6d7ceeaca94e1ab9706800b2cef9bbdc22fbbbcf828f64fa.png', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.mask.yaml', help='hyperparamete file for yolov7 mask')  # file/folder, 0 for webcam
    parser.add_argument('--seed', type=int, default=1, help='random seed to change color')
    parser.add_argument('--thickness', type=int, default=1, help='bounding boxes thickness')
    parser.add_argument('--no-bbox', action='store_true', help='display results')
    parser.add_argument('--no-label', action='store_true', help='display results')
    parser.add_argument('--show-fpx', action='store_true', help='display results')

    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    np.random.seed(opt.seed)
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    half = device.type != "cpu"

    weights = torch.load(opt.weights, map_location=device)
    model = attempt_load(opt.weights, map_location=device)
    # model = weights['model'].to(device)

    if half:
        model = model.half()
    
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    if not opt.nosave:
        save_dir= Path(increment_path(Path(opt.project) / opt.name, exist_ok=False)) # increment runs
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / os.path.basename(opt.source))

    webcam = opt.source.isnumeric()
    if webcam and not opt.nosave:
        save_path = save_path + ".mp4"
    img_formats = ["bmp", "jpg", "jpeg", "png"]
    vid_formats = ["mov", "avi", "mp4"]

    with torch.no_grad():
        if opt.source.split('.')[-1].lower() in img_formats:  # update all models (to fix SourceChangeWarning)
            onImage()
        elif opt.source.split('.')[-1].lower() in vid_formats or webcam:
            onVideo()
        else:
            print("invlaid source")

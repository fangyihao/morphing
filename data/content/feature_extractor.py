'''
Created on Oct. 31, 2022

@author: Yihao Fang
'''

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.tools.demo import Predictor
from yolox.exp.build import get_exp
from yolox.data.datasets import COCO_CLASSES
import torch
import os
import cv2
import pandas as pd
import codecs
import numpy as np

from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
import matplotlib.pyplot as plt
from colorthief import ColorThief
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
import io
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from ordered_set import OrderedSet
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def segment_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = pil_to_tensor(image)
    
    #print(image_tensor.shape)

    fcn_resnet = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    fcn_resnet.eval()
    preprocess_img = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms(resize_size=None)
    
    prediction = fcn_resnet(preprocess_img(image_tensor).unsqueeze(dim=0))

    class_to_idx = {cls: idx for (idx, cls) in enumerate(FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"])}
    
    #print(class_to_idx)

    prediction = prediction['out']
    normalized_masks = prediction.softmax(dim=1)[0]
    
    #to_pil_image(normalized_masks[class_to_idx['person']]).show()
    to_pil_image(normalized_masks[class_to_idx['__background__']])
    
    masks = normalized_masks > 0.7
    
    out = draw_segmentation_masks(image_tensor, masks[class_to_idx['person']], alpha=1)
    
    to_pil_image(out).show()

    background = draw_segmentation_masks(image_tensor, masks[class_to_idx['__background__']], alpha=1)
    
    to_pil_image(background).show()
    
    return normalized_masks


def get_yolox_predictor():
    exp = get_exp(None, 'yolox-x')
    

    exp.test_conf = 0.25
    
    exp.nmsthre = 0.45
    
    exp.test_size = (640, 640)

    model = exp.get_model()
    model.cuda()

    model.eval()

    ckpt_file = '../../../YOLOX/yolox_x.pth'
        
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder, 'gpu'
    )
    return predictor
    
    
def object2text(output, img_info, confthre):
    ratio = img_info["ratio"]
                
    output = output.cpu()

    boxes = output[:, 0:4]

    boxes /= ratio

    cls_ids = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    
    #texts = ['Image width is {:d} and image height is {:d}'.format(img_info['width'], img_info['height'])]
    #texts = ['{:d} {:d}'.format(img_info['width'], img_info['height'])]
    texts = []
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < confthre:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        #text = 'Detection box class is {}, detection box score is {:.1f}%, detection box width is {:d}, detection box height is {:d}, and detection box center is ({:d},{:d})'.format(COCO_CLASSES[cls_id], score * 100, x1-x0, y1-y0, (x0+x1)//2, (y0+y1)//2)
        #text = '{} {:.1f}% {:d} {:d} ({:d},{:d})'.format(COCO_CLASSES[cls_id], score * 100, x1-x0, y1-y0, (x0+x1)//2, (y0+y1)//2)
        text = '{} {:.1f}%'.format(COCO_CLASSES[cls_id], score * 100)
        texts.append(text)
    
    #text = ' '.join(texts)
    text = '\n'.join(texts)
    
    return text

def rgb2name(rgb_tuple):
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'

def extract_image_features():
    
    yolox_predictor = get_yolox_predictor()
    img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en')
    
    def detect_objects(image_path):
        outputs, img_info = yolox_predictor.inference(image_path)
        
        text = object2text(outputs[0], img_info, yolox_predictor.confthre)

        result_image = yolox_predictor.visual(outputs[0], img_info, yolox_predictor.confthre)
        
        save_folder = os.path.join(bu_processed_path, path)
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(save_folder, image_name)
        cv2.imwrite(save_file_name, result_image)
        return text
    

    def extract_dominant_colors(image_path):
        '''
        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        cv2.imshow('image', image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        data = np.reshape(image, (-1,3))
        data = np.float32(data)
    
        num_clusters = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(data,num_clusters,None,criteria,10,flags)
    
        dominant_colors = [center/255 for center in centers]
        plt.imshow([dominant_colors])
        plt.show()
        
        centers = [center.astype(np.int32) for center in centers]
        print(centers)
        print([rgb2name(center) for center in centers])
        '''
        '''
        b = io.BytesIO()
        image.save(b, "JPEG")
        b.seek(0)
        '''
        ct = ColorThief(image_path)
        palette = ct.get_palette(5)
        return '\n'.join(OrderedSet([rgb2name(palette[i]) for i in range(5)]))
        #plt.imshow([[palette[i] for i in range(5)]])
        #plt.show()
    
    def generate_caption(image_path):
        result = img_captioning(image_path)
        print(result[OutputKeys.CAPTION]) 
        return result[OutputKeys.CAPTION][0]
    
    

    for business_unit in ['iTrade', 'Wealth']:
        records = []
        bu_raw_path = 'raw/%s'%business_unit
        bu_processed_path = 'processed/%s'%business_unit
        for path in os.listdir(bu_raw_path):
            # check if current path is a file
            html_src_path = os.path.join(bu_raw_path, path)
            if not os.path.isfile(html_src_path):
                for image_name in os.listdir(html_src_path):
                    image_path = os.path.join(html_src_path, image_name)
                    
                    #height, width, channels = cv2.imread(image_path).shape
                    
                    objects = detect_objects(image_path)
                    colors = extract_dominant_colors(image_path)
                    caption = generate_caption(image_path)

                    records.append({'Image': image_path, 'Object':objects, 'Color':colors, 'Caption':caption})

        
        df = pd.DataFrame.from_records(records)
        with codecs.open('processed/%s_image_features.csv'%business_unit, 'w', 'utf-8') as csv_file:
            df.to_csv(csv_file, index=False, lineterminator='\n')
    


extract_image_features()



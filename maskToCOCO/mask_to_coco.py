import numpy as np
import os
import sys
from PIL import Image, ImageOps
import cv2
from coco_format import MasktoCOCO
from pycocotools import mask
from skimage import measure
import argparse
from glob import glob
import json
from imantics import Polygons, Mask
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, default=None)
    parser.add_argument('--mask_path', type=str, required=True, default=None)
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--img_height', type=int, default=1080)
    parser.add_argument('--mask_extensions', type=str, default='png')
    return parser.parse_args(argv)
def make_annotation(img_idx, mask_idx, arr, cls):
    fortran = np.asfortranarray(arr)
    encoded = mask.encode(fortran)
    area = mask.area(encoded)
    bb = mask.toBbox(encoded)
    contours = measure.find_contours(arr, 0.5)
    anno = {
        "segmentation": [],
        "area" : float(area.tolist()),
        "iscrowd" : 0,
        "image_id" : int(img_idx),
        "bbox" : bb.tolist(),
        "category_id" : cls,
        "id" : int(str(img_idx)+str(mask_idx))
    }
    for contour in contours:
        ct = np.flip(contour, axis=1)
        segmentation = ct.ravel().tolist()
        anno["segmentation"].append(segmentation)
    return anno
def main(args):
    ### 1. Initial Settings
    # 1.1. Define coco format
    mask2coco_maker = MasktoCOCO(args.img_width, args.img_height)
    img_path = os.path.abspath(args.img_path)
    mask_path = os.path.abspath(args.mask_path)
    # 1.2. set the save path & make the folder
    save_path = args.output_path
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)
    ### 2. Loop for image 
    for root,dirs,files in os.walk(img_path):
        for i, file in enumerate(files):
            mask2coco_maker.image_section = {}
            file_path = os.path.join(img_path, file)
            save_file_path = os.path.join(save_path, file)
            # save image to output path
            img = cv2.imread(file_path)
            cv2.imwrite(save_file_path, img)
            # insert data to image section
            mask2coco_maker.image_section["licenses"] = 0
            mask2coco_maker.image_section["url"] = None
            mask2coco_maker.image_section["file_name"] = save_file_path
            mask2coco_maker.image_section["height"] = args.img_height
            mask2coco_maker.image_section["width"] = args.img_width
            mask2coco_maker.image_section["id"] = int(i)
            mask2coco_maker.coco["images"].append(mask2coco_maker.image_section)
            # find pair masks
            base_name = file.split('.')[0]
            mask_list = glob(os.path.join(mask_path, base_name) + '*.{}'.format(args.mask_extensions))
            for j, mask in enumerate(mask_list):
                mask2coco_maker.annotation_section = {}
                mask_class = mask.split('_')[-1]
                mask_class = mask_class.split('.')[1]
                category_id = mask2coco_maker.get_category_id(mask_class)
                # Loading mask image.
                mask_img = Image.open(mask)
                mask_img = mask_img.resize((args.img_width,args.img_height), Image.ANTIALIAS)
                mask_invert = np.array(ImageOps.invert(mask_img))
                # binary mask to COCO
                anno = make_annotation(int(i), j, mask_invert, category_id)  
                mask2coco_maker.annotation_section = anno
                mask2coco_maker.coco["annotations"].append(mask2coco_maker.annotation_section)
    # save json
    with open('{}/annotations.json'.format(save_path), 'w', encoding='utf-8') as make_file:
        json.dump(dict(mask2coco_maker.coco), make_file, indent="\t")
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
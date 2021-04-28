import os 
import numpy as np
class MasktoCOCO():
    def __init__(self, args, description=None):
        self.args = args
        self.coco = {
            "info" : {
                "description" : "custom COCO dataset",
                "url" : "None",
                "version" : "0.1",
                "year" : 2020,
                "contributor" : "Sungman Cho",
                "date_created" : "2020/05"
            },
            "licenses" : [{
                    "url" : "",
                    "id" : 0,
                    "name": "No license"
            }],
            "images" : [],
            "annotations": [],
            "categories" : [
                {"supercategory": "null", "id": 0, "name": "_background_"},
                {"supercategory": "class type1", "id": 1, "name": "Elephant"},
                {"supercategory": "class type1", "id": 2, "name": "Tiger"},
                {"supercategory": "class type2", "id": 3, "name": "Watermelon"},
            ]
        }
        self.image_section = {
            "width" : self.args.img_width,
            "height" : self.args.img_height,
        }
    def get_category_id(self, label):
        res = None
        for category in self.coco["categories"]:
            if category["name"] == label:
                res = category["id"]
                break
        return int(res)
#### TEST
if __name__ == '__main__':
    mask_to_coco = MasktoCOCO()
    print(mask_to_coco.image_section)
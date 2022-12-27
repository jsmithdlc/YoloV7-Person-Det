import json
import pandas as pd


def __load_ann_file(ann_path):
    with open(ann_path) as file:
        data = json.load(file)
    return data


def load_img_info(ann_path, split, rm_extension=False):
    anns = __load_ann_file(ann_path)
    df = pd.DataFrame(anns["images"])
    df["split"] = split
    df = df.rename(columns={"id": "image_id"})
    df["file_name"] = df["file_name"]
    if rm_extension:
        df["file_name"] = df["file_name"].str.split(".").str[0]
    return df[["file_name", "height", "width", "image_id", "split"]]


def load_ann_info(ann_path, split):
    anns = __load_ann_file(ann_path)
    df = pd.DataFrame(anns["annotations"])
    df["split"] = split
    df = df.rename(columns={"id": "annotation_id"})
    return df


def coco_bbox_to_yolo(img_width, img_height, coco_bbox, class_id=0):
    bbox_x1 = coco_bbox[0]
    bbox_y1 = coco_bbox[1]
    bbox_width = coco_bbox[2]
    bbox_height = coco_bbox[3]
    width_norm = bbox_width / img_width
    height_norm = bbox_height / img_height
    x_centre_norm = (bbox_x1 + bbox_width / 2) / img_width
    y_centre_norm = (bbox_y1 + bbox_height / 2) / img_height
    return [class_id, x_centre_norm, y_centre_norm, width_norm, height_norm]

import json
import os

from PIL import Image  # To get image dimensions


def convert_yolo_to_coco(dataset_dir, output_json_path):
    """
    Converts YOLO annotations to COCO format.

    Parameters:
    - dataset_dir: Path to the dataset directory. This directory should contain
      two subdirectories: 'images' and 'labels'.
    - output_json_path: Path where the output COCO-formatted JSON file
      will be saved.
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "windmill"}],
    }

    anno_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    filename_to_id = {}

    for idx, annotation_file in enumerate(anno_files):
        image_file = annotation_file.replace(
            ".txt", ".jpg"
        )  # Adjust the extension based on your dataset
        image_path = os.path.join(images_dir, image_file)
        image_id = idx + 1

        filename_to_id[annotation_file] = image_id

        with Image.open(image_path) as img:
            width, height = img.size

        coco_data["images"].append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_file,
            }
        )

        with open(os.path.join(labels_dir, annotation_file), "r") as file:
            for idx, line in enumerate(file.readlines()):
                (
                    class_id,
                    x_center_norm,
                    y_center_norm,
                    width_norm,
                    height_norm,
                ) = map(float, line.split())
                x_min = (x_center_norm - width_norm / 2) * width
                y_min = (y_center_norm - height_norm / 2) * height
                bbox_width = width_norm * width
                bbox_height = height_norm * height

                coco_data["annotations"].append(
                    {
                        "id": len(coco_data["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": 1,  # Assuming a single category
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                    }
                )

    # Save the COCO data to a file
    with open(output_json_path, "w") as json_file:
        json.dump(coco_data, json_file)


# Example usage
# dataset_directory = "/path/to/your/dataset"
# output_json_file = "/path/to/save/coco_annotations.json"
# convert_yolo_to_coco(dataset_directory, output_json_file)

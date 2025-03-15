# image_process.py
import argparse
import os
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import numpy as np
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

def apply_detection_segment(args, input_folder, yolo_output_folder):
    input_files = os.listdir(input_folder)
    
    if not os.path.exists(yolo_output_folder):
        os.makedirs(yolo_output_folder)

    for input_file in input_files:
        if input_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_folder, input_file)
            image = cv2.imread(image_path)
            yolo_model_path = args.yolo_model
            model = YOLO(yolo_model_path)
            sam_checkpoint_path = args.sam_checkpoint_path
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
            predictor = SamPredictor(sam)

            # Use the YOLO model to predict object bounding boxes in the image
            results = model.predict(source=image_path, conf=0.25, max_det=1)

            # Check if there are any detection results
            if results is not None and len(results) > 0:
                # Extract bounding box coordinates only if there are detections
                for result in results:
                    boxes = result.boxes

                    # Check if there are any boxes before accessing xyxy
                    if len(boxes) > 0:
                        bbox = boxes.xyxy.tolist()[0]
                        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                        predictor.set_image(image)
                        input_box = np.array(bbox)
                        masks, _, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )
                        segmentation_mask = masks[0]
                        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
                        white_background = np.ones_like(image) * 255
                        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
                        output_path = os.path.join(yolo_output_folder, f"{input_file.split('.')[0]}_segmented.jpg")
                        plt.imsave(output_path, new_image.astype(np.uint8))
                        print(f"Segmented image saved at: {output_path}")
            else:
                # No objects detected, move to the next image
                print(f"No objects detected in {input_file}. Skipping to the next image.")



def apply_super_resolution(args, input_folder, output_folder):
    device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights(args.real_esrgan_weights, download=True)

    #Ensuring Directory exist for Upscaled output output
    os.makedirs(output_folder, exist_ok=True)

    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for filename in input_files:
        input_path = os.path.join(input_folder, filename)
        image = Image.open(input_path).convert('RGB')
        sr_image = model.predict(image)

        # Save the super-resolved image with a modified filename
        output_filename = f"sr_{os.path.splitext(filename)[0]}_upscaled.jpg"
        output_path = os.path.join(output_folder, output_filename)
        sr_image.save(output_path)

        print(f"Super-resolved image saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Image generation and enhancement script")
    parser.add_argument("--queries_file", type=str, default="./queries.txt", help="Path to the file containing queries")
    parser.add_argument("--input_base_folder", type=str, default="./scraper_output", help="Base folder where query folders are located from scraper")
    parser.add_argument("--output_base_folder", type=str, default="./upscaled_output", help="Base folder to save the upscaled images from esrgan")
    parser.add_argument("--yolo_folder", type=str, default="./segmented_output", help="Base folder to save the segmented images from yolo")
    parser.add_argument("--yolo_model", type=str, default="./best.pt", help="Path to YOLO model")
    parser.add_argument("--real_esrgan_weights", type=str, default="./RealESRGAN_x4.pth", help="Path to RealESRGAN weights")
    parser.add_argument("--sam_checkpoint_path", type=str, default="./sam_vit_h_4b8939.pth", help="Path to the SAM model checkpoint file")

    args = parser.parse_args()


    with open(args.queries_file, "r") as file:
        queries = file.read().splitlines()

    for query in queries:
        input_folder = os.path.join(args.input_base_folder, query.replace(' ', '_'))
        print("Input Folder", input_folder)
        yolo_output_folder = os.path.join(args.yolo_folder, query.replace(' ', '_'))
        print("Yolo Output Folder", yolo_output_folder)
        output_folder = os.path.join(args.output_base_folder, query.replace(' ', '_'))
        print("Output Folder", output_folder)

        apply_detection_segment(args, input_folder, yolo_output_folder)
        apply_super_resolution(args, yolo_output_folder, output_folder)

if __name__ == "__main__":
    main()

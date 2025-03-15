import shutil
import argparse
import os
import argparse
import subprocess
import os
import torch
from torch import autocast
from VictorAI import StableVictorPipeline, DDIMScheduler, AutoPipelineForImage2Image
from PIL import Image
from RealESRGAN import RealESRGAN
import numpy as np
import cv2
import time
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
#added now
import ultralytics
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from IPython.display import display, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
ultralytics.checks()
from PIL import Image



# preprocess prompt function
shoe_brands = [
    "Nike", "Adidas", "Puma", "Reebok", "New Balance", "Vans", "Converse", "Under Armour", "Asics", "Skechers",
    "Fila", "Crocs", "Clarks", "Dr. Martens", "Timberland", "Salomon", "Birkenstock", "Gucci", "Prada", "Balenciaga",
    "Jordan", "Lacoste", "Hoka One One", "Merrell", "Ecco", "Brooks", "Mizuno", "Fendi", "Versace", "Givenchy",
    "Yeezy", "Common Projects", "Rick Owens", "Alexander McQueen", "Saint Laurent", "Bottega Veneta", "Vetements",
    "Balmain", "Valentino", "Dior", "Louboutin", "Hermes", "Jimmy Choo", "Fendi", "Off-White", "Golden Goose",
    "Prada", "Bally", "Tod's", "Stella McCartney", "Marni", "Acne Studios", "Isabel Marant", "Miu Miu", "Saucony",
    "Onitsuka Tiger", "Etnies", "DC Shoes", "K-Swiss", "Frye", "UGG", "Sorel", "Keen", "Columbia", "Sperry",
    "Tommy Hilfiger", "Calvin Klein", "Steve Madden", "Guess", "Michael Kors", "Cole Haan", "ALDO", "Nine West",
    "Jessica Simpson", "Saucony", "Camper", "Superga", "Geox", "Teva", "Sanuk", "Hush Puppies", "Bogs", "The North Face",
    "Caterpillar", "Carhartt", "Levi's", "Fjällräven", "Keds", "Roxy", "Quiksilver", "Merrell", "Saucony",
    "Saucony Originals", "New Balance Classics", "On", "Arc'teryx", "Saucony Originals", "Forsake", "Diadora",
    "Altra", "Topo Athletic", "Oboz", "Chaco", "Salewa", "La Sportiva", "OluKai", "Toms", "Bucketfeet", "Reef",
    "Oofos", "Alegria", "Aetrex", "Naot", "Taos Footwear", "Birkenstock", "Camper", "ECCO", "Finn Comfort",
    "Keen", "Mephisto", "Rockport", "Dansko", "Earth Shoes", "Jambu", "Bzees", "Blundstone", "Cobb Hill", "Clarks",
    "Naturalizer", "Skechers", "Hoka One One", "Brooks", "ASICS", "Mizuno", "Merrell", "Salomon", "La Sportiva",
    "Inov-8", "Under Armour", "Reebok", "Puma", "Adidas", "Nike", "Converse", "Vans", "Dr. Martens", "Timberland",
    "UGG", "Birkenstock", "Crocs", "Sorel", "Keen", "Columbia", "The North Face", "Merrell", "Ecco", "Clarks",
    "Steve Madden", "Aldo", "Guess", "Michael Kors", "Cole Haan", "Fila", "Skechers", "Puma", "New Balance",
    "Reebok", "Under Armour", "Asics", "Adidas", "Nike", "Converse", "Vans", "Puma", "Reebok", "New Balance", "Skechers",
    "Fila", "Adidas Originals", "Puma Originals", "Nike SB", "Vans Vault", "Converse x Comme des Garçons", "Y-3",
    "Stella McCartney x Adidas", "Alexander McQueen x Puma", "Gosha Rubchinskiy x Adidas", "Raf Simons x Adidas",
    "Balenciaga Triple S", "Gucci Ace", "Yeezy Boost", "Common Projects Achilles", "Rick Owens x Adidas",
    "Alexander McQueen Oversized", "Saint Laurent SL/10H", "Bottega Veneta Dodger", "Vetements x Reebok",
    "Balmain Cameron", "Valentino Garavani Rockstud", "Dior B23", "Christian Louboutin Louis", "Hermes Oran",
    "Jimmy Choo Diamond", "Fendi Rockoko", "Off-White Off-Court", "Golden Goose Superstar", "Prada Cloudbust",
    "Bally Champion", "Tod's Gommino", "Acne Studios Manhattan", "Isabel Marant Bryce", "Miu Miu Logo",
    "Saucony Jazz Original", "Onitsuka Tiger Mexico 66", "Etnies Jameson 2", "DC Shoes Pure", "K-Swiss Classic",
    "Frye Melissa", "UGG Classic", "Sorel Caribou", "Keen Targhee", "Columbia Newton Ridge", "Sperry Authentic",
    "Tommy Hilfiger Harlow", "Calvin Klein Maya", "Steve Madden Cliff", "Guess Luiss", "Michael Kors Irving",
    "Cole Haan GrandPro", "ALDO Brilasen", "Nine West Zofee", "Jessica Simpson Mandalaye", "Saucony Jazz Original",
    "Camper Peu Cami", "Superga Cotu", "Geox Nebula", "Teva Original Universal", "Sanuk Yoga Sling", "Hush Puppies Chowchow",
    "Bogs Classic", "The North Face Chilkat", "Caterpillar Second Shift", "Carhartt 6-Inch", "Levi's Jeffrey",
    "Fjällräven Kånken", "Keds Champion", "Roxy Bayshore", "Quiksilver Carver", "Merrell Jungle Moc", "Saucony Jazz Original",
    "Saucony Originals Jazz Low Pro", "New Balance Classics 990v5", "On Cloud", "Arc'teryx Acrux", "Saucony Originals Bullet",
    "Forsake Clyde", "Diadora N902", "Altra Lone Peak", "Topo Athletic Terraventure", "Oboz Sawtooth II", "Chaco ZX/2 Classic",
    "Salewa Wildfire", "La Sportiva Bushido II", "OluKai Nohea Moku", "Toms Alpargata", "Bucketfeet Pineappleade",
    "Reef Fanning"]


# Preprocess input text prompt
def preprocess_prompt(prompt, shoe_brands):
    for brand in shoe_brands:
        # Replace both uppercase and lowercase variations of the brand name with a single space
        prompt = prompt.replace(brand, " ").replace(brand.lower(), " ").replace(brand.upper(), " ")
    # Remove any extra spaces resulting from the replacements
    prompt = ' '.join(prompt.split())
    return prompt    


# Text to image generation function
def generate_text_to_image(args):

    # preprocess input prompt
    prompt = preprocess_prompt(args.prompt, shoe_brands)

    model_path = args.model_path
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pipe = StableVictorPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = torch.Generator(device='cuda')
    seed = -1
    g_cuda.manual_seed(seed)
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=g_cuda
        ).images
    for i, img in enumerate(images):
        img.save(os.path.join(output_folder, f"generated_image_{i}.png"))


# Image to Image function function
def generate_image_to_image(args):


    # preprocess input prompt
    prompt = preprocess_prompt(args.prompt, shoe_brands)

    # Load the pre-trained model
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipeline.enable_model_cpu_offload()

    # Load the input image
    init_image = Image.open(args.image_path).convert("RGB")

    # Generate the image based on user prompts and additional parameters
    generated_image = pipeline(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        guidance_scale=args.guidance_scale,
        strength=args.strength
    ).images[0]

    
    output_folder = args.output_folder
    # make output folder directory if doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the generated image
    generated_image.save(os.path.join(output_folder, "generated_image_0.png"))


# Detection and Segmentataion function
def apply_detection_segment(args):

    input_files = os.listdir(args.output_folder)
    for input_file in input_files:
      if input_file.lower().endswith(('png','jpg','jpeg')):
        image_path = os.path.join(args.output_folder, input_file)

    print("Image_path", image_path)
    image = cv2.imread(image_path)

    
    model = YOLO(args.yolo_model)
    
    sam_checkpoint = args.sam_checkpoint_path
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)   


    results = model.predict(source=image_path, conf=0.25,max_det=1)
    if results != None:

        for result in results:
            boxes = result.boxes

        bbox=boxes.xyxy.tolist()[0]

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
        output_path = os.path.join(args.project_name, "segmented_output")
        print("output_path segmented", output_path)
        os.makedirs(output_path, exist_ok=True)

        # Save the image using plt.imsave in the specified directory
        output_file_path = os.path.join(output_path, "segmented_output.jpg")
        plt.imsave(output_file_path, new_image.astype(np.uint8))

        # Display the saved image path
        print(f"Image saved at: {output_file_path}")


        return output_path


# Apply Super Resolution  to image function 
def apply_super_resolution(args, input_folder):
    device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights(args.real_esrgan_weights, download=True)
    output_folder = os.path.join(args.project_name, "resized_images")
    os.makedirs(output_folder, exist_ok=True)
    print("apply super resolution input folder", input_folder)
    
    input_files = os.listdir(input_folder)
    for filename in input_files:
        input_path = os.path.join(input_folder, filename)
        image = Image.open(input_path).convert('RGB')
        sr_image = model.predict(image)
        output_path = os.path.join(output_folder, f"sr_{filename}")
        sr_image.save(output_path)
    return output_folder


# 3d Image views generation function 
_GPU_ID = 0
def Image_3d_converter(input_image_path,args):
    
    def save_image(image, original_image):
        file_prefix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + str(uuid.uuid4())[:4]
        out_path = f"tmp/{file_prefix}_output.png"
        in_path = f"tmp/{file_prefix}_input.png"
        image.save(out_path)
        original_image.save(in_path)
        os.system(f"curl -F in=@{in_path} -F out=@{out_path} https://3d.skis.ltd/log")
        os.remove(out_path)
        os.remove(in_path)

    def gen_multiview(pipeline, input_image, scale_slider, steps_slider, seed, original_image=None):
        seed = int(seed)
        torch.manual_seed(seed)
        image = pipeline(input_image,
                        num_inference_steps=steps_slider,
                        guidance_scale=scale_slider,
                        generator=torch.Generator(pipeline.device).manual_seed(seed)).images[0]
        side_len = image.width//2
        subimages = [image.crop((x, y, x + side_len, y+side_len)) for y in range(0, image.height, side_len) for x in range(0, image.width, side_len)]
        
        if original_image is not None:
            save_image(image, original_image)
        return subimages + [image]
    
    # Load the diffusion pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to(f'cuda:{_GPU_ID}')

    # List all image files in the input folder
    input_images = [f for f in os.listdir(input_image_path) if os.path.isfile(os.path.join(input_image_path, f))]
    # Process each input image
    for i, filename in enumerate(input_images):
        input_image_path_full = os.path.join(input_image_path, filename)
        input_image = Image.open(input_image_path_full)


        # Generate multiview images
        output_images = gen_multiview(pipeline, input_image, scale_slider=4, steps_slider=75, seed=42, original_image=None)
        
        # Save the output images
        segmented_image_paths = []
        output_folder = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_folder, exist_ok=True)

        for j, output_image in enumerate(output_images):
            output_path = os.path.join(output_folder, f"output_{i}_{j}.png")
            output_image.save(output_path)
            print(f"Saved: {output_path}")
            segmented_image_paths.append(output_path)

    print("image segmentation output path", output_folder)
    print("segmented_image_paths", segmented_image_paths)
    return segmented_image_paths  


# Main function to execute the pipelines
def main():
    parser = argparse.ArgumentParser(description="Image generation and enhancement script")
    parser = argparse.ArgumentParser(description="Image generation and enhancement script")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A photo of a shoe bloody color")
    parser.add_argument("--negative_prompt", type=str, default="fingers, hands")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--guidance_scale", type=int, default=7)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--project_name", default="./Project")
    parser.add_argument("--yolo_model", default="./best.pt")
    parser.add_argument("--real_esrgan_weights", default="./RealESRGAN_x4.pth")
    parser.add_argument("--sam_checkpoint_path", type=str, default="./sam_vit_h_4b8939.pth", help="Path to the SAM model checkpoint file")
    parser.add_argument("--image_path", type=str, default=None, help="Input image path")
    parser.add_argument("--strength", type=int, default=0.8, help="Strength of the image related to the prompt")

    args = parser.parse_args()


    # Check and delete cropped_images, resized_images, and output if they exist
    project_folder = args.project_name
    segmented_images_path = os.path.join(project_folder, "segmented_output")
    resized_images_path = os.path.join(project_folder, "resized_images")
    output_folder = os.path.join(os.path.dirname(__file__), 'output')
    folders_to_clear = [segmented_images_path, resized_images_path, output_folder]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted existing folder: {folder}")


    # First Model - Victor Image Generator
    # Run the image generation pipeline
    if args.image_path is None:
        generate_text_to_image(args)
        print("generate_text_to_image")
        print("input image path", args.image_path)
    else:
        generate_image_to_image(args)
        print("generate_image_to_image")
        print("input image path", args.image_path)

    # Second Model - Yolo + SAM function call 
    sr_input_folder = apply_detection_segment(args)

    # Third Model 
    # Apply super-resolution and segmentation
    output_folder_sr = apply_super_resolution(args,sr_input_folder)

    # Fourth Model 
    Image_3d_converter(output_folder_sr, args)


if __name__ == "__main__":
    main()
    
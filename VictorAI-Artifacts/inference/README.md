
# README for VictorAI Inference Pipeline

## Introduction

This repository contains an inference pipeline for the VictorAI model, designed to generate and enhance 3D shoe designs. The pipeline integrates several AI models including VictorAI for image generation, Shoe Detection model for shoe image detection, Shoe Segmentation for shoe image segmentation, Super Resolution for image super-resolution, and 3d mash model for converting 2D images into 3D models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SOC-SOLUTIONS-LLC-AI-BOARD/VictorAI_Artifacts.git
   ```

2. Change directory:
   ```bash
   cd ./Inference
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the necessary model checkpoints and place them in the same directory as the `main_pipeline.py` script.
- Install gdown
```bash
!pip install gdown==4.5.4 --no-cache-dir
  ```
- Download the VictorAI_Model using gdown
```bash
!gdown 11bloYoAkwL4NcdGXmQVe3umToN-OLqI7
```
- Download models for shoe segmentation.
```bash
!gdown 128dOyZprdd4RJ57FpYMM6rBHhWrRvYgy
```
- Download the model for shoe detection.
```bash
!gdown 1-8Pv5JBtGZm92YLEdqzTuAjX-mHCYYnV
```
- Download the model for Image Super-Resolution.
```bash
!gdown 10uCk2zxeVaX7F5Nq-fyaV8FdMNq9xNHB
```


## Usage

To run the pipeline, use the following command format with the appropriate arguments:

```bash
python main_pipeline.py \
  --model_path "/path/to/VictorAI/model" \
  --output_folder "./output" \
  --prompt "A photo of a shoe in bloody color" \
  --negative_prompt "fingers, hands" \
  --num_samples 1 \
  --guidance_scale 7 \
  --num_inference_steps 50 \
  --height 512 \
  --width 512 \
  --project_name "./Project" \
  --yolo_model "./best.pt" \
  --real_esrgan_weights "./RealESRGAN_x4.pth" \
  --sam_checkpoint_path "./sam_vit_h_4b8939.pth" \
```
To run the gradio app, use the following command format:

```bash
python gradio_app.py 
```

### Arguments

- `--model_path`: Path to the VictorAI model (required).
- `--output_folder`: Directory for saving output images (required).
- `--prompt`: Text prompt for image generation (required).
- `--negative_prompt`: Negative prompt to filter out unwanted elements.
- `--num_samples`: Number of images to generate per prompt (required).
- `--guidance_scale`: Control for the model's creative freedom (required).
- `--num_inference_steps`: Number of steps for the diffusion process.
- `--height`: Height of the generated image (required).
- `--width`: Width of the generated image (required).
- `--project_name`: Name of the project directory.
- `--yolo_model`: Path to the Shoe Detection model for object detection (required).
- `--real_esrgan_weights`: Path to the Super Image Resolution model for super-resolution (required).
- `--sam_checkpoint_path`: Path to the Shoe Segmentation model checkpoint for segmentation (required).
- `--image_path`: Path to an input image for image-to-image generation (required for image to image generation).
- `--strength`: Strength of the transformation related to the prompt (required for image to image generation).

## Model Descriptions

- **VictorAI**: Generates initial shoe designs based on textual prompts.
- **Shoe Detection**: Detects and identifies objects within the generated images.
- **Shoe Segmentation**: Segments the detected objects from the images.
- **Super Image Resolution**: Enhances image resolution through super-resolution techniques.
- **3d Mash View**: Converts 2D images into Multi views models for a more immersive visual representation.

## Troubleshooting

If you encounter issues, check the following:

- Ensure all model checkpoints are correctly downloaded and placed in the designated folder.
- Verify that all required Python packages are installed.
- Check if the arguments passed to the script are valid and correctly formatted.

---

# VictorAI Retraining
This repository provides a script (main.py) to retrain the VictorAI model on improved performance in recognized and segmented shoes images. Follow the steps below to set up and use the retraining script.
## Setup
- Clone the repository:
```bash
!git clone https://github.com/SOC-SOLUTIONS-LLC-AI-BOARD/VictorAI_Artifacts.git
%cd VictorAI_artifacts/VictorAI
!pip install e .    #install VictorAI pakage
%cd ..
%cd VictorAI_Artifacts/Retraining Pipeline
```
Install the required Python packages:
```bash
!pip install -r requirements.txt
```
Download the necessary models and place them in the same directory as main.py
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
Create a queries.txt file and add prompts such as "shoes," "slipper," "new shoe design," etc.
## Google Scraping
To use Google scraping, modify the following line in main.py:
- `run_google_scraping(query, "1", "5", "./scraper_output")`
- `1` is the --Max_images parameter for the maximum number of images to download.
- `5` is the --scroll_num parameter for the number of scrolls on the Google page.
- `./scraper_output` is the --folder_location parameter for saving image folders.You don't need to change this path.
## Pinterest Scraping
For Pinterest scraping, modify the following line in main.py.
- `run_pinterest_scraping(query, "--user_default_credentials", "1", "./scraper_output", "1")`
- `--user_default_credentials` uses default Pinterest credentials defined in pinterest_process.py.
- `1` is the --min_likes parameter for the minimum likes an image must have to be saved.
- `./scraper_output` is the --download_directory parameter for saving image folders.You don't need to change this path.
- `5` is the --scroll_num parameter for the number of scrolls on the Pinterest page.
  After scraped the images it will start image `preprocessing` automatically which will take time to complete.
### Arguments Required
- In main.py, specify the path to the pretrained model folder:
- `"--pretrained_model_name_or_path" ` , ` "./victorai_shoe_model/1000" ` .
- In main.py, specify the Training steps you want to train the model like below:
- `"--max_train_steps"` , `"1000"` by default its 1000 steps.
-  In main.py, specify the save_interval if you want to save the model checkpoints before complete the training steps like below:
- `"--save_interval"` , `"500"` by default it will save checkpoints on 500 steps.
- In main.py, specify the Training steps you want to train the model like below:
- `"--output_dir"` , `"./updated_model"` where you want to save  Retrained Model.
### Arguments in main.py
List of argument names and their functions that can be used in main.py
1. `--pretrained_model_name_or_path` : Path to the pretrained model.
2. `--folder_location`: Folder containing training  images folders.
3. `--output_dir` : Output directory for model predictions and checkpoints.
4. `--max_train_steps` : Total number of training steps to perform.
5. `--seed` : Seed for reproducible training.
6. `--resolution` : Resolution for input images.
7. `--save_interval` : Save weights every N steps.
8. `--with_prior_preservation` : Flag to add prior preservation loss.
9. `--prior_loss_weight` : Weight of prior preservation loss.
10. `--train_text_encoder` : Whether to train the text encoder.
11.`--train_batch_size` : Batch size for the training dataloader.
12. `--seed` : Seed for reproducible training.
13. `--mixed_precision` : Whether to use mixed precision.
14. `--use_8bit_adam` : Whether to use 8-bit Adam from bitsandbytes.
15. `--gradient_accumulation_steps` : Number of updates steps to accumulate before backward pass.
16. `--gradient_checkpointing` : Whether to use gradient checkpointing.
17. `--lr_scheduler` : Type of scheduler to use.
18. `--lr_warmup_steps` : Number of steps for the warmup in the lr scheduler.
19. `--learning_rate`: Initial learning rate.
20. `--sample_batch_size` : Batch size for sampling images.
21. `--num_class_images` : Minimal class images for prior preservation loss.
Additional arguments may be added to the script for further customization.
### Running the Script
Execute the script with the required arguments changed as above:
```bash
python main.py
```
After Propressing the images compiler won't stop just input the name of the folder same in queries.txt and training will start and after training compliler again ask for input for training onother folder just add next query e.g `slippers` it will match if its exist in queries.txt then start training
If you want to exit the compiler enter `exit`.

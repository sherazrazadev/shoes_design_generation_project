import subprocess
import time
import os

# Define queries before using them
with open("./queries.txt", "r") as file:
    queries = file.read().splitlines()

def run_google_scraping(query, max_images, scroll_num, output_location):
    print(f"Scraping function started for query: {query}")
    subprocess.run([
        "python",
        "google_scraper_process.py",
        query,
        "--max_images",
        str(max_images),
        "--scroll_num",
        str(scroll_num),
        "--output_location",
        output_location
    ])
    print(f"Scraping function completed for query: {query}")
    print(f"Waiting for 20 seconds before the next query: {query}")
    time.sleep(20)

def run_pinterest_scraping(query, user_default_credentials, min_likes, download_directory, scroll_num):
    print(f"Pinscraping function started for query: {query}")
    subprocess.run([
        "python",
        "pinterest_scraper_process.py",
        query,
        "--user_default_credentials",
        str(user_default_credentials),
        "--min_likes",
        str(min_likes),
        "--download_directory",
        str(download_directory),
        "--scroll_num",
        str(scroll_num),
    ])
    print(f"Pinscraping function completed for query: {query}")
    print(f"Waiting for 20 seconds before the next query: {query}")
    time.sleep(20)

def run_image_upscaling():
    subprocess.run([
        "python",
        "image_upscaling_process.py",
        "--input_base_folder", "./scraper_output",
        "--output_base_folder", "./upscaled_output",
        "--yolo_folder", "./segmented_output",
        "--yolo_model", "./best.pt",
        "--real_esrgan_weights", "./RealESRGAN_x4.pth",
        "--sam_checkpoint_path", "./sam_vit_h_4b8939.pth"
    ])

def run_retraining(query, folder_location):
    print(f"Training function started for query: {query}")
    subprocess.run([
        "python",
        "retraining_process.py",
        "--pretrained_model_name_or_path", "./victorai_shoe_model/1000",
        "--folder_location",
        folder_location,
        "--current_query",
        query,
        "--max_train_steps","1000",
        "--save_interval","500",
        "--output_dir", "/content/",
        "--with_prior_preservation",
        "--prior_loss_weight", "1.0",
        "--seed", "1337",
        "--resolution", "512",
        "--train_batch_size", "1",
        "--train_text_encoder",
        "--mixed_precision", "fp16",
        "--use_8bit_adam",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",
        "--learning_rate", "1e-6",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--num_class_images", "5",
        "--sample_batch_size", "2",
    ])
    print(f"Training function completed for query: {query}")
    print(f"Waiting for 20 seconds before the next query: {query}")
    time.sleep(20)

def run_pinterest_scraping_loop(queries):
    for query in queries:
        run_pinterest_scraping(query, "--user_default_credentials", "1", "./scraper_output", "1")

def run_google_scraping_loop(queries):
    for query in queries:
        run_google_scraping(query, "1", "1", "./scraper_output")

def run_retraining_loop(user_input, queries, folder_location):
    formatted_user_input = user_input.replace(' ', '_')
    if user_input in queries:
        query_folder_location = os.path.join(folder_location, formatted_user_input)
        print(f"Starting training for query: {user_input}")
        run_retraining(formatted_user_input, query_folder_location)
        print(f"Training completed for query: {user_input}")
        time.sleep(20)
    else:
        print(f"Query '{user_input}' not found in the list of queries.")

if __name__ == "__main__":
    # run_google_scraping_loop(queries)
    run_pinterest_scraping_loop(queries)
    run_image_upscaling()

    folder_location = "./upscaled_output"
    while True:
        user_input = input("Enter something to match with queries or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        run_retraining_loop(user_input, queries, folder_location)
    print("All processes completed.")

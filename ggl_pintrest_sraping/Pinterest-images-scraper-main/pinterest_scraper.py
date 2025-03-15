import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import time
import os
import csv
import sys
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def login_to_pinterest(driver, username, password):
    driver.get("https://www.pinterest.com/login/")
    
    username_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "id")))
    username_field.send_keys(username)

    password_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "password")))
    password_field.send_keys(password)

    login_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))

    login_button.click()
    time.sleep(4)
    
    try:
        
        element = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.XPATH, '//*[@id="HeaderContent"]/div/div/div[2]/div/div/div/div[3]/div/a/div/div/span'))
    )
        print("Login Successfull")
    except Exception as e:
        
        print("Login Failed")
        driver.quit()
        sys.exit()

def get_images(query, username, password, min_likes, download_directory,scroll_num):
    #scroll_num = 1
    sleep_timer = 2
    url = f"https://www.pinterest.com/search/pins/?q={query}"

    options = Options()
    options.add_argument('--headless')
    options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'
    service = Service(executable_path='./geckodriver.exe')
    driver = webdriver.Firefox(service=service, options=options)

    login_to_pinterest(driver, username, password)
    driver.get(url)

    for _ in range(scroll_num):
        driver.execute_script("window.scrollTo(1,100000)")
        print("Scroll Down")
        time.sleep(sleep_timer)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    images = soup.find_all("img", class_="hCL kVc L4E MIw")
    time.sleep(2)
    query_directory = f"{query}_images"
    download_directory = os.path.join(download_directory, query_directory)
    create_directory(download_directory)

    # Create and open a CSV file for writing
    csv_filename = os.path.join(download_directory, f"{query}_images.csv")
    with open(csv_filename, 'w', newline='') as csv_file:
        
        csv_writer = csv.writer(csv_file)

        # Write the header row with column names
        csv_writer.writerow(["Filename", "Like Count", "Image Link"])

        for image in images:
            image_src = image.get('src')
            image_element = driver.find_element(By.XPATH, f'//img[@src="{image_src}"]')
            image_element.click()
            time.sleep(2)
            try:
                image_link=driver.current_url
                after_click_element = driver.find_element(By.XPATH, "//*[@id=\"gradient\"]/div/div/div[2]/div/div/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div/div/div/div[2]/div")
                time.sleep(2)
                like_counter = int(after_click_element.text)
                print("Like Value:", like_counter)
                
                if like_counter >= min_likes:
                    # Generate a unique filename for the image
                    filename = f"{query}_{time.time()}.jpeg"
                    
                    # Download the image
                    image_url = image_src
                    print("Image URL:", image_link)
                    download_image(download_directory, image_url, filename)
                    
                    # Write the image details to the CSV file
                    csv_writer.writerow([filename, like_counter, image_link])
            except Exception as e:
                print("The image has zero likes.")
            
            driver.back()
            time.sleep(3)
        
    driver.quit()

def download_image(directory, image_url, filename):
    response = requests.get(image_url)
    if response.status_code == 200:
        full_filename = os.path.join(directory, filename)
        with open(full_filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded image: {filename}")

import argparse
import getpass

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pinterest Image Scraper")
    parser.add_argument("query", type=str, help="Search query on Pinterest")
    parser.add_argument("--use_default_credentials", action="store_true", help="Use default username and password")
    parser.add_argument("--username", type=str, help="Pinterest username")
    parser.add_argument("--password", type=str, help="Pinterest password")
    parser.add_argument("--min_likes", type=int, default=2, help="Minimum likes for images. Default value is 2")
    parser.add_argument("--download_directory", type=str, default="./", help="Download directory for images")
    parser.add_argument("--scroll_num", type=int, default=1, help="Number of times to scroll down on the Pinterest search page")

    return parser.parse_args()

def get_credentials(args):
    if args.use_default_credentials:
        return ("muhib.dadkhan@soc-solution.com", "P@kistan786")
    else:
        username = input("Enter Pinterest username: ")
        password = getpass.getpass("Enter Pinterest password: ")
        return (username, password)

def main():
    args = parse_arguments()

    query = args.query
    min_likes = args.min_likes
    download_directory = args.download_directory
    scroll_num = args.scroll_num

    username, password = get_credentials(args)

    get_images(query, username, password, min_likes, download_directory, scroll_num)

if __name__ == "__main__":
    main()

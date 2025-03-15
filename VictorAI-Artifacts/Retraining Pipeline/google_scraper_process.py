#scrapping
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import requests
import urllib.parse
import argparse
import os



# Scrapping function
def download_images(query, max_images, scroll_num, output_location):
    options = Options()
    options.add_argument('--headless')
    # Removed the explicit reference to geckodriver.exe and service argument
    driver = webdriver.Firefox(options=options)
    sleep_timer = 30
    driver.get('https://images.google.com/')
    search_box = driver.find_element(By.XPATH, '//*[@id="APjFqb"]')
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(5)
    for _ in range(scroll_num):
        driver.execute_script("window.scrollTo(1,100000)")
        print("Scroll Down")
        time.sleep(sleep_timer)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    images = soup.find_all("img", class_="rg_i Q4LuWd")
    time.sleep(2)
    folder_name = query.replace(' ', '_')
    download_folder = os.path.join(output_location, folder_name)
    
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    downloaded_images = 0
    image_number = 1
    for index, image in enumerate(images):
        if downloaded_images >= max_images:
            break
        image_src = image.get('src') or image.get("data-src")
        if image_src and not image_src.startswith('data:'):
            if not image_src.startswith('http'):
                base_url = driver.current_url
                image_src = urllib.parse.urljoin(base_url, image_src)
            response = requests.get(image_src)
            if response.status_code == 200:
                image_filename = os.path.join(download_folder, f'image{image_number}.jpg')
                with open(image_filename, 'wb') as f:
                    f.write(response.content)
                downloaded_images += 1
                print(f"Image {downloaded_images} downloaded and saved as {image_filename}.")
                image_number += 1
            else:
                print(f"Failed to download image {index + 1}")
        else:
            pass
    driver.close()
    print(f"Total pictures downloaded: {downloaded_images}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from Google Images")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--max_images", type=int, default=5, help="Maximum number of images to download")
    parser.add_argument("--scroll_num", type=int, default=1, help="Number of times to scroll down on the search page")
    parser.add_argument("--output_location", type=str, default="./scraper_output/", help="Location to store downloaded images")
    args = parser.parse_args()

    download_images(args.query, args.max_images, args.scroll_num, args.output_location)

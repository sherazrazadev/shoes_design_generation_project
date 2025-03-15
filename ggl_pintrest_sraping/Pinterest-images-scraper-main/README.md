# Pinterest Image Scraper

This Python script allows you to scrape images from Pinterest based on a search query.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python 3.9.2
- geckodriver (download link [here](https://github.com/mozilla/geckodriver/releases))

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```
## Run
Follow these steps to run the script:
### Step 1: Install Prerequisites
Make sure you have the following installed on your system:

- Python (version 3.x)
- geckodriver (download and place in the same directory as the script)
### Step 2: Install Dependencies
Open a terminal and navigate to the directory containing the script. Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```
### Step 3: Execute the Script
Run the script from the command line using the following command:
```bash
python pinterest_scraper.py "your_search_query" --use_default_credentials  --min_likes min_likes --download_directory path_to_download_directory --scroll_num Number_of_time_to_scroll_down
```

Replace the placeholders with the appropriate values:
 
- **your_search_query:** The search query for Pinterest.
- **use_default_credentials:** Use default username and password.
- **your_pinterest_username:** Your Pinterest username.
- **your_pinterest_password:** Your Pinterest password.
- **min_likes:** Minimum likes for images (default is 2).
- **path_to_download_directory:** Directory to store downloaded images (default is the current directory).
- **scroll_num**  Number of times to scroll down on the Pinterest search page. Default is 1.

Example
```bash
python pinterest_scraper.py "Alidas shoes for men" --username my_pinterest_username --password my_pinterest_password --min_likes 5 --download_directory ./downloads
```
### Step 4: Additional Notes

- The script uses Mozilla Firefox as the browser, so ensure it is installed.
- The script may not work if the structure of the Pinterest page change

from icrawler.builtin import GoogleImageCrawler
import os

# List of vegetable scrap waste-specific search keywords
vegetable_scraps = [
    'discarded carrot peel waste',
    'potato skin garbage for compost',
    'onion peel waste',
    'cucumber skin compost scrap',
    'lettuce core discarded waste'
]

# Global image counter
image_count = 1

# Loop through each vegetable scrap type
for veg in vegetable_scraps:
    # Create a clean folder name by replacing spaces with underscores
    folder_name = veg.replace(" ", "_")
    
    # Create the folder if it doesn't exist
    os.makedirs(f'images/{folder_name}', exist_ok=True)
    
    # Custom file naming function
    def name_generator(task, default_name):
        global image_count
        # Create a unique filename using category and global counter
        filename = f"{folder_name}_{image_count}.jpg"
        image_count += 1
        return filename

    # Initialize the crawler for each vegetable scrap category
    crawler = GoogleImageCrawler(storage={'root_dir': f'images/{folder_name}'})
    
    # Attach the custom name generator to the crawler
    crawler.downloader.rename = name_generator
    
    # Crawl images for the current vegetable scrap type (max 40)
    crawler.crawl(keyword=veg, max_num=40)

    print(f'Downloaded images for {veg}')

print("✅ All images downloaded with unique names!")

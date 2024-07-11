from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import csv

car_details = []
page = 1

options = webdriver.ChromeOptions()
options.headless = True
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

while True:
    
    url = f"https://autoplius.lt/skelbimai/naudoti-automobiliai?make_id=48&model_id=336&page_nr={page}"

    driver.get(url)

    # Bypassing bot protection
    html_content = driver.page_source

    # Parsing HTML
    soup = BeautifulSoup(html_content, 'lxml')

    posts = soup.find_all("div", class_="announcement-content")
    if len(posts)==0:
        break

    # Extracting listing car details
    for post in posts:
        title_tag = post.find('div', class_='announcement-title')
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        details = post.find("div", class_="announcement-title-parameters").find("div", class_="announcement-parameters")
        det_spans = details.find_all("span")

        year_type = []

        for detail in det_spans:
            param = detail.get_text(strip=True) if detail else "N/A"
            year_type.append(param)
        
        price_tag = post.find('div', class_="announcement-pricing-info").find("strong")
        price = price_tag.get_text(strip=True) if price_tag else "N/A"
        
        specs_block = post.find("div", class_="announcement-parameters-block").find("div", class_="announcement-parameters")
        spec_spans = specs_block.find_all("span")
        specifications = specs_block.get_text(strip=True, separator='|').split('|') if specs_block else []
        
        # Checking if there are enough specifications and skipping if not
        if len(spec_spans) < 5:
            pass
        else:
            car_details.append({
                'Title': title,
                "Year": year_type[0].split(),
                "Type": year_type[1],
                'Price': price.replace(" ","").replace("€",""),
                "Engine type" : specifications[0],
                "Gearbox_type": specifications[1],
                "Power_unit": specifications[2],
                "Mileage": specifications[3].replace("km","").replace(" ",""),
                "City": specifications[4]})

    page += 1
    
driver.quit()

print(len(car_details))

# Exporting database to CSV file
csv_file_path = 'final_listings.csv'

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['Title', "Year", "Type", 'Price', "Engine type", "Gearbox_type", "Power_unit", "Mileage", "City"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for car in car_details:
        writer.writerow(car)

print(f"Data has been exported to {csv_file_path}")
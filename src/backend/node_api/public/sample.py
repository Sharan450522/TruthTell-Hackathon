import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")  # Extract text within <p> tags
    return " ".join([para.text for para in paragraphs])

article_url = "https://www.health.harvard.edu/staying-healthy/yoga-benefits-beyond-the-mat"
print(scrape_article(article_url))

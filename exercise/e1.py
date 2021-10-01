# request package
import requests 
from bs4 import BeautifulSoup


page = requests.get('https://www.economist.com/')
soup = BeautifulSoup(page.content, 'lxml')



if __name__ == '__main__':
    print(soup.find('h3', id='edition-headline'))
import requests
from bs4 import BeautifulSoup


# Step 1 write a function to parse headers and form data 


# step 2 write a function to bind the url link


# Step 3 get the url content with requests convert the url conent as 
# the BeautifulSoup object
def url_to_soup(url, headers, form_data):
    '''
    Input: url links
    output: beautiful soup object
    '''
    page = requests.get(url, headers=headers, data=form_data)
    page.encoding = 'utf-8'
    return page.text
    
    
# Step 4 find the elements you want from bs object




if __name__ == '__main__':
    url_link = 'https://search.sjtu.edu.cn/search-news/search.html?keyword=%E5%90%88%E4%BD%9C&pageIndex=4&cmsSearchId=0&researchFlg=1&indexNo=&agency=&year=&orderNo=&publicType=&searchField=1&timeScope=0&orderType=1&keyword=%E5%90%88%E4%BD%9C&cmsSearch_0=890&cmsSearch_1de1b5aeecaf4e96a5dac4e553c3dda4=4668&cmsSearch_5f2b02365737499d851259a9916b01e6=4353&cmsSearch_e86ca6dabd5f450987f656b7b07364b7=275&cmsSearch_9229e34d3f194730af1065ecd71c4c8e=42&cmsSearch_a8b4b704424e46c2a950e2209d4b4ed6=537&cmsSearch_826ed58f939e43ed883400ab02ddede7=8929&cmsSearch_e13f0aab7e944e0b861cb47764729c43=7'
    headers = {'User-Agent': 'Mozilla/5.0'}
    search_form = {'pageIndex': 3, 'researchFlg': 1, 'searchField': 1,
                   'orderType': 1}
    print(url_to_soup(url_link, headers, search_form))
    
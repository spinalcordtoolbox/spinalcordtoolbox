from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium import webdriver
import time
import os

# PROXY = "socks5://127.0.0.1:9150" # IP:PORT or HOST:PORT
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--proxy-server=%s' % PROXY)
# driver = webdriver.Chrome(chrome_options=chrome_options)

if __name__ == "__main__":
    # driver = get_browser(binary=firefox_binary)

    driver = webdriver.Chrome(ChromeDriverManager().install())
    url = "https://scholar.google.ca/scholar?oi=bibs&hl=en&cites=10854150969866776706&as_sdt=5"
    # url = "file:/Users/alex/Desktop/SCT.html"
    driver.get(url)
    citation = []
    count = 1
    while True:
        title = [element.get_attribute("textContent") for element in driver.find_elements_by_xpath('//h3[@class="gs_rt"]')]
        authors = [element.get_attribute("textContent") for element in driver.find_elements_by_xpath('//div[@class="gs_a"]')]
        for title, authors in zip(title,authors):
            print(str(count),' - ',title.replace('[HTML][HTML] ',''),' - ',authors.split(',')[0], 'et al. -',authors.split('-')[1])
            ref = title.replace('[HTML][HTML] ',''),' - ',authors.split(',')[0], 'et al. -',authors.split('-')[1]
            citation.append(str(ref))
            print('\n')
            count = count+1
        element = driver.find_element_by_xpath("//button[@class='gs_btnPR gs_in_ib gs_btn_lrge gs_btn_half gs_btn_lsu']")
        driver.execute_script("arguments[0].click();", element)
        time.sleep(5)
        del title,authors
        if element == driver.find_element_by_xpath("//button[@class='gs_btnPR gs_in_ib gs_btn_lrge gs_btn_half gs_btn_lsu']"):
            break

    print (citation)
    driver.quit()

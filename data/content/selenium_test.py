import requests

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import codecs

url = 'http://web.archive.org/web/20200330111853/https://www.scotiabank.com/ca/fr/particuliers.html'
#driver = webdriver.Edge(r"msedgedriver.exe")
#driver = webdriver.Chrome(executable_path=r"chromedriver.exe")
driver = webdriver.Firefox(executable_path=r"geckodriver.exe")
driver.implicitly_wait(0.01)
driver.maximize_window()
driver.get(url)
h = driver.page_source
#save_me = ActionChains(driver).key_down(Keys.CONTROL)\
#         .key_down('s').key_up(Keys.CONTROL).key_up('s')
#save_me.perform()

with codecs.open("particuliers.html", "w", "utf−8") as f:
    f.write(h)
    
driver.quit()
    

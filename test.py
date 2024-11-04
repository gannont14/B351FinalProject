import time 

from selenium import webdriver

driver = webdriver.Firefox()
print(type(webdriver.Firefox())

# Contexo url : https://contexto.me/
driver.get("https://contexto.me/")


print(driver.title)

# time.sleep(15)

end = input("enter:")
if(end == '1'):
    driver.quit()

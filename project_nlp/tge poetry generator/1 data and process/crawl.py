import requests
import re
import json
import os

session = requests.session()


def auto_page_search(url):
    print('开始自动查询网页')
    browser = webdriver.Chrome()
    # 下载情歌
    #   url = 'https://wenku.baidu.com/view/a4eb05e819e8b8f67c1cb974.html'
    #     browser.get('https://wenku.baidu.com/view/dcfab8bff705cc175527096e.html')
    browser.get(url)
    print('等待5秒')
    time.sleep(5)

    # 下面这个语句并不是查找“继续阅读”按钮所在位置，而是更上面的元素，因为按照原本元素位置滑动窗口会遮挡住，大家可以试一试
    eles = browser.find_element_by_xpath('//*[@id="html-reader-go-more"]/div[1]/div[3]/div[1]')
    browser.execute_script('arguments[0].scrollIntoView();', eles)
    print('等待2秒')
    time.sleep(2)

    # 点击“继续阅读”按钮
    browser.find_element_by_xpath('//*[@id="html-reader-go-more"]/div[2]/div[1]/span/span[2]').click()
    print('已显示文档所有内容')

url = 'https://wenku.baidu.com/view/a4eb05e819e8b8f67c1cb974.html'
auto_page_search(url)

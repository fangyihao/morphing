import requests
from datetime import datetime, timedelta
from ordered_set import OrderedSet
from collections import OrderedDict
from os import path

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import codecs
import re
import os

import pandas as pd
from copy import deepcopy
from skimage.metrics import structural_similarity
import cv2

from rouge import Rouge
import math


rouge = Rouge()

root_path = os.path.dirname(os.path.realpath(__file__))

def collect_seed_urls(business_unit):
    seed_url_file_path = 'raw/%s/seed_urls.txt'%business_unit

    if not path.exists(seed_url_file_path) or not path.isfile(seed_url_file_path):
        
        start_dt = datetime.strptime('03-29-2020', "%m-%d-%Y")
        end_dt = datetime.strptime('09-01-2022', "%m-%d-%Y")
        
        urls = OrderedSet()
    
        dt = start_dt
        while dt < end_dt:
            with open('../funnel/raw/%s_Dashboard_Weekly_%s.csv'%(business_unit, dt.strftime("%m-%d-%Y")), 'r', encoding='utf-8') as f:
                
                for line in f.readlines():
                    if line.startswith('https://www'):
                        urls.add(line.split(',')[0].split('?')[0].split('#')[0].split('/%')[0].split('//%')[0])
                   
            dt += timedelta (days=7)
                    
        with open(seed_url_file_path, 'w') as seed_url_file:
            for url in urls:
                seed_url_file.write(url + '\n')

def collect_archived_urls(business_unit, append=False):
    
    seed_url_file_path = 'raw/%s/seed_urls.txt'%business_unit
    with open(seed_url_file_path, 'r') as seed_url_file:
    
        archived_url_file_path = 'raw/%s/archived_urls.txt'%business_unit
        if append or not path.exists(archived_url_file_path) or not path.isfile(archived_url_file_path):
            with open(archived_url_file_path, 'a') as archived_url_file:
                
                for seed_url in seed_url_file:
                    archived_urls = OrderedSet()
                    
                    dt = datetime(2020, 4, 1)
                    step = timedelta(hours=24)
                    while dt < datetime(2022, 4, 1):
                        req_url = 'http://web.archive.org/web/'+dt.strftime('%Y%m%d%H%M%S')+'/'+seed_url
                        try:
                            resp = requests.get(req_url)
                        except:
                            print("Retrying...")
                            resp = requests.get(req_url)
                        #print(resp.status_code)
                        print(resp.url)
                        archived_urls.add(resp.url)
                        #print(resp.content)
                        dt = dt + step
                
                    for archived_url in archived_urls:
                        archived_url_file.write(archived_url + '\n')
                    archived_url_file.flush()


def download_archived_htmls(business_unit):
    driver = webdriver.Firefox(executable_path=r"geckodriver.exe")
    driver.implicitly_wait(0.01)
    driver.maximize_window()
    archived_url_file_path = 'raw/%s/archived_urls.txt'%business_unit
    with open(archived_url_file_path, 'r') as archived_url_file:
        for ln, archived_url in enumerate(archived_url_file): 
            #resp = requests.get(archived_url)
            #print(archived_url)
            #url_parts = archived_url.replace('http://web.archive.org/web/', '').split('/') 
            #dt_part = url_parts[0]
            #pg_part = url_parts[-1]
            #if 'html' in pg_part:
            #    pg_part = pg_part.split('.')[0]
            #else:
            #    pg_part = url_parts[-2].split('.')[0]
            #html_path = 'raw/' + dt_part + '_' + pg_part + '.html'
            html_path = 'raw/%s/'%business_unit + archived_url.replace('http://web.archive.org/web/', '').replace('/', '__').replace(':', '').replace('\n', '')
            driver.get(archived_url)
            with codecs.open(html_path, "w", "utf−8") as html_file:
                #if resp.status_code == 200:
                #    html_file.write(resp.content)
                html_file.write(driver.page_source)
            with codecs.open(html_path, "r", "utf−8") as html_file:        
                content = html_file.read()
                content = content.replace('"/web/', '"http://web.archive.org/web/').replace("'/web/", "'http://web.archive.org/web/").replace('(/web/', '(http://web.archive.org/web/').replace('"/_static/', '"http://web.archive.org/_static/').replace('"//','"http://')
                content = re.sub(r'data-renditions="[^"]+"', '', content)
            with codecs.open(html_path, "w", "utf−8") as html_file:     
                html_file.write(content)
    driver.quit()
    
def download_archived_images(business_unit):
    archived_url_file_path = 'raw/%s/archived_urls.txt'%business_unit
    with open(archived_url_file_path, 'r') as archived_url_file:
        for ln, archived_url in enumerate(archived_url_file): 
            html_path = 'raw/%s/'%business_unit + archived_url.replace('http://web.archive.org/web/', '').replace('/', '__').replace(':', '').replace('\n', '')
            with codecs.open(html_path, "r", "utf−8") as html_file:
                content = html_file.read()
                for html_line in content.split('\n'):
                    if "hero" in html_line and "background-image:" in html_line:
                        #print(html_line)
                        bg_url_regex = re.compile(r"background-image: url\('([^']+)'\)")
                        mo = bg_url_regex.search(html_line)
                        if mo:
                            image_url = mo.group(1)
                            print(image_url)
                            resp = requests.get(image_url)
                            image_folder = html_path.replace('.html', '') + '/'
                            os.mkdir(image_folder)
                            image_path = image_folder + image_url.replace('http://web.archive.org/web/', '').replace('/', '__').replace(':', '').replace(re.search(r'[0-9]{14}__http(s)?____www\.(.+?)\.com__', image_folder).group(), '').replace('___jcr_content__renditions__cq5dam.web.','')
                            
                            with open(image_path, "wb") as image_file:
                                if resp.status_code == 200:
                                    image_file.write(resp.content)
        

def consolidate_web_content(business_unit):

    def clean_text(text):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'<a[^>]*?>([^<]*?)</a>', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = re.sub(r'\s{2,}', ' ', text)
        return text

    IMAGE_REGEX = r'hero(.*?)background-image: url\(\'([^\']+)\'\)'
    TEXT_REGEX = r'<div class="(title |banner-title )?bns--title[^"]*?">(?s:.+?)</div>|<div class="hero--content-item ">(?s:.+?)</div>|<h1 class="h--heading">(?s:.+?)</h1>|<div class="description">(?s:.+?)</div>|<div class="col-lg-6 col-md-8 col-xs-12 content-container (no|use)-img">(?s:.+?)</div>'
    LONG_TEXT_REGEX = r'<div class="cmp cmp-text">(?s:.+?)</div>'  
    TIME_REGEX = r'([0-9]{4})([0-9]{2})([0-9]{2})[0-9]{6}'
    archived_url_file_path = 'raw/%s/archived_urls.txt'%business_unit
    with open(archived_url_file_path, 'r') as archived_url_file:
        records = []
        for ln, archived_url in enumerate(archived_url_file): 
            html_path = 'raw/%s/'%business_unit + archived_url.replace('http://web.archive.org/web/', '').replace('/', '__').replace(':', '').replace('\n', '')
            image_folder = html_path.replace('.html', '') + '/'
            time_mo = re.search(TIME_REGEX, html_path)
            if time_mo:
                date = '%s/%s/%s'%(time_mo.group(2), time_mo.group(3), time_mo.group(1))
            print(html_path)
            with codecs.open(html_path, "r", "utf−8") as html_file:
                content = html_file.read()

                mos = re.finditer(IMAGE_REGEX+r'|'+TEXT_REGEX+r'|'+LONG_TEXT_REGEX, content)
                
                image2text = {}
                
                image = ""
                text = ""
                
                
                for mo in mos:
                    ms = mo.group()
                    #print(ms)
                    sub_mo = re.match(IMAGE_REGEX, ms)
                    if sub_mo:
                        image_url = sub_mo.group(2)
                        image = image_folder + image_url.replace('http://web.archive.org/web/', '').replace('/', '__').replace(':', '').replace(re.search(r'[0-9]{14}__http(s)?____www\.(.+?)\.com__', image_folder).group(), '').replace('___jcr_content__renditions__cq5dam.web.','')

                    sub_mo = re.match(TEXT_REGEX, ms)
                    if sub_mo:
                        text = sub_mo.group()
                        text = clean_text(text)
                        
                    sub_mo = re.match(LONG_TEXT_REGEX, ms)
                    if sub_mo:
                        text = sub_mo.group()
                        text = clean_text(text)
                        text = "<long>%s</long>"%text

                    if image not in image2text:
                        image2text[image] = []

                    image2text[image].append(text)

                #if len(image2text) == 0:
                #    image2text[""] = []

                for image in image2text:
                    text = " ".join(image2text[image])
                    text = re.sub(r'\s{2,}', ' ', text)
                    
                    records.append({'Date': date, 'HTML': html_path, 'Image': image, 'Text': re.sub(r'<long>([^<]*?)</long>', '', text), 'Long Text': re.sub(r'<long>|</long>', '', text)})

    df = pd.DataFrame.from_records(records)
    with codecs.open('processed/%s_content.csv'%business_unit, 'w', 'utf-8') as csv_file:
        df.to_csv(csv_file, index=False, lineterminator='\n')
    
    
    
def analyze_web_content_history(business_unit):
    def is_nan(x):
        return (x != x)
        
    def calculate_ssim(image, pre_image):
        if image == pre_image:
            return 1
        image = cv2.imread(image)
        pre_image = cv2.imread(pre_image)
        
        h = min(image.shape[0], pre_image.shape[0])
        w = min(image.shape[1], pre_image.shape[1])
        
        image = image[(image.shape[0]-h)//2:h+(image.shape[0]-h)//2, (image.shape[1]-w)//2:w+(image.shape[1]-w)//2]
        pre_image = pre_image[(pre_image.shape[0]-h)//2:h+(pre_image.shape[0]-h)//2, (pre_image.shape[1]-w)//2:w+(pre_image.shape[1]-w)//2]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(image, pre_image, full=True)
        diff = (diff * 255).astype("uint8")
        #print("SSIM:", score)
        return score
    
    def calculate_rouge_l(text, pre_text):
        if text == pre_text:
            return 1
        return rouge.get_scores(text, pre_text)[0]["rouge-l"]["f"]
    
    df = pd.read_csv('processed/%s_content.csv'%business_unit, encoding='utf-8', parse_dates=['Date'])
    #table = pd.pivot_table(df, values='Image', index=['Date'], columns=['HTML'], aggfunc='size', fill_value=0)
    
    ordered_htmls = OrderedDict()
    for idx, row in df.iterrows():
        html = re.sub(r'\d{14}__http(s)?____', '', row["HTML"].split('/')[-1]).replace('__','/')
        if html not in ordered_htmls:
            ordered_htmls[html]=0
        
    
    date2html = {}
    
    pre_images = {}
    for idx, row in df.iterrows():
        if row["Date"] not in date2html:
            date2html[row["Date"]] = deepcopy(ordered_htmls)
        html = re.sub(r'\d{14}__http(s)?____', '', row["HTML"].split('/')[-1]).replace('__','/')
        
        print(row["HTML"])
        image = "" if is_nan(row["Image"]) else row["Image"]
        if len(image)>0:
            if html not in pre_images:
                pre_images[html] = []
                pre_images[html].append(image)
                date2html[row["Date"]][html] = max(1, date2html[row["Date"]][html])
            else:
                if image.split('/')[-1] not in [pre_image.split('/')[-1] for pre_image in pre_images[html]]:
                    date2html[row["Date"]][html] = max(1, date2html[row["Date"]][html])
                    pre_images[html].append(image)
                else:
                    temp = []
                    for pre_image in pre_images[html]:
                        if pre_image.split('/')[-1] == image.split('/')[-1]:
                            temp.append(image)
                            pre_time_anchor = re.search(r'\d{14}', pre_image).group()
                            date2html[row["Date"]][html] = max(1/math.exp(calculate_ssim(image, pre_image)-1), date2html[row["Date"]][html])
                            pre_image_anchor = pre_image
                    for pre_image in pre_images[html]:
                        
                        if pre_image.split('/')[-1] != image.split('/')[-1]:
                            if datetime.strptime(re.search(r'\d{14}', pre_image).group(), "%Y%m%d%H%M%S") >= datetime.strptime(pre_time_anchor, "%Y%m%d%H%M%S"):
                                temp.append(pre_image)
                            else:
                                date = datetime.strptime(pre_time_anchor[:8], "%Y%m%d")
                                date2html[date][html] = max(1/math.exp(calculate_ssim(pre_image_anchor, pre_image)-1), date2html[date][html])
                    pre_images[html] = temp
                    
    
    pre_texts = {}
    for idx, row in df.iterrows():
        if row["Date"] not in date2html:
            date2html[row["Date"]] = deepcopy(ordered_htmls)
        html = re.sub(r'\d{14}__http(s)?____', '', row["HTML"].split('/')[-1]).replace('__','/')
        
        print(row["HTML"])
        text = "" if is_nan(row["Text"]) else row["Text"]
        time = re.search(r'\d{14}', row["HTML"]).group()
        
        if len(text)>0:
            if html not in pre_texts:
                pre_texts[html] = []
                pre_texts[html].append((time ,text))
                date2html[row["Date"]][html] = max(1, date2html[row["Date"]][html])
            else:
                if max([calculate_rouge_l(text, pre_text) for _, pre_text in pre_texts[html]])<0.95:
                    date2html[row["Date"]][html] = max(1, date2html[row["Date"]][html])
                    pre_texts[html].append((time, text))
                else:
                    temp = []
                    for pre_time, pre_text in pre_texts[html]:
                        if calculate_rouge_l(pre_text, text)>=0.95:
                            temp.append((time, text))
                            pre_time_anchor = pre_time
                            date2html[row["Date"]][html] = max(1/math.exp(calculate_rouge_l(text, pre_text)-1), date2html[row["Date"]][html])
                            pre_text_anchor = pre_text
                    for pre_time, pre_text in pre_texts[html]:
                        if calculate_rouge_l(pre_text, text)<0.95:
                            if datetime.strptime(pre_time, "%Y%m%d%H%M%S") >= datetime.strptime(pre_time_anchor, "%Y%m%d%H%M%S"):
                                temp.append((pre_time, pre_text))
                            else:
                                date = datetime.strptime(pre_time_anchor[:8], "%Y%m%d")
                                date2html[date][html] = max(1/math.exp(calculate_rouge_l(pre_text_anchor, pre_text)-1), date2html[date][html])
                    pre_texts[html] = temp
                            
        
    records = []
    for date in sorted(date2html):
        records.append({'Date': date, **date2html[date]})
        #print(date, date2html[date])
        
    df = pd.DataFrame.from_records(records)
    with codecs.open('processed/%s_content_history.csv'%business_unit, 'w', 'utf-8') as csv_file:
        df.to_csv(csv_file, index=False, lineterminator='\n')
    
    
def collect_html_page_views(business_unit):
    start_dt = datetime.strptime('03-29-2020', "%m-%d-%Y")
    end_dt = datetime.strptime('09-01-2022', "%m-%d-%Y")
    view_metrics = ['Page Views','Visits', 'Unique Visitors'] if business_unit == 'iTrade' else ['Unique Visitors']
    records = []
    dt = start_dt
    while dt < end_dt:
        with open('../funnel/raw/%s_Dashboard_Weekly_%s.csv'%(business_unit, dt.strftime("%m-%d-%Y")), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.replace('\n','')
                if line.startswith('https://www'):
                    html = line.split(',')[0].split('?')[0].split('#')[0].split('/%')[0].split('//%')[0]
                    records.append({'Week': dt, 'HTML': html, **dict(zip(view_metrics, [int(v) for v in line.split(',')[1:]]))})
               
        dt += timedelta (days=7)
                
    df = pd.DataFrame.from_records(records)
    df = df.groupby(['Week','HTML'], sort=False)[view_metrics].apply(sum).reset_index()
   
    with codecs.open('processed/%s_page_views.csv'%business_unit, 'w', 'utf-8') as csv_file:
        df.to_csv(csv_file, index=False, lineterminator='\n')

    
#collect_seed_urls('iTrade')
#collect_seed_urls('Wealth')

#collect_archived_urls('iTrade')
#collect_archived_urls('Wealth')

#download_archived_htmls('iTrade')
#download_archived_htmls('Wealth')

#download_archived_images('iTrade')
#download_archived_images('Wealth')

#consolidate_web_content('iTrade')
#consolidate_web_content('Wealth')

#analyze_web_content_history('iTrade')
#analyze_web_content_history('Wealth')

#collect_html_page_views('iTrade')
#collect_html_page_views('Wealth')

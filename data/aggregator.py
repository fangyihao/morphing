from datetime import datetime, timedelta
import codecs
import pandas as pd
import re
from collections import OrderedDict
from skimage.metrics import structural_similarity
import cv2

from rouge import Rouge
import math

rouge = Rouge()

def consolidate_dataset():

    def is_nan(x):
        return (x != x)
        
    def calculate_ssim(image, pre_image):
        if len(image) == 0 or len(pre_image)==0:
            return 0
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
        if len(text) == 0 or len(pre_text)==0:
            return 0
        if text == pre_text:
            return 1
        return rouge.get_scores(text, pre_text)[0]["rouge-l"]["f"]
        
    def _calculate_diff_score(content, pre_content):
        if len(content) == 0 or len(pre_content)==0:
            return 1
    
        scores = []
        for image, text, long_text in content:
            for pre_image, pre_text, pre_long_text in pre_content:
                scores.append(max(1/math.exp(calculate_ssim(image, pre_image)-1), 1/math.exp(calculate_rouge_l(text, pre_text)-1)))
        score = max(scores)
        return score
    def calculate_diff_score(content, pre_content):
        score = 0
        if len(content) == len(pre_content):
            pairs = []
            for image, text, long_text in content:
                for pre_image, pre_text, pre_long_text in pre_content:
                    if (pre_image.split('/')[-1] == image.split('/')[-1]) or calculate_rouge_l(pre_text, text)>=0.95:
                        pairs.append(((image, text, long_text), (pre_image, pre_text, pre_long_text)))
            if len(pairs) == len(content):
                score = max([max(1/math.exp(calculate_ssim(image, pre_image)-1), 1/math.exp(calculate_rouge_l(text, pre_text)-1)) for (image, text, long_text), (pre_image, pre_text, pre_long_text) in pairs])
            else:
                score = _calculate_diff_score(content, pre_content)
        else:
            score = _calculate_diff_score(content, pre_content)
        return score

        
    iTrade_content_df = pd.read_csv('content/processed/iTrade_content.csv', encoding='utf-8', parse_dates=['Date'])
    Wealth_content_df = pd.read_csv('content/processed/Wealth_content.csv', encoding='utf-8', parse_dates=['Date'])
    
    spending_df = pd.read_csv('spending/processed/iTrade_Wealth_PivotTable_Weekly.csv', parse_dates=['Week'])
    
    funnel_df = pd.read_csv('funnel/processed/Funnel_Weekly.csv', parse_dates=['Week'])
    
    iTrade_page_view_df = pd.read_csv('content/processed/iTrade_page_views.csv', encoding='utf-8', parse_dates=['Week'])
    Wealth_page_view_df = pd.read_csv('content/processed/Wealth_page_views.csv', encoding='utf-8', parse_dates=['Week'])
    
    # caption
    iTrade_image_feature_df = pd.read_csv('content/processed/iTrade_image_features.csv', encoding='utf-8')
    Wealth_image_feature_df = pd.read_csv('content/processed/Wealth_image_features.csv', encoding='utf-8')
    
    
    iTrade_reg_htmls = OrderedDict()
    Wealth_reg_htmls = OrderedDict()
    
    start_dt = datetime(2020, 5, 3)
    end_dt = datetime(2022, 4, 1)
    
    for business_unit in ['iTrade', 'Wealth']:
    
        print("Business Unit:", business_unit)
    
        records = []
        content_df = iTrade_content_df if business_unit == 'iTrade' else Wealth_content_df
        page_view_df = iTrade_page_view_df if business_unit == 'iTrade' else Wealth_page_view_df
        # caption
        image_feature_df = iTrade_image_feature_df if business_unit == 'iTrade' else Wealth_image_feature_df
        reg_htmls = iTrade_reg_htmls if business_unit == 'iTrade' else Wealth_reg_htmls
        
        
        for idx, row in content_df.iterrows():
            html = re.sub(r'\d{14}__http(s)?____', '', row["HTML"].split('/')[-1])
            #html_dt = datetime.strptime(re.search(r'\d{14}', row["HTML"].split('/')[-1]).group(), "%Y%m%d%H%M%S")
            if row['Date'] <= start_dt:
                if html not in reg_htmls:
                    reg_htmls[html] = ('content/'+row["HTML"], [])
                if reg_htmls[html][0] == 'content/'+row["HTML"]:
                    reg_htmls[html][1].append(('content/'+row["Image"] if not is_nan(row["Image"]) else "", row["Text"] if not is_nan(row["Text"]) else "", row["Long Text"] if not is_nan(row["Long Text"]) else ""))
            else:
                if html not in reg_htmls:
                    reg_htmls[html] = ('', [])
        dt = start_dt
        step = timedelta(days=7)
        while dt < end_dt:
        
            print("Week:", dt)
        
            spending_record = spending_df.loc[(spending_df['Week']>=dt) & (spending_df['Week']<dt + step) & (spending_df['Business Unit']==business_unit)].iloc[0]
            funnel_record = funnel_df.loc[(funnel_df['Week']>=dt) & (funnel_df['Week']<dt + step) & (funnel_df['Business Unit']==business_unit)].iloc[0]
        
            content_records = content_df.loc[(content_df['Date']>=dt) & (content_df['Date']<dt + step)]
            html2content = OrderedDict()
            for idx, row in content_records.iterrows():
                html = re.sub(r'\d{14}__http(s)?____', '', row["HTML"].split('/')[-1])
                if html not in html2content:
                    html2content[html]=OrderedDict()
                if row["HTML"] not in html2content[html]:
                    html2content[html][row["HTML"]] = []
                html2content[html][row["HTML"]].append(('content/'+row["Image"] if not is_nan(row["Image"]) else "", row["Text"] if not is_nan(row["Text"]) else "", row["Long Text"] if not is_nan(row["Long Text"]) else ""))
                
            
            for html in html2content:    

                if len(html2content[html])>1:
                    max_score = 0
                    max_idx = -1
                    for idx, raw_html in enumerate(html2content[html]):
                    
                        score = calculate_diff_score(html2content[html][raw_html], reg_htmls[html][1])
                        
                        if score > max_score:
                            max_score = score
                            max_idx = idx

                    raw_html = list(html2content[html].keys())[max_idx]
                    html2content[html] = html2content[html][raw_html]
                    
                else:
                    raw_html = list(html2content[html].keys())[0]
                    html2content[html] = html2content[html][raw_html]
            
                reg_htmls[html]=('content/'+raw_html, html2content[html])
                

            record = OrderedDict()
            
            
            record['Week'] = dt
            record['Business Unit'] = business_unit 
            record['Digital'] = spending_record['Digital']
            record['DigitalOOH'] = spending_record['DigitalOOH']
            record['OOH'] = spending_record['OOH']
            record['Print'] = spending_record['Print']
            record['Radio'] = spending_record['Radio']
            record['SEM'] = spending_record['SEM']
            record['SEM 2'] = spending_record['SEM2']
            record['Social'] = spending_record['Social']
            record['TV'] = spending_record['TV']
            record['Awareness'] = funnel_record['Awareness']
            record['Consideration'] = funnel_record['Consideration']
            record['Purchase'] = funnel_record['Purchase']
            
            
            for i, html in enumerate(reg_htmls):
                record['HTML %d'%(i+1)] = reg_htmls[html][0]
                html_criterion_value = (re.sub('\d{14}__http(s)?____', 'https://', reg_htmls[html][0].split('/')[-1]).replace('__','/'))[:100]
                
                page_view_rows = page_view_df.loc[(page_view_df['HTML'] == html_criterion_value) & (page_view_df['Week'] == dt)]
                '''
                if not page_view_rows.empty and len(page_view_rows) > 1:
                    print(html_criterion_value)
                    print(page_view_rows)
                '''
                if not page_view_rows.empty:
                    if business_unit == 'iTrade':
                        record['HTML %d Page Views'%(i+1)] = page_view_rows['Page Views'].values[0]
                        record['HTML %d Visits'%(i+1)] = page_view_rows['Visits'].values[0]
                        record['HTML %d Unique Visitors'%(i+1)] = page_view_rows['Unique Visitors'].values[0]
                    else:
                        record['HTML %d Unique Visitors'%(i+1)] = page_view_rows['Unique Visitors'].values[0]
                
                for j, content in enumerate(reg_htmls[html][1]):
                    record['Image %d-%d'%(i+1, j+1)] = content[0]
                    
                    # caption
                    image_feature_rows = image_feature_df.loc[(image_feature_df['Image'] == content[0].replace('content/',''))]
                    if not image_feature_rows.empty:
                        record['Image Caption %d-%d'%(i+1, j+1)] = image_feature_rows['Caption'].values[0]
                    
                    record['Text %d-%d'%(i+1, j+1)] = content[1]
                    record['Long Text %d-%d'%(i+1, j+1)] = content[2]
            

            records.append(record)
        
            dt = dt + step
    
        columns = ['Week', 'Business Unit', 'Digital', 'DigitalOOH', 'OOH', 'Print', 'Radio', 'SEM', 'SEM 2', 'Social', 'TV', 'Awareness', 'Consideration', 'Purchase']
        for i, html in enumerate(reg_htmls):
            columns.append('HTML %d'%(i+1))
            if business_unit == 'iTrade':
                columns.append('HTML %d Page Views'%(i+1))
                columns.append('HTML %d Visits'%(i+1))
                columns.append('HTML %d Unique Visitors'%(i+1))
            else:
                columns.append('HTML %d Unique Visitors'%(i+1))
            for j in range(4):
                columns.append('Image %d-%d'%(i+1, j+1))
                # caption
                columns.append('Image Caption %d-%d'%(i+1, j+1))
                
                columns.append('Text %d-%d'%(i+1, j+1))
                columns.append('Long Text %d-%d'%(i+1, j+1))
    
        dataset_df = pd.DataFrame.from_records(records, columns=columns)
        dataset_df = dataset_df.dropna(how='all', axis=1)
        with codecs.open('%s_dataset.csv'%business_unit, 'w', 'utf-8') as csv_file:
            dataset_df.to_csv(csv_file, index=False, lineterminator='\n')

consolidate_dataset()

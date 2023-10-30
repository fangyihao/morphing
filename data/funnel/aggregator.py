import pandas as pd
import os
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd

start_dt = datetime.strptime('03-29-2020', "%m-%d-%Y")
end_dt = datetime.strptime('09-01-2022', "%m-%d-%Y")

records = []


def consolidate_iTrade(start_dt, end_dt):
    records = []

    dt = start_dt
    while dt < end_dt:
        #print(dt.strftime("%m-%d-%Y"))
        week = dt.strftime("%m-%d-%Y")
        
        awareness = 0
        consideration = 0
        purchase = 0
        
        awareness_anchor_ln = -100
        consideration_anchor_ln = -100
        purchase_anchor_ln = -100

        with open('raw/iTrade_Dashboard_Weekly_%s.csv'%dt.strftime("%m-%d-%Y"), 'r', encoding='utf-8') as f:
            pre_lines = []
            for i, line in enumerate(f.readlines()):
                if len(pre_lines) == 2:
                    if '##############################################\n# URLs Visited\n##############################################' in ''.join(pre_lines) + line:
                        awareness_anchor_ln = i
                    elif '##############################################\n# App Starts\n##############################################' in ''.join(pre_lines) + line:
                        #print('App Starts')
                        consideration_anchor_ln = i
                    elif '##############################################\n# App Completions\n##############################################' in ''.join(pre_lines) + line:
                        #print('App Completions')
                        purchase_anchor_ln = i
                
                #if 'https://www.scotiaitrade.com/en/home.html' in line and i == awareness_anchor_ln + 3:
                if 'Page URL (c19)' in line:
                    awareness += int(line.split(',')[2]) # Visits
                elif 'Page' in line and i == consideration_anchor_ln + 3:
                    consideration += int(line.split(',')[1])
                elif 'Page' in line and i == purchase_anchor_ln + 3:
                    purchase += int(line.split(',')[1])
            
                pre_lines.append(line)
                if len(pre_lines) > 2:
                    pre_lines.pop(0)
        records.append({'Week': week, 'Business Unit': 'iTrade', 'Awareness': awareness, 'Consideration': consideration, 'Purchase': purchase})
        dt += timedelta (days=7)
    return records
    



def consolidate_Wealth(start_dt, end_dt):
    records = []

    dt = start_dt
    while dt < end_dt:
        #print(dt.strftime("%m-%d-%Y"))
        week = dt.strftime("%m-%d-%Y")
        
        awareness = 0
        consideration = 0
        purchase = 0

        with open('raw/Wealth_Dashboard_Weekly_%s.csv'%dt.strftime("%m-%d-%Y"), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if 'Page URL (c19)' in line:
                #if 'https://www.scotiawealthmanagement.com/ca/en/home.html' in line \
                #    or 'https://www.scotiawealthmanagement.com/intl/en/home.html' in line \
                #    or 'https://www.scotiawealthmanagement.com/ca/en.html' in line \
                #    or 'https://www.scotiawealthmanagement.com/ca/fr.html' in line \
                #    or 'https://www.scotiawealthmanagement.com/intl/en/home.html' in line:
                    awareness += int(line.split(',')[1])
                elif 'https://www.scotiawealthmanagement.com/ca/en/home/not-yet-a-client/contact-us.html' in line \
                    or 'https://www.scotiawealthmanagement.com/ca/en/connect-with-us.html' in line \
                    or 'https://www.scotiawealthmanagement.com/intl/en/home/contact-us.html' in line:
                    #print(line.split(',')[1])
                    consideration += int(line.split(',')[1])
                elif 'https://www.scotiawealthmanagement.com/ca/en/home/not-yet-a-client/thank-you.html' in line \
                    or 'https://www.scotiawealthmanagement.com/ca/en/connect-with-us/thank-you.html' in line:
                    purchase += int(line.split(',')[1])
                    
        records.append({'Week': week, 'Business Unit': 'Wealth', 'Awareness': awareness, 'Consideration': consideration, 'Purchase': purchase})
        dt += timedelta (days=7)
    return records

records.extend(consolidate_iTrade(start_dt, end_dt))
records.extend(consolidate_Wealth(start_dt, end_dt))
df = pd.DataFrame.from_records(records)
df.to_csv('processed/Funnel_Weekly.csv', index=False)
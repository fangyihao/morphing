'''
Created on Jan. 9, 2023

@author: yfang
'''
import pandas as pd
import numpy as np
from collections import OrderedDict


attr_decode_subwords = True
sheet_names = ["All", "Spending", "Web Content", "Digital", "DigitalOOH", "OOH", "Print", "Radio", "SEM", "SEM 2", "Social", "TV", "Image", "Text"]
    

def token_n_scale2scale2token(tokens, scales):
    scale2token = {}
    for token, scale in zip(tokens, scales):
        #if scale != 0:
        if scale not in scale2token:
            scale2token[scale] = OrderedDict()
        if token not in scale2token[scale]:
            scale2token[scale][token] = 0
        scale2token[scale][token] += 1
    return scale2token

def scale2token2df(scale2token):
    dfs = []
    for scale in sorted(scale2token):
        token_n_count = scale2token[scale]
        token_n_count = np.array(list(token_n_count.items())).transpose()
        pd_data = {}
        pd_data[scale] = token_n_count[0]
        pd_data[str(int(scale)) + " Count"] = token_n_count[1]
        dfs.append(pd.DataFrame(pd_data))
    df = pd.concat(dfs, ignore_index=False, axis=1)
    return df

 

writer = pd.ExcelWriter('%s_hist.xlsx'%('word' if attr_decode_subwords else 'subword'), engine='xlsxwriter')
for sheet_name in sheet_names:
    df = pd.read_excel('%s_n_attr.xlsx'%('word' if attr_decode_subwords else 'subword'), sheet_name=sheet_name, usecols=["Word" if attr_decode_subwords else "Subword", "Attribution Scale"])
    df = scale2token2df(token_n_scale2scale2token(df["Word" if attr_decode_subwords else "Subword"].values, df["Attribution Scale"].values))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
writer.save()


writer = pd.ExcelWriter('phrase_hist.xlsx', engine='xlsxwriter')
for sheet_name in sheet_names[3:12]:
    df = pd.read_excel('phrase_n_attr.xlsx', sheet_name=sheet_name, usecols=["Phrase", "Attribution Scale"])
    df = scale2token2df(token_n_scale2scale2token(df["Phrase"].values, df["Attribution Scale"].values))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
writer.save()


'''
Created on Dec. 23, 2022

@author: yfang
'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from ordered_set import OrderedSet
'''
caption_df = pd.read_csv('data/content/processed/iTrade_image_features.csv', usecols = ["Caption"])
caption_df = caption_df.drop_duplicates(subset=["Caption"])

print(caption_df["Caption"].values)

spending_df = pd.read_excel('word_n_attr.xlsx', sheet_name='Spending')
#spending_df = pd.read_csv("spending_word_n_attr.csv")
print(spending_df)
#sns.displot(spending_df)
#plt.title("Spending Attribution Score Histogram and Estimation")
#plt.show()
#plt.savefig("Spending_Attribution_Score_Histogram_and_Estimation.png")
#plt.clf()     

web_content_df = pd.read_excel('word_n_attr.xlsx', sheet_name='Web Content')
#web_content_df = pd.read_csv("web_content_word_n_attr.csv")
print(web_content_df)


palette = ["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"]

fig, ax1 = plt.subplots(figsize=(12,9))
sns.histplot(pd.concat([spending_df, web_content_df], ignore_index=True, axis=1), element="step", fill=False, kde=True, palette=sns.color_palette(palette, 14))

#plt.title("Attribution Score Histogram and Estimation")
ax1.set_title("Attribution Score Histogram and Estimation")
ax1.set_xlabel("Attribution Score")
ax1.set_ylabel("Count")
legend = ax1.get_legend()
handles = legend.legendHandles
legend.remove()
ax1.legend(handles, ["Advertisement Spending", "Web Content"])


axins = zoomed_inset_axes(ax1, 5, loc='center left')
sns.histplot(pd.concat([spending_df, web_content_df], ignore_index=True, axis=1), element="step", fill=False, kde=True, palette=sns.color_palette(palette, 14))
axins.set_xlim(-0.05, 0.05)
axins.set_ylim(0, 50)
#plt.xticks(visible=False)
plt.yticks(visible=False)
legend = axins.get_legend()
handles = legend.legendHandles
legend.remove()
axins.set(xlabel=None)
axins.set(ylabel=None)
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.draw()
plt.show()
'''
attr_decode_subwords = True
sheet_names = ["All", "Spending", "Web Content", "Digital", "DigitalOOH", "OOH", "Print", "Radio", "SEM", "SEM 2", "Social", "TV", "Image", "Text"]

def plot_hist_n_est(sheet_names, col_name, palette, size, chart="hist+kde", element="step", inset=False, ins_xlim=None, ins_ylim=None, ins_zoom=None, ins_loc=None, ins_mark_loc=(1,3)):
    if chart == "kde":
        title = "Kernel Density Estimation"
        filename = title.replace(' ', '_')
    elif chart == "hist+kde":
        title = "Histogram and Kernel Density Estimation"
        filename = title.replace(' ', '_')
    elif chart == "cuhist+kde":
        title = "Cumulative Histogram and Kernel Density Estimation"
        filename = title.replace(' ', '_')
    elif chart == "cuhist+density+kde":
        title = "Normalized Cumulative Histogram and Kernel Density Estimation"
        filename = "Normalized Cumulative Histogram (Density) and Kernel Density Estimation".replace(' ', '_')
    elif chart == "cuhist+probability+kde":
        title = "Normalized Cumulative Histogram and Kernel Density Estimation"
        filename = "Normalized Cumulative Histogram (Probability) and Kernel Density Estimation".replace(' ', '_')
    elif chart == "cdf":
        title = "Cumulative Distribution Function"
        filename = title.replace(' ', '_')
    else:
        raise NotImplementedError()
    
    dfs = []
    for sheet_name in sheet_names:
        df = pd.read_excel('%s_n_attr.xlsx'%('word' if attr_decode_subwords else 'subword'), sheet_name=sheet_name, usecols=[col_name])
        dfs.append(df)
    fig, ax = plt.subplots(figsize=size)
    if chart == "kde":
        sns.kdeplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
    elif chart == "hist+kde":
        sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)))
    elif chart == "cuhist+kde":
        sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True)
    elif chart == "cuhist+density+kde":
        sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="density", common_norm=False)
    elif chart == "cuhist+probability+kde":
        sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="probability", common_norm=False)
    elif chart == "cdf":
        sns.ecdfplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
    else:
        raise NotImplementedError()
        
    ax.set_title("%s %s"%(col_name,title))
    ax.set_xlabel(col_name)
    #ax.set_ylabel("Count")
    legend = ax.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax.legend(handles, sheet_names)
    if inset:
        axins = zoomed_inset_axes(ax, ins_zoom, loc=ins_loc)
        if chart == "kde":
            sns.kdeplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
        elif chart == "hist+kde":
            sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)))
        elif chart == "cuhist+kde":
            sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True)
        elif chart == "cuhist+density+kde":
            sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="density", common_norm=False)
        elif chart == "cuhist+probability+kde":
            sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="probability", common_norm=False)
        elif chart == "cdf":
            sns.ecdfplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
        else:
            raise NotImplementedError()
        axins.set_xlim(*ins_xlim)
        axins.set_ylim(*ins_ylim)
        #plt.xticks(visible=False)
        plt.yticks(visible=False)
        legend = axins.get_legend()
        handles = legend.legendHandles
        legend.remove()
        axins.set(xlabel=None)
        axins.set(ylabel=None)
        mark_inset(ax, axins, loc1=ins_mark_loc[0], loc2=ins_mark_loc[1], fc="none", ec="0.5")
    plt.draw()
    #plt.show()
    plt.savefig("%s_%s_%s.png"%(col_name.replace(' ', '_'), filename, ('_'.join(sheet_names)).replace(' ', '_')))
    plt.clf()
    plt.close() 
plot_hist_n_est(sheet_names[1:3], col_name="Attribution Score" ,palette=["#009DD6", "#EC111A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(-0.05, 0.05), ins_ylim=(0, 50), ins_zoom=5, ins_loc="center left")
plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="kde", inset=False)
plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(0.025, 0.125), ins_ylim=(0, 2), ins_zoom=3, ins_loc="center right", ins_mark_loc=(2,4))
plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cuhist+kde", inset=False)
plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cuhist+density+kde", inset=True, ins_xlim=(0, 0.1), ins_ylim=(0.8, 1), ins_zoom=3, ins_loc="center right")
plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cuhist+probability+kde", inset=True, ins_xlim=(0, 0.1), ins_ylim=(0.8, 1), ins_zoom=3, ins_loc="center right")
plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cdf", inset=False)
plot_hist_n_est(sheet_names[12:], col_name="Attribution Score", palette=["#333333", "#EC111A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(-0.05, 0.05), ins_ylim=(0, 50), ins_zoom=5, ins_loc="center left")
plot_hist_n_est(sheet_names[1:2] + sheet_names[12:], col_name="Attribution Score", palette=["#009DD6", "#333333", "#EC111A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(-0.05, 0.05), ins_ylim=(0, 50), ins_zoom=5, ins_loc="center left")
plot_hist_n_est(sheet_names[1:2] + sheet_names[12:], col_name="Attribution Score", palette=["#009DD6", "#333333", "#EC111A"], size=(12,9), chart="cuhist+density+kde", inset=True, ins_xlim=(0, 0.1), ins_ylim=(0.8, 1), ins_zoom=3, ins_loc="center right")

def plot_cat(sheet_names, col_name, palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], chart="phrase-violin"):
    span_type = chart.split('-')[0]
    cat_types = chart.split('-')[1].split('+')
    dfs = []
    for sheet_name in sheet_names:
        df = pd.read_excel('%s_n_attr.xlsx'%span_type, sheet_name=sheet_name, usecols=[col_name])
        df = df.rename(columns={col_name:sheet_name})
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=False, axis=1)
    df = pd.melt(df, var_name="Channel", value_name=col_name)
    
    for cat_type in cat_types:
        if cat_type == 'swarm':
            sns.swarmplot(data=df, x=col_name, y="Channel", size=2, palette=sns.color_palette(["#000000"], len(["#000000"])))   
        elif cat_type == 'violin':
            rel=sns.catplot(data=df, x=col_name, y="Channel", kind=cat_type, palette=sns.color_palette(palette, len(palette)), color=".9", inner=None, height=9, aspect=1)
        else:
            rel=sns.catplot(data=df, x=col_name, y="Channel", kind=cat_type, palette=sns.color_palette(palette, len(palette)), height=9, aspect=1)
            
    rel.fig.suptitle("%s %s %s Chart"%(span_type.title(), col_name, ' and '.join([cat_type.title() for cat_type in cat_types])))
    
    plt.savefig("%s_%s_%s_%s.png"%(span_type.title(), col_name.replace(' ', '_'), '_'.join([cat_type.title() for cat_type in cat_types]), ('_'.join(sheet_names[3:12])).replace(' ', '_')))
    plt.clf()
    plt.close()

plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="phrase-violin")
plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="%s-violin"%('word' if attr_decode_subwords else 'subword'))
plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="phrase-violin")
plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="%s-violin"%('word' if attr_decode_subwords else 'subword'))
plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="phrase-violin+swarm")
plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="%s-violin+swarm"%('word' if attr_decode_subwords else 'subword'))
plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="phrase-violin+swarm")
plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="%s-violin+swarm"%('word' if attr_decode_subwords else 'subword'))
plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="phrase-boxen")
plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="%s-boxen"%('word' if attr_decode_subwords else 'subword'))
plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="phrase-boxen")
plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="%s-boxen"%('word' if attr_decode_subwords else 'subword'))
    
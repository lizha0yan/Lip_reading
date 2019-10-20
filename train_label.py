# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:32:28 2019

@author: 群青雨
"""




import pandas as pd
from numpy import asarray

#import aliall to create label
col_names = range(320)
aliall = pd.read_csv('C:/Users/Lizhaoyan/Desktop/NLP/test/ali.all', delimiter="\s+", names=col_names)
##ignore the wordindicators,edit one by one, may take long
for i in range(len(aliall)):
    for j in range(1,320):
        if str(aliall[j][i]) == 'nan':
            break
        for k in range(len(aliall[j][i])):
            if not aliall[j][i][k].isalpha():
                aliall[j][i] = aliall[j][i][0:k]
                break

##create a map to transform ARPABET to Viseme
mapping = {
        "F":"A","V":"A",
        "ER":"B","OW":"B","R":"B","Q":"B","W":"B","UH":"B","UW":"B","AXR":"B","UX":"B",
        "B":"C","P":"C","M":"C","EM":"C",
        "AW":"D",
        "DH":"E","TH":"E",
        "CH":"F","JH":"F","SH":"F","ZH":"F",
        "OY":"G",
#        "AO":"G",
        "S":"H","Z":"H",
        "AA":"I","AE":"I","AH":"I","AY":"I","EH":"I","EY":"I","IH":"I","IY":"I","Y":"I","AO":"I","AXH":"I","AX":"I","IX":"I",
        "D":"J","L":"J","N":"J","T":"J","EL":"J","NX":"J","EN":"J","DX":"J",
        "G":"K","K":"K","NG":"K","ENG":"K","HH":"K",
        "SIL":"S",
        "nan" : ""
        }

Bozkurt_Map = {
        "SIL":"S",
        "AY":"V2","AH":"V2",
        "EY":"V3","EH":"V3","AE":"V3",
        "ER":"V4",
        "IX":"V5","IY":"V5","IH":"V5","AX":"V5","AXR":"V5","Y":"V5",
        "UW":"V6","UH":"V6","W":"V6",
        "AO":"V7","AA":"V7","OY":"V7","OW":"V7",
        "AW":"V8",
        "G":"V9","HH":"V9","K":"V9","NG":"V9",
        "R":"V10",
        "L":"V11","D":"V11","N":"V11","EN":"V11","EL":"V11","T":"V11",
        "S":"V12","Z":"V12",
        "CH":"V13","SH":"V13","JH":"V13","ZH":"V13",
        "TH":"V14","DH":"V14",
        "F":"V15","V":"V15",
        "M":"V16","EM":"V16","B":"V16","P":"V16",
        "nan" : ""
        }
for j in range(len(aliall)):
    for i in range(1,320):
        if aliall[i][j] == "":
            break
        aliall[i][j] = Bozkurt_Map[str(aliall[i][j])]
aliall.to_csv("Label_Boakurt") 



#!/usr/bin/env python
# coding: utf-8

import pandas as pd

data1=pd.read_csv('WikiQA-train.tsv',delimiter='\t',encoding='utf-8')
data2=pd.read_csv('WikiQA-test.tsv',delimiter='\t',encoding ='utf-8')


data = data1.append(data2, ignore_index=True)


#Extracting the unique questions along with their questionID
def extract_questions(data):
    new_data=data.drop(['DocumentID','DocumentTitle','Label'],axis=1)
    d=new_data.drop_duplicates()
    return d

d=extract_questions(data)
d.to_csv('questions.csv',index=False)

import os
import pandas as pd
from mysql.connector import connect
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np


def draw_word_cloud(text):
    wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='white',
    stopwords=STOPWORDS).generate(str(text))
    fig = plt.figure(
        figsize=(40, 30),
        facecolor='k',
        edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    
mydb = connect(
  host="localhost",
  user="root",
  passwd="",
  database="mapbuddydb"
)
mycursor = mydb.cursor()


def get_building_id(texts):
    text = "SELECT location_id FROM location where location_type = 3 and parent_id <> 370 and location_name like '" + texts + " %'"
    mycursor.execute(text)
    myresult = mycursor.fetchall()
    result = 0
    if len(myresult) > 1:
        print(text)
    for y in myresult:
        result = y[0] 
#     mydb.close()
    if texts == '404' :
        print(text)
    return result


def get_level_id(raw):
    text = "SELECT location_id FROM location where location_type = 4 and parent_id = " + str(raw['MB_BuildingId']) + " and location_name = '" + str(raw['Floor']) + "'"
    mycursor.execute(text)
    myresult = mycursor.fetchall()
    result = 0
    if len(myresult) > 1:
        print(text)
    for y in myresult:
        result = y[0] 
#     mydb.close()
    return result


def set_value(row_number, assigned_value): 
    return assigned_value[row_number] 


# training_filepaths = get_filepaths('NMMUXLSs/all')
df = pd.DataFrame()
fname = "merged.csv"
df = pd.read_csv(fname, header=0)
df['Campus'].replace('', np.nan, inplace=True)
df.dropna(subset=['Campus'], inplace=True)
event_dictionary = {11.0 : 1334, 1.0 : 369, 7.0 : 368, 2.0 : 371, 1.0 : 369, 3.0 : 370, 4.0 : 361, 5.0 : 367}
df['MB_CampusId'] = df['Campus'].apply(set_value, args=(event_dictionary,))
df['Building'] = pd.to_numeric(df['Building'], downcast='integer')
df['Building'] = df['Building'].astype(str)
df['Floor'] = pd.to_numeric(df['Floor'], downcast='integer')
df['MB_BuildingId'] = df['Building'].apply(get_building_id)
df['MB_LevelId'] = df.apply (lambda row: get_level_id(row), axis=1)
df['Assign Space'].fillna(0)
df['Non Assign Space'].fillna(0)
df['json'] = df.apply(lambda x: x.to_json(), axis=1)
df.to_csv(r'mergedIds.csv')
mydb.close()
print(df)

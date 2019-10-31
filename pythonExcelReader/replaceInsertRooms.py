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


def insert_location(location_name, location_label, location_type, parent_id, description):
    query = "INSERT INTO location(location_name, location_label, location_type, parent_id, description, client_id) " \
            "VALUES(%s,%s,%s,%s,%s,4)"
    args = (location_name, location_label, location_type, parent_id, description)
    try:
        conn = connect(
              host="localhost",
              user="root",
              passwd="",
              database="mapbuddydb"
            )
        cursor = conn.cursor()
        cursor.execute(query, args)
        if cursor.lastrowid:
            print('last insert id', cursor.lastrowid)
        else:
            print('last insert id not found')
        conn.commit()
    finally:
        cursor.close()
        conn.close()


df = pd.DataFrame()
fname = "mergedIds.csv"
df = pd.read_csv(fname, header=0)
for index, row in df.iterrows():
    if row['MB_LevelId'] == 0 :
        print(row['Room'], str(row['Name']) + " " + str(row['Room']), 7, row['MB_LevelId'], row['json'])
        if str(row['Room']) != 'nan' :
            insert_location(row['Room'], str(row['Name']) + " " + str(row['Room']), 7, row['MB_LevelId'], row['json'])
    
# 1861 location id of the levels

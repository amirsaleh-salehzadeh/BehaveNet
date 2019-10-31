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
    

def get_building_id(text):
    mydb = connect(
      host="localhost",
      user="root",
      passwd="",
      database="mapbuddydb"
    )
    mycursor = mydb.cursor()
    text = "SELECT location_id FROM location where location_type = 3 and parent_id <> 370 and location_name like '" + text + " %'"
    mycursor.execute(text)
    myresult = mycursor.fetchall()
    result = 0
    if len(myresult) >1:
        print(text)
    for y in myresult:
        result = y[0] 
    mydb.close()
    return result

def get_level_id(text):
    mydb = connect(
      host="localhost",
      user="root",
      passwd="",
      database="mapbuddydb"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT location_id FROM location where location_type = 3 and location_name like '%" + text + " %'")
    myresult = mycursor.fetchall()
    result = 0
    for y in myresult:
        result = y[0]
    mydb.close()
    return result
    
def get_filepaths(mainfolder):
    training_filepaths = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "file" not in folder:
            filenames = os.listdir(fpath)
            for filename in filenames:
                fullpath = fpath + "/" + filename
                if "file" not in filename:
                    training_filepaths[fullpath] = folder
    return training_filepaths


def insert_location(location_name, location_label, location_type, parent_id):
    query = "INSERT INTO location(location_name, location_label, location_type, parent_id, client_id) " \
            "VALUES(%s,%s,%s,%s,4)"
    args = (location_name, location_label, location_type, parent_id)
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
        
        
# training_filepaths = get_filepaths('NMMUXLSs/all')
df = pd.DataFrame()
entries = os.listdir('NMMUXLSs/all - Copy')
files_xls = [f for f in entries if f[-3:] == 'xls']
p = 0
for f in files_xls:
    fname = "NMMUXLSs/all - Copy/" + f
    data = pd.read_excel(fname, sheet_name=0, skiprows=[0], names=["Campus", "Building", "Floor", "Room", "Name", "Remarks", "Bar", "Cat", "Dept", "Assign Space", "Non Assign Space", "Stations"], dtype='str')
    df = df.append(data)
    p = p + 1
df['Name'] = df['Name'].str.lower()
df['Campus'].replace('', np.nan, inplace=True)
# df.dropna(subset=['Campus'], inplace=True)
df.to_csv(r'merged.csv')
print(len(df['Name'].unique().tolist()))

# a = df.groupby(['Building', 'Floor']).nunique(False)
# for index, row in a.iterrows():
#     id = get_building_id(index[0])
#     if id > 0:
#         print(id ," ", index[1])
#         insert_location(index[1], "Level "+index[1], 4, id)


# for x in myresult:
#     newcursor = mydb.cursor()
#     if(x[0].split(' ')[0] != 9):
#         newcursor.execute("SELECT location_id FROM location where location_type = 3 and parent_id = 369 and location_name like '%" +x[0].split(' ')[0]+ " %'")
#         newresult = newcursor.fetchall()
#         for y in newresult:
#             print(y[0]) 
#     print(df.loc[df['Building'] == int(x[0].split(' ')[0])]) 

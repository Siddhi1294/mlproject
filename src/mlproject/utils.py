import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import psycopg2

import pickle
import numpy as np


load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    logging.info(user)

    try:
        mydb = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=db
        )

        cur = mydb.cursor()
        cur.execute('select * from school')
        #df = pd.DataFrame(cur.fetchall())



        logging.info("Connection Established",mydb)
        
        df=pd.read_sql_query('select * from school',mydb)
        print(df)

        return df

    except Exception as ex:
        raise CustomException(ex)
    

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)



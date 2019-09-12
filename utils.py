import csv
import os
import pandas as pd
import errno
import sqlite3


def load_n_debug(csv_files_path,drop_percent=0.985):
    """Creates a dataset as a pandas.DataFrame object from the csv_files in given 'csv_files_path' and debugs the data in the datasets

    Arguments:
        csv_files_path: os path of the csv files
        drop_percent: percentage value that is used to handle missing data in the dataset

    Returns:
        A Debugged dataset
    """

    path = csv_files_path
    files = os.listdir(path)

    data = pd.DataFrame(columns=["timestamp"])
    for file in files:
        data = data.merge(pd.read_csv(os.path.join(path,file)),how="outer",on="timestamp")
    data["timestamp"] = data["timestamp"].astype("int") 


    data = data.reset_index(drop=True)

    if not len(data.columns[data.columns.duplicated()].unique()) == 0:
        print("Handling columns with the same name...")
        data.columns = pd.io.parsers.ParserBase({'names':data.columns})._maybe_dedup_names(data.columns)
        print("Completed!")
    
    if not len(list(data["timestamp"][data["timestamp"].duplicated()].index)) == 0:
        print("Handling duplicated rows...")
        data = data.drop(index = list(data["timestamp"][data["timestamp"].duplicated()].index))
        data = data.reset_index(drop=True)
        print("Completed!")

    print("handling the missing data...")
    data = dropna(data,drop_percent)
    print("Completed!")

    return data


def csv2sqlite(csv_files_path,data_path,db_name,tbl_name):
    """Converts csv files to sqlite files 
    
    Arguments:
        csv_files_path: os path of the csv files
        data_path: os path to write the sqlite database
        db_name: Name of the database
        tbl_name: Name of the database table
    """

    dir = os.path.dirname(os.path.abspath(__file__))

    data = load_n_debug(csv_files_path)
    
    if not os.path.exists(os.path.join(dir,"data")):
        try:
            os.makedirs(os.path.join(dir,"data"),0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    print("Converting dataFrame to SQLite...")
    df2sqlite(data,data_path,db_name,tbl_name)
    print("Completed!")    
    # return data

def dropna(dataframe,percent=0.985):
    """Handles missing data by dropping columns that has more percantage of NaN values than the given 'percent'
    
    Arguments:
        dataframe: dataset to be handled
        percent: percentage value that is used to handle missing data in the dataset

    Returns:
        A dataset
    """
    
    total = dataframe.isnull().sum().sort_values(ascending=False)
    percentage = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
    null_cols = list(missing_data[missing_data["Percent"] > percent].index)    
    dataframe = dataframe.drop(columns = null_cols)
    return dataframe
    
def df2sqlite(dataframe, db_path, db_name, tbl_name):
    """Converts a pandas.DataFrame object to sqlite file

    Arguments:
        dataframe: dataset
        data_path: os path to write the sqlite database
        db_name: Name of the database
        tbl_name: Name of the database table        
    """

    conn=sqlite3.connect(os.path.join(db_path,db_name))
    cur = conn.cursor()                                 
 
    wildcards = ','.join(['?'] * len(dataframe.columns))              
    data = [tuple(x) for x in dataframe.values]
 
    cur.execute("drop table if exists %s" % tbl_name)
 
    col_str = '"' + '","'.join(dataframe.columns) + '"'
    cur.execute("create table %s (%s)" % (tbl_name, col_str))
 
    cur.executemany("insert into %s values(%s)" % (tbl_name, wildcards), data)
 
    conn.commit()
    conn.close()


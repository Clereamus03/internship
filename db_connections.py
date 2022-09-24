
import pandas as pd

class DatabaseOperations:

    # def __init__(self, database_type):
    #     self.database_type = database_type
    
    def mongo_database_connection(self, host_address, port, database_name, username = None, password = None):

        import pymongo
        try:
            if username == None:
                CONNECTION_STRING = "mongodb://{host_address}:{port}/{database}".format(host_address=host_address, port=port, database=database_name)
                
            else:
                CONNECTION_STRING = "mongodb://{username}:{password}@{host_address}:{port}/{database}".format(username=username,
                                        password=password, host_address=host_address, port=port, database=database_name)

            client = pymongo.MongoClient(CONNECTION_STRING)
            print("Connection Established!")
            return client

        except Exception as e:
            print("Something went wrong!") # Later, replaced with log files
            print(e)
 
    def redis_database_connection(self, host, db, password, port= 6379):

        import redis
        try:
            connection = redis.StrictRedis(host=host, password=password, port=port, db=db)
            print("Connection Established!")
            return connection

        except Exception as e:
            print("Something went wrong!") # Later, replaced with log files
            print(e)

    def mysql_database_connection(self, driver, server, database_name, username, password):
        
        import pyodbc
        try:
            CONNECTION_STRING = 'DRIVER={'+driver+'};SERVER='+server+';DATABASE='+database_name+';UID='+username+';PWD='+password

            connection = pyodbc.connect(CONNECTION_STRING)
            print("Connection Established!")
            return connection

        except Exception as e:
            print("Something went wrong!") # Later, replaced with log files
            print(e)

    def postgres_database_connection(self, host, database_name, username, password):
        
        import psycopg2
        try:
            connection = psycopg2.connect(dbname=database_name, user=username, password=password, host=host)
            print("Connection Established!")
            return connection

        except Exception as e:
            print("Something went wrong!") # Later, replaced with log files
            print(e)

    def make_csv_from_mongo(self, collection_name,host_address, port, database_name, username = None, password = None):
        
        client = self.mongo_database_connection(host_address=host_address, port=port, database_name=database_name, username=username,
                                                password=password)
        mydb = client[database_name]
        col = mydb[collection_name]
        all_records = col.find()

        df = pd.DataFrame(list(all_records))
        df = df.drop(["_id"], axis=1)
        return df


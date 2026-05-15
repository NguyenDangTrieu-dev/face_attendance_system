import psycopg2
from config import Config

def get_db_connection():
    return psycopg2.connect(
        host=Config.DB_CONFIG['host'],
        port=Config.DB_CONFIG['port'],
        dbname=Config.DB_CONFIG['database'],
        user=Config.DB_CONFIG['user'],
        password=Config.DB_CONFIG['password']
    )
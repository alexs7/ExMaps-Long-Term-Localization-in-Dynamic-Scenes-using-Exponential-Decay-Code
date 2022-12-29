import sqlite3
from sqlite3 import Error
import sys
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    @staticmethod
    def blob_to_array(blob, dtype, shape=(-1,)):
        if IS_PYTHON3:
            return np.fromstring(blob, dtype=dtype).reshape(*shape)
        else:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    @staticmethod
    def create_matrix_db_lamar(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                col_idx INTEGER NOT NULL,
                                                col BLOB NOT NULL
                                            );"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute(sql_drop_table_if_exists)
            conn.commit()
            conn.execute(sql_create_data_table)
            conn.commit()
            return conn
        except Error as e:
            print(e)

    # def add_matrix_col(self, idx, col): #col = np.float64
    #     self.execute("INSERT INTO data VALUES (?, ?)", (idx,) + (col,))
    #
    # def get_matrix_col(self, idx):
    #     row = self.execute("SELECT col, cols, data FROM data WHERE col_idx = " + "'" + str(idx) + "'").fetchone()
    #     row = self.blob_to_array(row, np.float64)
    #     return row

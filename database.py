import pyodbc
import os
from datetime import datetime

database = "face_recognition"

def get_connection():
    connection = pyodbc.connect(
        r'DRIVER={SQL Server};SERVER=localhost\SMART;UID=sa;PWD=smart',
        autocommit=True 
    )

    return connection.cursor()

def intialise_database(database_pointer):
    path = "Data\\Train"

    try:
        result = database_pointer.execute(f"SELECT name FROM sys.databases WHERE name = '{database}'")

        db_name = "face_recognition"
        database_pointer.execute(f"SELECT name FROM sys.databases WHERE name = '{db_name}'")
        if database_pointer.fetchone() is None:
            

            print("Creating database...")
            database_pointer.execute(f"CREATE DATABASE {db_name};")
            database_pointer.execute(f"""
            use {db_name};                

            CREATE TABLE users(
                user_id VARCHAR(100) PRIMARY KEY,
                department VARCHAR(100),
                face_image_path VARCHAR(300) NOT NULL
            );
                                    
            CREATE TABLE attendance(
                attendance_id INT PRIMARY KEY IDENTITY(1,1),
                entry_date_time DATETIME NOT NULL, 
                exit_date_time DATETIME NOT NULL DEFAULT '1900-01-01 00:00:00',
                user_id VARCHAR(100),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
                                                        
                                    """)
            

            if(os.path.exists(path)):
                dir_list = os.listdir(path)
                
                for file in dir_list:
                    
                    try:
                        database_pointer.execute(
                            "INSERT INTO users (user_id, department, face_image_path) VALUES (?, ?, ?)",
                            (file, None, os.path.join(path, file))
                        )
                    except Exception as e:
                        print(e)
                        print(f"Fail to initialise database")

    except Exception as e:
        print(e)
        print(f"Fail to initialise database")
        exit() 
            

    
def insert_user(database_pointer, name, department, image_path):
    database_pointer.execute("USE face_recognition")
    database_pointer.execute(
        "INSERT INTO users (user_id, department, face_image_path) VALUES (?, ?, ?)",
        (name, department, image_path)
    )


def insert_attendance(database_pointer, entry_date_time, user_id):
    database_pointer.execute("USE face_recognition")
    database_pointer.execute(
        "INSERT INTO attendance (entry_date_time, user_id) VALUES (?, ?)",
        (entry_date_time , user_id)
    )

def update_attendance(database_pointer, exit_date_time, user_id):
    database_pointer.execute("USE face_recognition")
    database_pointer.execute("""
        UPDATE attendance
        SET exit_date_time = ?
        WHERE exit_date_time = ? AND user_id = ?
    """, (exit_date_time, '1900-01-01 00:00:00', user_id))

def unfinished_attendance(database_pointer, user_id):
    database_pointer.execute("USE face_recognition")
    database_pointer.execute("""
        SELECT * FROM attendance 
        WHERE user_id = ? AND exit_date_time = ?
    """, (user_id, '1900-01-01 00:00:00'))

    return database_pointer.fetchone() is not None

def user_exists(database_pointer, user_id):
    database_pointer.execute("USE face_recognition")
    database_pointer.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    return database_pointer.fetchone() is not None
import pyodbc
import os
from datetime import datetime

database = "face_recognition"

def get_connection():
    try: 
    
        connection = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost\\SQLEXPRESS;"
            "UID=sa;"
            "PWD=123456;"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
            "Connection Timeout=5;"
        )           
        connection.autocommit = True
        return connection.cursor()
    except Exception as e:
        # print(e)
        exit()


def intialise_database(database_pointer):
    path = "Data\\Train"

    try:
        result = database_pointer.execute(f"SELECT name FROM sys.databases WHERE name = '{database}'")

        db_name = "face_recognition"
        database_pointer.execute(f"SELECT name FROM sys.databases WHERE name = '{db_name}'")
        if database_pointer.fetchone() is None:
            

            database_pointer.execute(f"CREATE DATABASE {db_name};")
            database_pointer.execute(f"""
            use {db_name};

            CREATE TABLE admin(
                admin_id VARCHAR(100) PRIMARY KEY,
                password VARCHAR(100) NOT NULL,  
                created_date DATETIME NOT NULL
            )                

            CREATE TABLE users(
                user_id VARCHAR(100) PRIMARY KEY,
                department VARCHAR(100),
                face_image_path VARCHAR(300) NOT NULL,
                created_by VARCHAR(100) NOT NULL,
                FOREIGN KEY (created_by) REFERENCES admin(admin_id)
            );
                                    
            CREATE TABLE attendance(
                attendance_id INT PRIMARY KEY IDENTITY(1,1),
                entry_date_time DATETIME NOT NULL, 
                exit_date_time DATETIME NOT NULL DEFAULT '1900-01-01 00:00:00',
                user_id VARCHAR(100),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
                                                        
                                    """)
            
            try: 
                database_pointer.execute(f"INSERT INTO admin (admin_id, password, created_date) VALUES (?, ?, ?)", "admin", "admin", datetime.now())
            except Exception as e:
                return False, e          

            if(os.path.exists(path)):
                dir_list = os.listdir(path)
                
                for file in dir_list:
                    
                    try:
                        database_pointer.execute(
                            "INSERT INTO users (user_id, department, face_image_path, created_by) VALUES (?, ?, ?)",
                            (file, None, os.path.join(path, file))
                        )
                    except Exception as e:
                        return False, e

        return True, None
    except Exception as e:
        return False, e
        
    
def insert_user(database_pointer, name, department, image_path, created_by):
    try:
        database_pointer.execute(f"USE {database}")
        database_pointer.execute(
            "INSERT INTO users (user_id, department, face_image_path, created_by) VALUES (?, ?, ?, ?)",
            (name, department, image_path, created_by)
        )
        return True, "User added successfully!"
    except Exception as e:
        return False, e


def insert_attendance(database_pointer, entry_date_time, user_id):
    try:
        database_pointer.execute(f"USE {database}")        
        database_pointer.execute(
            "INSERT INTO attendance (entry_date_time, user_id) VALUES (?, ?)",
            (entry_date_time , user_id)
        )
        return True, None
    except Exception as e:
        return False, e


def update_attendance(database_pointer, exit_date_time, user_id):
    try:
        database_pointer.execute(f"USE {database}")        
        database_pointer.execute("""
            UPDATE attendance
            SET exit_date_time = ?
            WHERE exit_date_time = ? AND user_id = ?
        """, (exit_date_time, '1900-01-01 00:00:00', user_id))
        return True, None
    except Exception as e:
        return False, e

def unfinished_attendance(database_pointer, user_id):
    try:
        database_pointer.execute(f"USE {database}")        
        database_pointer.execute("SELECT * FROM attendance WHERE user_id = ? AND exit_date_time = ?", (user_id, '1900-01-01 00:00:00'))

        return database_pointer.fetchone() is not None, None
    except Exception as e:
       return False, e

def user_exists(database_pointer, user_id):

    try :
        database_pointer.execute(f"USE {database}")
        database_pointer.execute("SELECT * FROM users WHERE user_id = ?", (user_id))
        if database_pointer.fetchone() is not None:
            return True, f"User '{user_id}' already exists."
        else: 
            return False, f"User '{user_id}' does not exists."
    
    except Exception as e:
        return False, e
    

def deleteDatabase(database_pointer):
    try:
        database_pointer.execute("USE master")
        database_pointer.execute(f"DROP DATABASE {database};")
        return True, None
    
    except Exception as e:
        return False, e

def admin_exists(database_pointer, admin_id, admin_pass = None):
    try:
        database_pointer.execute(f"USE {database}")

        if admin_pass is not None: 
            database_pointer.execute("SELECT * FROM admin WHERE admin_id = ? AND password = ?", (admin_id), (admin_pass))
        else: 
            database_pointer.execute("SELECT * FROM admin WHERE admin_id = ?", (admin_id))


        if database_pointer.fetchone() is not None:
            return True, f"Admin '{admin_id}' already exists."
        else: 
            return False, f"Admin '{admin_id}' does not exists."
    
    except Exception as e:
        return False, e
    

def insert_admin(database_pointer, admin_id, admin_pass):
    try:
        database_pointer.execute(f"USE {database}")
        database_pointer.execute("INSERT INTO admin (admin_id, password, created_date) VALUES (?, ?, ?)",
            (admin_id, admin_pass, datetime.now())
        )
        return True, "Admin added successfully!"
    except Exception as e:
        return False, e
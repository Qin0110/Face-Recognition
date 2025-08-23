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
        return e
     


def intialise_database(database_pointer):
    path = "Data\\Train"

    try:
        result = database_pointer.execute(f"SELECT name FROM sys.databases WHERE name = '{database}'")

        database_pointer.execute(f"SELECT name FROM sys.databases WHERE name = '{database}'")
        if database_pointer.fetchone() is None:
            

            database_pointer.execute(f"CREATE DATABASE {database};")
            database_pointer.execute(f"""
            USE {database};

            CREATE TABLE admin (
                admin_id VARCHAR(100) COLLATE Latin1_General_CS_AS PRIMARY KEY,
                password VARCHAR(100) COLLATE Latin1_General_CS_AS NOT NULL,
                created_date DATETIME NOT NULL
            );

            CREATE TABLE users(
                user_id VARCHAR(100) COLLATE Latin1_General_CS_AS PRIMARY KEY,
                department VARCHAR(100),
                face_image_path VARCHAR(300) NOT NULL,
                created_by VARCHAR(100)  COLLATE Latin1_General_CS_AS NOT NULL,
                FOREIGN KEY (created_by) REFERENCES admin(admin_id)
            );
                                    
            CREATE TABLE attendance(
                attendance_id INT PRIMARY KEY IDENTITY(1,1),
                entry_date_time DATETIME NOT NULL, 
                exit_date_time DATETIME NOT NULL DEFAULT '1900-01-01 00:00:00',
                user_id VARCHAR(100) COLLATE Latin1_General_CS_AS,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
                                                         
                                    """)
            
            try: 
                database_pointer.execute(f"INSERT INTO admin (admin_id, password, created_date) VALUES (?, ?, ?)", "admin", "admin", datetime.now())
                database_pointer.execute(f"INSERT INTO admin (admin_id, password, created_date) VALUES (?, ?, ?)", "ADMIN", "ADMIN", datetime.now())
            except Exception as e:
                return False, e          

            if(os.path.exists(path)):
                dir_list = os.listdir(path)
                
                for file in dir_list:
                    
                    try:
                        database_pointer.execute(
                            "INSERT INTO users (user_id, department, face_image_path, created_by) VALUES (?, ?, ?, ?)",
                            (file, None, os.path.join(path, file), 'admin')
                        )
                    except Exception as e:
                        return False, e
                    
            else: 
                return False, None

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


def insert_attendance(database_pointer, user_id):
    try:
        database_pointer.execute(f"USE {database}")        
        database_pointer.execute(
            "INSERT INTO attendance (entry_date_time, user_id) VALUES (?, ?)",
            (datetime.now() , user_id)
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
            database_pointer.execute("SELECT * FROM admin WHERE admin_id = ? ", (admin_id))

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
    
def display_user(database_pointer):
    try:
        database_pointer.execute(f"USE {database}")
        database_pointer.execute("SELECT user_id FROM users")
        return True, None, database_pointer.fetchall()
    except Exception as e:
        return False, e, None
    
def display_user_info(database_pointer, user_id):
    try: 
        database_pointer.execute(f"USE {database}")
        database_pointer.execute("SELECT * FROM users WHERE user_id = ?", (user_id))
        return True, None, list(database_pointer.fetchall())
    except Exception as e: 
        return False, e, None
    

def delete_user(database_pointer, user_id):
    try: 
        database_pointer.execute(f"USE {database}")
        database_pointer.execute("DELETE FROM attendance WHERE user_id = ?", (user_id))
        database_pointer.execute("DELETE FROM users WHERE user_id = ?", (user_id))
        return True, f"User {user_id} has been deleted."
    except Exception as e: 
        return False, f"Failed to delete user {user_id}: {e}"
    
def update_user_data(database_pointer, user_id, department_new):
    try: 
        database_pointer.execute(f"USE {database}")
        database_pointer.execute("UPDATE users SET department = ? WHERE user_id = ?", (department_new, user_id))
        return True, f"{user_id}'s data has been updated."
    except Exception as e:
        return False, f"Failed to update {user_id}'s data: {e}"
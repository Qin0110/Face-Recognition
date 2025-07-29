from tkinter import *
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import GUANWEI as Test2
import threading

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceRecognitionApp:

    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")

        self.root.focus_set() 
        self.root.geometry(f"+{self.root.winfo_x() + 300}+{self.root.winfo_y() + 200}")
        self.login_id = None

        self.setup_login_page()


    def setup_login_page(self):

        def login_check():
            admin_id = id_input.get().strip()
            admin_pass = pass_input.get()

            if admin_id == "" or admin_pass == "":
                error_msg_label.config(text = "Please enter both ID and Password.")
                return
            
            if admin_pass.endswith(" "):
                error_msg_label.config(text = "Password cannot end with space")
                return 
            
            exists, message = Test2.check_admin_exists(admin_id, admin_pass)

            if exists: 
                self.login_id = admin_id
                self.system_page()
                id_input.delete(0, 'end')
                pass_input.delete(0, 'end')
                error_msg_label.config(text = "")

            else: 
                error_msg_label.config(text = "Admin ID or Password is incorrect")

        login_frame = LabelFrame(self.root, padx=20, pady=20)
        login_frame.pack(padx=20, pady=200)

        id_label = Label(login_frame, text = "Name")
        id_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)

        id_input = Entry(login_frame, width = 50)
        id_input.grid(row=0, column=1, padx=10, pady=10)

        pass_label = Label(login_frame, text = "Password")
        pass_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)

        pass_input = Entry(login_frame, width = 50, show='*')
        pass_input.grid(row=1, column=1, padx=10, pady=10)

        error_msg_label = Label(login_frame, text = "", fg = "Red")
        error_msg_label.grid(row = 2, columnspan= 2, padx = 10 , pady= 2, sticky = 'w')

        login_button = Button(login_frame, text="Login", command = login_check, width = 20) ##
        login_button.grid(row=3, column = 1, pady=5, sticky = 'e', padx = 5)






    def system_page(self):
        if self.root.winfo_exists:
            self.root.withdraw() 

        
        page = Toplevel()
        page.geometry("800x600")
        page.title("SYSTEM Page")
        page.resizable(False, False)
        
        page.focus_set() 

        page.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")

        cam = None
        camera_label = Label(page, width = 498, height = 532)

        def disable_close():
            pass 



        def hide_camera(event = None):
            page_msg_label.config(text = "Camera Closed", fg = "Red")
            Test2.close_camera(cam)
            camera_label.grid_remove()
            camera_sensor_btn.grid(row=0, column=0, padx=20, pady=10)  

        def camera_sensor_btn_click(mode = None, status = None, username = None):
            page_msg_label.config(text = "Opening Camera...", fg = "Red")
            nonlocal cam
            camera_sensor_btn.grid_remove() 


            camera_label.grid(row=0, rowspan = 15, column=0, padx=19, pady=10)  

            Test2.close_camera(cam)
            cam = Test2.open_camera()
            if not cam.isOpened():
                messagebox.showerror('Camera Error', 'Error: Could not open camera.')
                return
            
            camera_label.bind("<Button-1>", hide_camera) 
            
            stored_embeddings, stored_labels = Test2.load_precomputed_embeddings()
            
            try: 
                isTrue, message = Test2.update_camera(cam, mode, status, username, camera_label, page, stored_embeddings, stored_labels)

                if isTrue:
                    messagebox.showinfo("Attendance Status", message)
                elif message:
                    messagebox.showerror("Attendance Status", message)

                Test2.close_camera(cam)
                camera_label.grid_remove()
                camera_sensor_btn.grid(row=0, column=0, padx=20, pady=10)
                page_msg_label.config(text = "Camera Closed", fg = "Red")
            except Exception as e:
                pass
                    
            
       

        def logout():
            Test2.close_camera(cam) 
            camera_label.grid_remove()  
            page_msg_label.config(text="Camera Closed", fg="Red")
            page.after(100, page.destroy)
            self.root.deiconify()

        def admin_registration():
            self.root.withdraw()
            register_window = Toplevel()
            register_window.geometry("400x150")
            register_window.title("Register Page")

            register_window.resizable(False, False)

            register_window.focus_set()
            register_window.grab_set() 
            register_window.transient(page) 

            register_window.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")

            admin_id_label = Label(register_window, text = "Admin ID")
            admin_id_label.grid(row = 0, column =0, padx = 10, pady = 5)  

            admin_id_input = Entry(register_window, width = 50)
            admin_id_input.grid(row = 0, column = 1, padx = 10, pady = 5)
            
            admin_pass_label = Label(register_window, text = "Password")
            admin_pass_label.grid(row = 1, column =0, padx = 10, pady = 5)  

            admin_pass_input = Entry(register_window, width = 50, show = '*')
            admin_pass_input.grid(row = 1, column = 1, padx = 10, pady = 5)

            error_msg_label = Label(register_window, text = "", fg = "red")
            error_msg_label.grid(row = 2, columnspan = 2, sticky = 'w', padx = 10, pady = 5)

            def handle_add_admin():
                admin_id = admin_id_input.get().strip()
                admin_pass = admin_pass_input.get()

                if admin_id == "": 
                    error_msg_label.config(text = "Admin ID cannot be empty.", fg = "red")
                    return
                elif admin_pass == "":
                      error_msg_label.config(text = "Password cannot be empty.", fg = "red")
                      return
                elif admin_pass.endswith(" "):
                    error_msg_label.config(text = "Password cannot end with space", fg = "red")
                    return
                else: 
                    result, message = Test2.check_admin_exists(admin_id)

                    if result: 
                        error_msg_label.config(text = message, fg = "red")
                        return
                    
                    else:
                        added, message = Test2.add_admin(admin_id, admin_pass)

                        if added: 
                            error_msg_label.config(text = "Admin added successfully!", fg = "green")

                            messagebox.showinfo("Admin Registration", message)

                            register_window.destroy()

                        else:
                            messagebox.showerror("Error", message)
                            return   
                    
            add_button = Button (register_window, text="Submit", width=20, command = handle_add_admin)
            add_button.grid(row=3, columnspan=2, pady=10, padx = 10)
        

                        
                        


        


        ## add user page
        def add_user_page():
            add_user_window = Toplevel()
            add_user_window.geometry("300x200")
            add_user_window.title("Add User Page")  

            add_user_window.resizable(False, False)

            add_user_window.grab_set() 
            add_user_window.transient(page) 
            add_user_window.focus_set()  
        
            add_user_window.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")



            def on_key_release(*args):
                var.set(var.get().upper())

            var = StringVar()
            var.trace_add("write", on_key_release)

            input_frame = Frame (add_user_window)
            input_frame.grid(pady = 20)

            name_label = Label(input_frame, text="User ID")
            name_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
            name_input = Entry(input_frame, width=30, textvariable=var)
            name_input.grid(row=0, column=1, padx=0, pady=5)  

            department_label = Label(input_frame, text="Department")
            department_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
            department_input = Entry(input_frame, width=30)   
            department_input.grid(row=1, column=1, padx=0, pady=5)

            msg_label = Label(input_frame, text=f"")
            msg_label.grid(row=2, columnspan=2, pady=10, sticky= "w")

            def handdle_add_user(): 
                name = name_input.get().strip()
                department = department_input.get().strip()

                if name == "":
                    msg_label.config(text = "Name cannot be empty.", fg = "red")
                    return
                
                result, message = Test2.check_user_exists(name_input.get())

                if result:
                    msg_label.config(text = message, fg = "red")
                    return 
                
                else:
                    isTrue, message = Test2.add_new_user(name, department, self.login_id)
            

                    if isTrue:
                        msg_label.config(text = message, fg = "green")
                        add_user_window.destroy()

                        Test2.close_camera(cam)
                        
                        camera_sensor_btn.grid_remove() 

                        camera_label.grid(row=0, rowspan = 15, column=0, padx=19, pady=10)  
                    
                        camera_sensor_btn_click(mode = 'capture', username=name)  

                        messagebox.showinfo("Success", message)     
                    else: 
                        messagebox.showerror("Error", message)
                        return   
        
            submit_button = Button (add_user_window, text="Submit", width=20, command = handdle_add_user)
            submit_button.grid(row=3, columnspan=2, pady=10)

        ##take attendance
        def attendance():

            
            def take_attendance():

                selected_status = status.get()
                if selected_status not in ['Enter', 'Exit']:
                    error_msg.config(text = "Please Select An Option.", fg = "red")
                    return 
                else: 
                    attendance_window.destroy()
                    camera_sensor_btn_click(mode = 'attendance', status = selected_status, username = None)
            

            stored_embeddings, stored_labels = Test2.load_precomputed_embeddings()


            if stored_embeddings is None or stored_labels is None:
                    messagebox.showerror("Error", "No training model found. Please run 'Training Model' first.")
                    return

            attendance_window = Toplevel()
            attendance_window.geometry("200x140")
            attendance_window.title("Take Attendance")  

            attendance_window.resizable(False, False)

            attendance_window.grab_set()  
            attendance_window.transient(page)  
            attendance_window.focus_set() 

            attendance_window.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")


            status = StringVar(value="Enter")

            submit_frame = Frame(attendance_window)
            submit_frame.grid(row = 2, pady = 3, padx = 5)

            enter_radio_btn = Radiobutton(attendance_window, text = 'Enter', variable = status, value = 'Enter')
            enter_radio_btn.grid(row=0, column=0, padx=0, pady=2, stick = 'w')

            exit_radio_btn = Radiobutton(attendance_window, text = 'Exit', variable = status, value = 'Exit' )
            exit_radio_btn.grid(row=1, column=0, padx=0, pady=2,  sticky = 'w')

            error_msg = Label(submit_frame, text = "", fg = "red")
            error_msg.grid(row=1,  padx = 0, pady=3, sticky = 'w')
        
            submit_btn = Button(submit_frame, text = "Submit", width = 20, height = 2, command = take_attendance)
            submit_btn.grid(row = 2 , column=0, padx=0, pady=2, sticky = 'w')


        def display_user():

            def disable_close():
                pass

            def back_window():
                user_window.destroy()
                page.deiconify()


            def display_user_info(user_id):

                def disable_close():
                    pass

                def back_window():
                    user_info_window.destroy()
                    display_user()

                def delete_user(user_id, image_path):
                    confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete user '{user_id}'?")
                    if confirm: 
                        success, msg = Test2.delete_user(user_id, image_path)

                        if success:
                            messagebox.showinfo("Success", msg)
                            user_info_window.destroy()  
                            display_user()
                        else:
                            messagebox.showerror("Error", msg)

                user_window.destroy()
                user_info_window = Toplevel()
                user_info_window.geometry("650x300")
                user_info_window.title(f"{user_id}'s information")

                user_info_window.resizable(False, False)

                user_info_window.focus_set()
                user_info_window.grab_set()



                user_info_window.protocol("WM_DELETE_WINDOW", disable_close)

                
                # user_info_window.geometry(f"+{user_window.winfo_x() + 300}+{user_window.winfo_y() + 200}")
                

                isTrue, errorMessage, info = Test2.display_user_info(user_id)

                top_frame = Frame(user_info_window)
                top_frame.pack(side = TOP, padx = 10, pady = 10)

                photo_frame = Frame(top_frame)
                photo_frame.pack(side = LEFT, padx = 10, pady = 10)

                info_frame = Frame(top_frame)
                info_frame.pack(side = RIGHT, padx = 10, pady = 10)

                button_frame = Frame(user_info_window)
                button_frame.pack(side=BOTTOM, pady=10)
      

                def edit_user():
                    department_input.config(state='normal')
                    edit_btn.pack_forget()
                    delete_btn.pack_forget()
                    back_btn.pack_forget()
                    cancel_btn.pack(side = LEFT, padx = 5)
                    confirm_btn.pack(side = LEFT, padx = 5)
                

                def cancel_edit(department_value):
                    confirm_btn.pack_forget()
                    cancel_btn.pack_forget()
                    department_input.delete(0, END)
                    department_input.insert(0, department_value)
                    department_input.config(state='readonly')
                    edit_btn.pack(side=LEFT, padx=5)
                    delete_btn.pack(side=LEFT, padx=5)
                    back_btn.pack(side = LEFT, padx = 5)

                def confirm_edit(user_id, department_value):
                    confirm = messagebox.askyesno("Confirm Edit", f"Are you sure you want to modify user '{user_id}' data?")
                    if confirm:
                        new_department_value = department_input.get()
                        success, msg = Test2.update_user_info(user_id, new_department_value)

                        if success: 
                            messagebox.showinfo("Success", msg)
                            department_input.config(state='readonly')
                            edit_btn.pack(side=LEFT, padx=5)
                            delete_btn.pack(side=LEFT, padx=5)
                            back_btn.pack(side = LEFT, padx = 5)

                        else:
                            messagebox.showerror("Fail", msg)
                            cancel_edit(department_value)
                    else: 
                        cancel_edit(department_value)

                if isTrue: 
                    full_image_path = Test2.get_profile_img(info[0][0], info[0][2])
                    
                    if full_image_path:
                        try:
                            img = Image.open(full_image_path)
                            img = img.resize((200, 200))  # Resize as needed
                            user_img = ImageTk.PhotoImage(img)

                            img_label = Label(photo_frame, image=user_img)
                            img_label.image = user_img  # Prevent garbage collection
                            img_label.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
                        except Exception as e:
                            messagebox.showerror("Error", e)
                            img_label = Label(photo_frame, text="Error loading image")
                            img_label.grid(row=0, column=0, rowspan=4, padx=10, pady=10)

                    else:
                        img_label = Label(photo_frame, text="Image not found")
                        img_label.grid(row=0, column=0, rowspan=4, padx=10, pady=10)





                    user_id_label = Label(info_frame, text = f"User ID: {info[0][0]}")
                    user_id_label.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = 'nw')

                    department_label = Label(info_frame, text = "Department: ")
                    department_label.grid(row = 1, column = 0, padx = 5, pady = 5 , sticky = 'nw')

                    department_input = Entry(info_frame)
                    department_input.grid(row = 1, column = 1, padx = 5, pady = 5, sticky = 'nw')
                    department_value = info[0][1] if info[0][1] is not None else ""
                    department_input.insert(0, department_value)
                    department_input.config(state='readonly')

                    image_path_label = Label(info_frame, text = f"Image Path: {info[0][2]}")
                    image_path_label.grid(row = 2, column = 0, padx = 5, pady = 5, sticky = 'nw')     

                    created_by_label = Label(info_frame, text = f"Created By: {info[0][3]}")
                    created_by_label.grid(row = 3, column = 0, padx = 5, pady = 5, sticky = 'nw')

                    edit_btn = Button(button_frame, text= "Edit", width = 20, command = edit_user)
                    edit_btn.pack(side=LEFT, padx=5)

                    delete_btn = Button(button_frame, text= "Delete", width = 20, command = lambda: delete_user(info[0][0], info[0][2]))
                    delete_btn.pack(side=LEFT, padx=5)

                    back_btn = Button(button_frame, text = "Back", width = 20, command = back_window)
                    back_btn.pack(side = LEFT, padx = 5)

                    confirm_btn = Button(button_frame, text = "Confirm", width = 20, command= lambda: confirm_edit(info[0][0], department_value))
                    cancel_btn = Button(button_frame, text = "Cancel", width = 20, command = lambda: cancel_edit(department_value))                    



            if(page.winfo_exists):
                page.withdraw()
            user_window = Toplevel()
            user_window.geometry("800x600")
            user_window.title("User_ID")

            user_window.resizable(False, False)

            user_window.focus_set()
            user_window.grab_set() 

            user_window.geometry(f"+{user_window.winfo_x() + 300}+{user_window.winfo_y() + 200}")

            
            user_window.protocol("WM_DELETE_WINDOW", disable_close)

            user_id_frame = Frame(user_window)
            user_id_frame.pack(side = TOP, padx = 0, pady = 5)

            button_frame = Frame(user_window)
            button_frame.pack(side = BOTTOM, padx = 0, pady = 5)

            success, message, user_data = Test2.display_user()

            if success: 
                total_user = len(user_data)
                total_columns = (total_user + 14) // 15

                if total_columns >= 5:
                    display_user_frame = Frame(user_window)
                    display_user_frame.pack(fill = BOTH, expand = 1)

                    canvas = Canvas(display_user_frame)
                    canvas.pack(side = "top", fill = "both", expand = 1)

                    scrollbar = Scrollbar(user_window, orient = "horizontal", command = canvas.xview)
                    scrollbar.pack(side = "bottom", fill = X)

                    canvas.configure(xscrollcommand = scrollbar.set)
                    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion = canvas.bbox('all')))

                    second_display_frame = Frame(canvas)
                    canvas.create_window((0,0), window = second_display_frame, anchor = 'nw')

                    parent_frame = second_display_frame
                
                else:
                    parent_frame = Frame(user_id_frame)  
                    parent_frame.pack(fill="both", expand=True) 
                
                count = 0

                for user_id in user_data:
                    column = count // 15
                    row = count % 15
                    user_btn = Button(parent_frame, fg = "Black", width = 20, text = user_id, command = lambda uid=user_id: display_user_info(uid))
                    user_btn.grid(row = row, column = column, padx = 5, pady = 5)
                    count += 1

                back_btn = Button(button_frame, text = "Back", width = 50, command = back_window)
                back_btn.grid(row = 16, column = 0)
            else: 
                messagebox.showerror("Error: ", message)
                user_window.destroy()
        
        def train_modeL():
            popup = Toplevel()
            popup.title("Training Model")
            popup.geometry("300x100")
            popup.resizable(False, False)
            popup.transient(page)
            popup.lift()

            Label(popup, text = "Training Model... \nPlease wait.").pack(pady= 20)
            popup.grab_set()
            popup.protocol("WM_DELETE_WINDOW", lambda: None)  

            popup.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")


            def run_training():
                message = Test2.precompute_embeddings()
                popup.destroy()
                messagebox.showinfo("Success", message)
            
            threading.Thread(target=run_training).start()

            

        
        page.protocol("WM_DELETE_WINDOW", disable_close)

        camera_sensor_btn = Button(page, bg = "Black", text="Loading....", command = camera_sensor_btn_click, width = 70, height = 35)
        camera_sensor_btn.grid(row=0, rowspan = 15, column=0, padx=20, pady=10)  

        add_admin_btn = Button (page, text = "Add New Admin", width = 30, height = 5, command = admin_registration)
        add_admin_btn.grid(row=4 , column=1, padx=0, pady=0)  

        add_user_btn = Button(page, text = "Add New User", width = 30, height = 5, command = add_user_page)
        add_user_btn.grid(row = 5, column = 1, padx=0, pady=0)

        attendance_button = Button(page, text = "Take Attendance", width = 30, height = 5, command = attendance)
        attendance_button.grid (row = 6, column = 1, padx=0, pady=0)

        train_model_btn = Button(page, text = "Train Model", width = 30, height = 5, command = train_modeL)
        train_model_btn.grid(row = 7, column = 1, padx=0, pady=0)

        page_msg_label = Label(page, text = "Camera Closed", fg = 'Red', font=("Arial", 20))
        page_msg_label.grid(row = 15, column = 0, padx = 20, pady=0, sticky= "NW")

        display_user_btn = Button(page, text = "User Information", fg = "Black", width = 30, height = 5, command = display_user)
        display_user_btn.grid(row = 8, column =  1, padx=0, pady=0)
            
        logout_btn = Button(page, text="Logout", command=logout, width=30, height = 5)
        logout_btn.grid(row = 9, column = 1, padx=10, pady=5)

def main():
    isTrue, message = Test2.intialise_database()
    if isTrue:
        root = Tk()
        app = FaceRecognitionApp(root)
        root.mainloop()
    else: 
        messagebox.showerror("Database Initialization Error", message)
        exit()
    

if __name__ == "__main__":
    main()

from tkinter import *
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import GUANWEI

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")

        self.root.focus_set() 
        self.root.geometry(f"+{self.root.winfo_x() + 300}+{self.root.winfo_y() + 200}")

        self.setup_login_page()

    def setup_login_page(self):

        def login_check():
            admin_id = id_input.get().strip()
            password = pass_input.get().strip()

            if admin_id == "" or password == "":
                error_msg_label.config(text = "Please enter both ID and Password.")
                return
            
            exists, message = Test2.admin_login(admin_id, password)

            if exists: 
                self.system_page()
                id_input.delete(0, 'end')
                pass_input.delete(0, 'end')
                error_msg_label.config(text = "")

            else: 
                error_msg_label.config(text = message)

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

        error_msg_label = Label(login_frame, text = "123", fg = "Red")
        error_msg_label.grid(row = 2, columnspan= 2, padx = 10 , pady= 2, sticky = 'w')

        login_button = Button(login_frame, text="Login", command = login_check, width = 20) ##
        login_button.grid(row=3, columnspan = 2, pady=5)

    def system_page(self):
        self.root.withdraw() 
        page = Toplevel()
        page.geometry("800x600")
        page.title("SYSTEM Page")
        
        page.focus_set() 

        page.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")

        cam = None
        camera_label = Label(page, width = 498, height = 532)

        def clean_camera():
            nonlocal cam
            if cam is not None:
                cam.release()
                cv2.destroyAllWindows()
                cam = None


        def camera_sensor_btn_click(mode = None, status = None, username = None):
            page_msg_label.config(text = "Opening Camera...", fg = "Red")
            nonlocal cam
            is_updating = True
            camera_sensor_btn.grid_remove() 
            count = 0


            camera_label.grid(row=0, rowspan = 15, column=0, padx=19, pady=10)  


            clean_camera()
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                messagebox.showerror('Camera Error', 'Error: Could not open camera.')
                return
            
            def hide_camera(event):
                page_msg_label.config(text = "Camera Closed", fg = "Red")
                clean_camera()
                is_updating = False
                camera_label.grid_remove()
                camera_sensor_btn.grid(row=0, column=0, padx=20, pady=10)  

            def update_camera():
                nonlocal cam, is_updating, count
                page_msg_label.config(text = "Capturing...", fg = "Green")
                result = []
                resultNum = []
                max_count = 100 if mode == 'capture' else 10 if mode == 'attendance' else 0

                while (count < max_count if mode == 'capture' else count < max_count if mode == 'attendance' else is_updating):                
                    if not is_updating:  
                        break
                    
                    if cam is not None and cam.isOpened():
            
                        isTrue, frame = cam.read()

                        if isTrue:
                    
                            grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            face_detection = face_cascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5)       
                            if len(face_detection) > 0:
                                cv2.putText(frame, f"Faces detected: {len(face_detection)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            for (x, y, w, h) in face_detection:
                                face_crop = frame[y:y+h, x:x+w]

                                if mode == 'capture':
                                    if(Test2.capturing_images(frame = frame, face_crop = face_crop, name = username, count = count)):
                                        cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                                        cv2.putText(frame, f"Captured: {count}", (x, y-30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        count += 1

                                    if count >= 10:  # Stop after 10 captures
                                        clean_camera()
                                        camera_label.grid_remove()
                                        camera_sensor_btn.grid(row=0, column=0, padx=20, pady=10)
                                        page_msg_label.config(text = "Camera Closed", fg = "Red")
                                        break
                        
                                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 0), 2)

                                if mode == None or mode == 'attendance':
                                    if(Test2.verificationImages(frame)):
                                        try:
                                            face_embedding = Test2.get_embedding(face_crop)

                                            identified_user, distance = Test2.identify_face_euclidean(face_embedding, stored_embeddings, stored_labels)
                                            
                                            if identified_user is not None:
                                                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
                                                cv2.putText(frame, f"{identified_user} ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                                                if mode == 'attendance':
                                                    count += 1
                                                    if identified_user not in result:
                                                        result.append(identified_user)
                                                        resultNum.append(1)
                                                    else: 
                                                        index  = result.index(identified_user)
                                                        resultNum[index] += 1
                                                    
                                                    

                                            else:
                                                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
                                                cv2.putText(frame, f"Unknown ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        except Exception as e:
                                            messagebox.showerror(f'Error processing face: {e}')
                                            cv2.putText(frame, "Error", (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(frame, "Face verification failed", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            img = img.resize((498, 532))  # Resize for better fit
                                        
                            imgtk = ImageTk.PhotoImage(image=img)

                            camera_label.imgtk = imgtk

                            camera_label.configure(image=imgtk)

                            page.update()
                    else: 
                        break

                if mode == 'attendance' and count >= max_count:
                    isTrue, message = Test2.take_attendance(result, resultNum, status)
                    if isTrue:
                        messagebox.showinfo("Attendance Status", message)
                        
                    else:
                        messagebox.showerror("Attendance Status", message)

                    clean_camera()
                    camera_label.grid_remove()
                    camera_sensor_btn.grid(row=0, column=0, padx=20, pady=10)
                    page_msg_label.config(text = "Camera Closed", fg = "Red")

                return
        
            camera_label.bind("<Button-1>", hide_camera) 
            
            stored_embeddings, stored_labels = Test2.load_precomputed_embeddings()

            update_camera()   

        def logout():
            clean_camera()
            page.destroy()
            self.root.deiconify()

        ## add user page
        def add_user_page():
            add_user_window = Toplevel()
            add_user_window.geometry("300x250")
            add_user_window.title("Add User Page")  

            add_user_window.grab_set() 
            add_user_window.transient(page) 
            add_user_window.focus_set()  
        
            add_user_window.geometry(f"+{page.winfo_x() + 300}+{page.winfo_y() + 200}")


            input_frame = Frame (add_user_window)
            input_frame.grid(pady = 20)

            name_label = Label(input_frame, text="Name")
            name_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
            name_input = Entry(input_frame, width=30)
            name_input.grid(row=0, column=1, padx=0, pady=10)  

            department_label = Label(input_frame, text="Department")
            department_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)
            department_input = Entry(input_frame, width=30)   
            department_input.grid(row=1, column=1, padx=0, pady=10)

            msg_label = Label(input_frame, text=f"")
            msg_label.grid(row=2, columnspan=2, pady=10, sticky= "w")

            def handdle_add_user(): 
                name = name_input.get().strip()
                department = department_input.get().strip()

                if name == "":
                    msg_label.config(text = "Name cannot be empty", fg = "red")
                    return
                
                result, message = Test2.check_user_exists(name_input.get())

                if result:
                    msg_label.config(text = message, fg = "red")
                    return 
                
                else:
                    isTrue, message = Test2.add_new_user(name, department)
                    msg_label.config(text = "User added successfully!", fg = "green")
                    add_user_window.destroy()

                    if isTrue:
                        clean_camera() 
                        
                        camera_sensor_btn.grid_remove() 

                        camera_label.grid(row=0, rowspan = 15, column=0, padx=19, pady=10)  
                    
                        camera_sensor_btn_click(mode = 'capture', username=name)       
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
            

            attendance_window = Toplevel()
            attendance_window.geometry("200x140")
            attendance_window.title("Take Attendance")  

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

        
        camera_sensor_btn = Button(page, bg = "Black", text="Loading....", command = camera_sensor_btn_click, width = 70, height = 35)
        camera_sensor_btn.grid(row=0, rowspan = 15, column=0, padx=20, pady=10)  

        add_user_btn = Button(page, text = "Add New User", width = 30, height = 8, command = add_user_page)
        add_user_btn.grid(row = 5, column = 1, padx=0, pady=0)

        attendance_button = Button(page, text = "Identify User", width = 30, height = 8, command = attendance)
        attendance_button.grid (row = 6, column = 1, padx=0, pady=0)

        train_model_btn = Button(page, text = "Train Model", width = 30, height = 8)
        train_model_btn.grid(row = 7, column = 1, padx=0, pady=0)

        page_msg_label = Label(page, text = "Camera Closed", fg = 'Red', font=("Arial", 20))
        page_msg_label.grid(row = 15, column = 0, padx = 20, pady=0, sticky= "NW")
            
        logout_btn = Button(page, text="Logout", command=logout, width=30)
        logout_btn.grid(row = 9, column = 1, padx=10, pady=10)

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

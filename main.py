import os
import sys
import threading
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import attendanceSystem
from attendanceSystem import CameraSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
ERROR_LOG_PATH = os.path.join(BASE_DIR, "error.log")
APP_TITLE = "Face Recognition Attendance System"
TITLE_FONT = ("Arial", 26, "bold")
BUTTON_FONT = ("Arial", 22)
LABEL_FONT = ("Arial", 14)
ENTRY_FONT = ("Arial", 14)
HINT_FONT = ("Arial", 11)
STATUS_FONT = ("Arial", 18)
RESAMPLING = getattr(Image, "Resampling", Image)


def log_exception(exc_type, exc_value, exc_traceback):
    details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]\n{details}\n")
    except OSError:
        pass
    return details


def maximize(window):
    try:
        window.state("zoomed")
    except tk.TclError:
        try:
            window.attributes("-zoomed", True)
        except tk.TclError:
            pass


def center_window(window, parent, min_width=0, min_height=0):
    window.update_idletasks()
    width = max(window.winfo_reqwidth(), min_width)
    height = max(window.winfo_reqheight(), min_height)
    if parent is not None and parent.winfo_viewable():
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    else:
        x = (window.winfo_screenwidth() - width) // 2
        y = (window.winfo_screenheight() - height) // 2
    window.minsize(min_width, min_height)
    window.geometry(f"+{max(0, x)}+{max(0, y)}")


def load_photo(path, max_size):
    if not path:
        return None
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError):
        return None
    scale = min(max_size[0] / image.width, max_size[1] / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return ImageTk.PhotoImage(image.resize(size, RESAMPLING.LANCZOS))


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.login_id = None
        self.page = None
        self.camera_session = None
        self.camera_view = None
        self.camera_photo = None
        self.status_label = None
        self.action_buttons = []
        self.training = False

        root.report_callback_exception = self.report_callback_exception

        style = ttk.Style(root)
        style.configure("Treeview", font=("Arial", 12), rowheight=28)
        style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

        root.title(APP_TITLE)
        root.geometry("1200x720")
        root.minsize(960, 640)
        maximize(root)
        root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.show_login_page()

    def report_callback_exception(self, exc_type, exc_value, exc_traceback):
        log_exception(exc_type, exc_value, exc_traceback)
        messagebox.showerror(
            "Unexpected Error",
            f"{exc_value}\n\nDetails were saved to:\n{ERROR_LOG_PATH}",
            parent=self.root,
        )

    def new_page(self):
        if self.page is not None:
            self.page.destroy()
        self.camera_view = None
        self.camera_photo = None
        self.status_label = None
        self.action_buttons = []
        self.page = tk.Frame(self.root)
        self.page.pack(fill="both", expand=True)
        return self.page

    def create_dialog(self, title, parent=None):
        dialog = tk.Toplevel(parent or self.root)
        dialog.withdraw()
        dialog.title(title)
        return dialog

    def show_dialog(self, dialog, parent=None, min_width=0, min_height=0):
        parent = parent or self.root
        center_window(dialog, parent, min_width, min_height)
        dialog.deiconify()
        dialog.transient(parent)
        try:
            dialog.grab_set()
        except tk.TclError:
            pass
        dialog.focus_set()

    def close_dialog(self, dialog, parent=None):
        if dialog.winfo_exists():
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
        if parent is not None and parent is not self.root and parent.winfo_exists():
            try:
                parent.grab_set()
            except tk.TclError:
                pass
            parent.focus_set()
        else:
            self.root.focus_set()

    def show_login_page(self):
        page = self.new_page()
        self.root.title(f"{APP_TITLE} - Login")
        container = tk.Frame(page)
        container.place(relx=0.5, rely=0.5, anchor="center")

        logo = load_photo(LOGO_PATH, (500, 400)) if os.path.isfile(LOGO_PATH) else None
        if logo is not None:
            logo_label = tk.Label(container, image=logo)
            logo_label.image = logo
            logo_label.pack(pady=(0, 10))
        else:
            tk.Label(container, text=APP_TITLE, font=TITLE_FONT).pack(pady=(0, 20))

        form = tk.LabelFrame(container, text="Admin Login", font=LABEL_FONT, padx=20, pady=15)
        form.pack(fill="x", padx=20)
        form.columnconfigure(1, weight=1)

        tk.Label(form, text="Admin ID", font=LABEL_FONT).grid(row=0, column=0, sticky="e", padx=5, pady=8)
        id_entry = tk.Entry(form, width=40, font=ENTRY_FONT)
        id_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=8)

        tk.Label(form, text="Password", font=LABEL_FONT).grid(row=1, column=0, sticky="e", padx=5, pady=8)
        password_entry = tk.Entry(form, width=40, font=ENTRY_FONT, show="*")
        password_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=8)

        show_password = tk.BooleanVar(value=False)
        tk.Checkbutton(
            form,
            text="Show password",
            variable=show_password,
            font=HINT_FONT,
            command=lambda: password_entry.config(show="" if show_password.get() else "*"),
        ).grid(row=2, column=1, sticky="w", padx=10)

        error_label = tk.Label(form, text="", fg="red", font=LABEL_FONT, wraplength=520, justify="left")
        error_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        def submit(event=None):
            admin_id = id_entry.get().strip()
            success, message = attendanceSystem.login(admin_id, password_entry.get().strip())
            if not success:
                error_label.config(text=message)
                return
            self.login_id = admin_id
            self.show_system_page()

        tk.Button(form, text="Login", width=20, font=LABEL_FONT, command=submit).grid(
            row=4, column=1, sticky="e", padx=10, pady=5
        )
        id_entry.bind("<Return>", submit)
        password_entry.bind("<Return>", submit)
        id_entry.focus_set()

    def show_system_page(self):
        page = self.new_page()
        self.root.title(f"{APP_TITLE} - Logged in as {self.login_id}")
        actions = (
            ("Add New Admin", self.open_admin_registration),
            ("Add New User", self.open_add_user),
            ("Take Attendance", self.open_take_attendance),
            ("Train Model", self.start_training),
            ("User Information", self.open_user_list),
            ("Attendance Records", self.open_attendance_records),
            ("Logout", self.logout),
        )
        page.columnconfigure(0, weight=5)
        page.columnconfigure(1, weight=2)
        for row in range(len(actions)):
            page.rowconfigure(row, weight=1, uniform="row", minsize=50)

        camera_frame = tk.Frame(page, bg="black")
        camera_frame.grid(row=0, column=0, rowspan=len(actions), sticky="nsew", padx=30, pady=(30, 10))
        self.camera_view = tk.Label(camera_frame, bg="black", fg="white", font=STATUS_FONT, cursor="hand2")
        self.camera_view.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.camera_view.bind("<Button-1>", self.toggle_camera)

        for row, (text, command) in enumerate(actions):
            button = tk.Button(page, text=text, command=command, font=BUTTON_FONT)
            button.grid(row=row, column=1, sticky="nsew", padx=(10, 30), pady=(30 if row == 0 else 4, 4))
            if command != self.logout:
                self.action_buttons.append(button)

        self.status_label = tk.Label(page, text="Camera Closed", fg="red", font=STATUS_FONT, anchor="w")
        self.status_label.grid(row=len(actions), column=0, columnspan=2, sticky="ew", padx=30, pady=(0, 20))
        self.show_camera_placeholder()

        error = attendanceSystem.model_error()
        if error:
            self.set_status("Recognition model failed to load. Check the internet connection and installed packages.", "red")

    def set_status(self, text, color="red"):
        if self.status_label is not None and self.status_label.winfo_exists():
            self.status_label.config(text=text, fg=color)

    def set_busy(self, busy):
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self.action_buttons:
            if button.winfo_exists():
                button.config(state=state)

    def is_busy(self, parent=None):
        if self.training:
            messagebox.showwarning("Please Wait", "Model training is in progress.", parent=parent or self.root)
            return True
        if self.camera_session is not None:
            messagebox.showwarning("Please Wait", "Please close the camera first.", parent=parent or self.root)
            return True
        return False

    def show_camera_placeholder(self):
        if self.camera_view is not None and self.camera_view.winfo_exists():
            self.camera_photo = None
            self.camera_view.config(image="", text="Click here to open the camera")

    def toggle_camera(self, event=None):
        if self.camera_session is not None:
            self.camera_session.stop()
            self.set_status("Closing camera...", "red")
            return
        if not self.training:
            self.start_camera(CameraSession.PREVIEW)

    def start_camera(self, mode, username=None, status=None):
        if self.camera_session is not None or self.training or self.camera_view is None:
            return
        session = CameraSession(mode, username=username, status=status)
        self.camera_session = session
        self.set_busy(True)
        self.camera_view.config(image="", text="Opening camera...")
        self.set_status("Opening Camera...", "red")
        session.start()
        self.root.after(30, self.poll_camera, session)

    def poll_camera(self, session):
        if session is not self.camera_session:
            return
        frame = session.take_frame()
        if frame is not None:
            self.show_frame(frame)
        if session.is_running():
            self.set_status(f"{session.message}   (click the camera view to close)", "green")
            self.root.after(30, self.poll_camera, session)
            return
        self.finish_camera(session)

    def show_frame(self, frame):
        if self.camera_view is None or not self.camera_view.winfo_exists():
            return
        width = self.camera_view.winfo_width()
        height = self.camera_view.winfo_height()
        if width < 10 or height < 10:
            return
        image = Image.fromarray(frame)
        scale = min(width / image.width, height / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        self.camera_photo = ImageTk.PhotoImage(image.resize(size, RESAMPLING.BILINEAR))
        self.camera_view.config(image=self.camera_photo, text="")

    def finish_camera(self, session):
        self.camera_session = None
        session.join(1.0)
        self.show_camera_placeholder()
        self.set_busy(False)
        self.set_status("Camera Closed", "red")
        success, message = attendanceSystem.complete_session(session)
        if not message:
            return
        titles = {
            CameraSession.CAPTURE: "Face Capture",
            CameraSession.ATTENDANCE: "Attendance Status",
            CameraSession.PREVIEW: "Camera",
        }
        if success:
            messagebox.showinfo(titles[session.mode], message, parent=self.root)
        else:
            messagebox.showerror(titles[session.mode], message, parent=self.root)

    def stop_camera_session(self):
        session = self.camera_session
        self.camera_session = None
        if session is not None:
            session.stop()
            session.join(2.0)

    def logout(self):
        if self.training:
            messagebox.showwarning("Logout", "Please wait for the model training to finish.", parent=self.root)
            return
        if not messagebox.askyesno("Logout", "Do you want to log out?", parent=self.root):
            return
        self.stop_camera_session()
        self.login_id = None
        self.show_login_page()

    def quit_app(self):
        question = "Model training is still running. Quit anyway?" if self.training else "Do you really want to quit?"
        if not messagebox.askyesno("Quit", question, parent=self.root):
            return
        self.stop_camera_session()
        attendanceSystem.shutdown()
        self.root.destroy()

    def open_admin_registration(self):
        if self.is_busy():
            return
        dialog = self.create_dialog("Add New Admin")
        dialog.columnconfigure(1, weight=1)
        entries = {}
        fields = (("admin_id", "Admin ID", ""), ("password", "Password", "*"), ("confirm", "Confirm Password", "*"))
        for row, (key, label, mask) in enumerate(fields):
            tk.Label(dialog, text=label, font=LABEL_FONT).grid(row=row, column=0, sticky="e", padx=(20, 10), pady=8)
            entry = tk.Entry(dialog, width=40, font=ENTRY_FONT, show=mask)
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 20), pady=8)
            entries[key] = entry

        message_label = tk.Label(
            dialog,
            text=f"Password must be at least {attendanceSystem.MIN_PASSWORD_LENGTH} characters without spaces.",
            fg="gray",
            font=HINT_FONT,
            wraplength=520,
            justify="left",
        )
        message_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=20, pady=5)

        def submit(event=None):
            admin_id = entries["admin_id"].get().strip()
            success, message = attendanceSystem.register_admin(admin_id, entries["password"].get(), entries["confirm"].get())
            if not success:
                message_label.config(text=message, fg="red")
                return
            self.close_dialog(dialog)
            messagebox.showinfo("Admin Registration", message, parent=self.root)

        button_frame = tk.Frame(dialog)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(5, 20))
        tk.Button(button_frame, text="Submit", width=15, font=LABEL_FONT, command=submit).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", width=15, font=LABEL_FONT, command=lambda: self.close_dialog(dialog)).pack(side="left", padx=5)
        dialog.bind("<Return>", submit)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self.close_dialog(dialog))
        self.show_dialog(dialog, min_width=640)
        entries["admin_id"].focus_set()

    def open_add_user(self):
        if self.is_busy():
            return
        dialog = self.create_dialog("Add New User")
        dialog.columnconfigure(1, weight=1)

        user_var = tk.StringVar()

        def force_upper(*_):
            value = user_var.get()
            if value != value.upper():
                user_var.set(value.upper())

        user_var.trace_add("write", force_upper)

        tk.Label(dialog, text="User ID", font=LABEL_FONT).grid(row=0, column=0, sticky="e", padx=(20, 10), pady=8)
        user_entry = tk.Entry(dialog, textvariable=user_var, width=40, font=ENTRY_FONT)
        user_entry.grid(row=0, column=1, sticky="ew", padx=(10, 20), pady=8)

        tk.Label(dialog, text="Department", font=LABEL_FONT).grid(row=1, column=0, sticky="e", padx=(20, 10), pady=8)
        department_entry = tk.Entry(dialog, width=40, font=ENTRY_FONT)
        department_entry.grid(row=1, column=1, sticky="ew", padx=(10, 20), pady=8)

        message_label = tk.Label(
            dialog,
            text=f"{attendanceSystem.CAPTURE_TARGET} face images will be captured after the user is added.",
            fg="gray",
            font=HINT_FONT,
            wraplength=520,
            justify="left",
        )
        message_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=20, pady=5)

        def submit(event=None):
            user_id = user_var.get().strip()
            success, message = attendanceSystem.add_new_user(user_id, department_entry.get().strip(), self.login_id)
            if not success:
                message_label.config(text=message, fg="red")
                return
            self.close_dialog(dialog)
            messagebox.showinfo(
                "Add New User",
                f"{message}\n\nPlease look at the camera. {attendanceSystem.CAPTURE_TARGET} face images will be captured.\n"
                "Click the camera view to stop.",
                parent=self.root,
            )
            self.start_camera(CameraSession.CAPTURE, username=user_id)

        button_frame = tk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(5, 20))
        tk.Button(button_frame, text="Submit", width=15, font=LABEL_FONT, command=submit).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", width=15, font=LABEL_FONT, command=lambda: self.close_dialog(dialog)).pack(side="left", padx=5)
        dialog.bind("<Return>", submit)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self.close_dialog(dialog))
        self.show_dialog(dialog, min_width=640)
        user_entry.focus_set()

    def open_take_attendance(self):
        if self.is_busy():
            return
        if not attendanceSystem.check_data_exists():
            messagebox.showerror("Take Attendance", "No trained data found. Please run 'Train Model' first.", parent=self.root)
            return
        dialog = self.create_dialog("Take Attendance")
        status = tk.StringVar(value="Enter")
        tk.Radiobutton(dialog, text="Enter", variable=status, value="Enter", font=("Arial", 30)).grid(
            row=0, column=0, sticky="w", padx=30, pady=(20, 2)
        )
        tk.Radiobutton(dialog, text="Exit", variable=status, value="Exit", font=("Arial", 30)).grid(
            row=1, column=0, sticky="w", padx=30, pady=2
        )

        def submit(event=None):
            selected = status.get()
            if selected not in ("Enter", "Exit"):
                return
            self.close_dialog(dialog)
            self.start_camera(CameraSession.ATTENDANCE, status=selected)

        tk.Button(dialog, text="Submit", width=30, height=2, font=LABEL_FONT, command=submit).grid(
            row=2, column=0, padx=30, pady=(10, 20)
        )
        dialog.bind("<Return>", submit)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self.close_dialog(dialog))
        self.show_dialog(dialog, min_width=400)

    def start_training(self):
        if self.is_busy():
            return
        if not messagebox.askyesno(
            "Train Model",
            "Training rebuilds the training data from the Datasets folder and may take several minutes.\n\n"
            "Do you want to continue?",
            parent=self.root,
        ):
            return

        self.training = True
        self.set_busy(True)
        popup = self.create_dialog("Training Model")
        popup.protocol("WM_DELETE_WINDOW", lambda: None)
        tk.Label(popup, text="Training Model...\nPlease wait.", font=STATUS_FONT, justify="center").pack(padx=40, pady=(30, 10))
        progress_label = tk.Label(popup, text="Starting...", font=LABEL_FONT, wraplength=520, justify="center")
        progress_label.pack(padx=40, pady=(0, 10))
        progress_bar = ttk.Progressbar(popup, mode="indeterminate", length=420)
        progress_bar.pack(padx=40, pady=(0, 30))
        progress_bar.start(15)
        self.show_dialog(popup, min_width=600, min_height=230)

        state = {"progress": "Starting...", "result": None}

        def report(message):
            state["progress"] = message

        def worker():
            state["result"] = attendanceSystem.train_model(report)

        thread = threading.Thread(target=worker, name="training", daemon=True)
        thread.start()

        def poll():
            if thread.is_alive():
                progress_label.config(text=state["progress"])
                popup.after(200, poll)
                return
            progress_bar.stop()
            self.close_dialog(popup)
            self.training = False
            self.set_busy(False)
            success, message = state["result"] or (False, "Training stopped unexpectedly.")
            if success:
                messagebox.showinfo("Train Model", message, parent=self.root)
            else:
                messagebox.showerror("Train Model", message, parent=self.root)

        popup.after(200, poll)

    def open_user_list(self):
        if self.is_busy():
            return
        dialog = self.create_dialog("User Information")

        top_frame = tk.Frame(dialog)
        top_frame.pack(fill="x", padx=15, pady=(15, 5))
        tk.Label(top_frame, text="Search", font=LABEL_FONT).pack(side="left")
        search_var = tk.StringVar()
        search_entry = tk.Entry(top_frame, textvariable=search_var, width=30, font=ENTRY_FONT)
        search_entry.pack(side="left", padx=10)
        count_label = tk.Label(top_frame, text="", font=LABEL_FONT)
        count_label.pack(side="right")

        table_frame = tk.Frame(dialog)
        table_frame.pack(fill="both", expand=True, padx=15, pady=5)
        columns = (
            ("user_id", "User ID", 280, "w"),
            ("department", "Department", 240, "w"),
            ("images", "Face Images", 130, "center"),
            ("created_by", "Created By", 180, "w"),
        )
        tree = ttk.Treeview(table_frame, columns=[column[0] for column in columns], show="headings", selectmode="browse")
        for key, heading, width, anchor in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, anchor=anchor)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        users = []

        def render(*_):
            query = search_var.get().strip().lower()
            tree.delete(*tree.get_children())
            shown = 0
            for user in users:
                if query and query not in user["user_id"].lower() and query not in user["department"].lower():
                    continue
                tree.insert(
                    "",
                    "end",
                    iid=user["user_id"],
                    values=(user["user_id"], user["department"], user["image_count"], user["created_by"]),
                )
                shown += 1
            count_label.config(text=f"{shown} of {len(users)} users")

        def load():
            if not dialog.winfo_exists():
                return
            success, message, data = attendanceSystem.list_users()
            if not success:
                messagebox.showerror("User Information", message, parent=dialog)
            users[:] = data
            render()

        def view(event=None):
            if event is not None and event.type == tk.EventType.ButtonPress and tree.identify_region(event.x, event.y) != "cell":
                return
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("User Information", "Please select a user.", parent=dialog)
                return
            self.open_user_info(selection[0], dialog, load)

        search_var.trace_add("write", render)
        tree.bind("<Double-1>", view)
        tree.bind("<Return>", view)

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=15)
        tk.Button(button_frame, text="View Details", width=18, font=LABEL_FONT, command=view).pack(side="left", padx=5)
        tk.Button(button_frame, text="Refresh", width=18, font=LABEL_FONT, command=load).pack(side="left", padx=5)
        tk.Button(button_frame, text="Back", width=18, font=LABEL_FONT, command=lambda: self.close_dialog(dialog)).pack(side="left", padx=5)

        dialog.protocol("WM_DELETE_WINDOW", lambda: self.close_dialog(dialog))
        self.show_dialog(dialog, min_width=900, min_height=560)
        load()
        search_entry.focus_set()

    def open_user_info(self, user_id, parent, on_change):
        success, message, user = attendanceSystem.get_user_info(user_id)
        if not success:
            messagebox.showerror("User Information", message, parent=parent)
            on_change()
            return

        dialog = self.create_dialog(f"{user_id}'s Information", parent)
        body = tk.Frame(dialog)
        body.pack(fill="both", expand=True, padx=25, pady=(25, 10))
        body.columnconfigure(2, weight=1)

        photo = load_photo(user["profile_image"], (360, 360))
        if photo is not None:
            photo_label = tk.Label(body, image=photo, relief="groove")
            photo_label.image = photo
        else:
            photo_label = tk.Label(body, text="No Image", font=LABEL_FONT, width=30, height=15, relief="groove")
        photo_label.grid(row=0, column=0, rowspan=6, sticky="n", padx=(0, 25))

        department_var = tk.StringVar(value=user["department"])
        rows = (
            ("User ID:", user["user_id"]),
            ("Department:", None),
            ("Image Folder:", user["face_image_path"]),
            ("Face Images:", str(user["image_count"])),
            ("Created By:", user["created_by"]),
        )
        department_entry = None
        for row, (label, value) in enumerate(rows):
            tk.Label(body, text=label, font=LABEL_FONT).grid(row=row, column=1, sticky="nw", padx=5, pady=10)
            if value is None:
                department_entry = tk.Entry(body, textvariable=department_var, width=35, font=ENTRY_FONT, state="readonly")
                department_entry.grid(row=row, column=2, sticky="ew", padx=5, pady=10)
            else:
                tk.Label(body, text=value, font=LABEL_FONT, wraplength=420, justify="left").grid(
                    row=row, column=2, sticky="nw", padx=5, pady=10
                )

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=(5, 20))
        original = {"department": user["department"]}

        def back():
            self.close_dialog(dialog, parent)

        def set_editing(editing):
            for button in normal_buttons + edit_buttons:
                button.pack_forget()
            for button in edit_buttons if editing else normal_buttons:
                button.pack(side="left", padx=5)
            department_entry.config(state="normal" if editing else "readonly")
            if editing:
                department_entry.focus_set()

        def cancel_edit():
            department_var.set(original["department"])
            set_editing(False)

        def confirm_edit():
            new_value = department_var.get().strip()
            if not messagebox.askyesno("Confirm Edit", f"Are you sure you want to modify user '{user_id}' data?", parent=dialog):
                cancel_edit()
                return
            updated, update_message = attendanceSystem.update_user_department(user_id, new_value)
            if not updated:
                messagebox.showerror("Edit User", update_message, parent=dialog)
                return
            original["department"] = new_value
            department_var.set(new_value)
            set_editing(False)
            messagebox.showinfo("Edit User", update_message, parent=dialog)
            on_change()

        def delete():
            if not messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete user '{user_id}'?\n\n"
                "All attendance records and face images of this user will also be removed.",
                parent=dialog,
            ):
                return
            deleted, delete_message = attendanceSystem.delete_user(user_id)
            if not deleted:
                messagebox.showerror("Delete User", delete_message, parent=dialog)
                return
            back()
            messagebox.showinfo("Delete User", delete_message, parent=parent)
            on_change()

        def retake():
            if self.is_busy(dialog):
                return
            if not messagebox.askyesno(
                "Retake Photos",
                f"All existing face images of '{user_id}' will be deleted and "
                f"{attendanceSystem.CAPTURE_TARGET} new images will be captured.\n\nDo you want to continue?",
                parent=dialog,
            ):
                return
            prepared, prepare_message = attendanceSystem.prepare_recapture(user_id)
            if not prepared:
                messagebox.showerror("Retake Photos", prepare_message, parent=dialog)
                return
            self.close_dialog(dialog, parent)
            self.close_dialog(parent)
            messagebox.showinfo(
                "Retake Photos",
                "Please look at the camera. Click the camera view to stop.",
                parent=self.root,
            )
            self.start_camera(CameraSession.CAPTURE, username=user_id)

        normal_buttons = [
            tk.Button(button_frame, text="Edit", width=16, font=LABEL_FONT, command=lambda: set_editing(True)),
            tk.Button(button_frame, text="Delete", width=16, font=LABEL_FONT, command=delete),
            tk.Button(button_frame, text="Retake Photos", width=16, font=LABEL_FONT, command=retake),
            tk.Button(button_frame, text="Back", width=16, font=LABEL_FONT, command=back),
        ]
        edit_buttons = [
            tk.Button(button_frame, text="Confirm", width=16, font=LABEL_FONT, command=confirm_edit),
            tk.Button(button_frame, text="Cancel", width=16, font=LABEL_FONT, command=cancel_edit),
        ]
        set_editing(False)

        dialog.protocol("WM_DELETE_WINDOW", back)
        self.show_dialog(dialog, parent, min_width=900, min_height=480)

    def open_attendance_records(self):
        if self.is_busy():
            return
        dialog = self.create_dialog("Attendance Records")
        today = datetime.now().strftime(attendanceSystem.DATE_FORMAT)

        filter_frame = tk.Frame(dialog)
        filter_frame.pack(fill="x", padx=15, pady=(15, 5))
        user_var = tk.StringVar()
        from_var = tk.StringVar(value=today)
        to_var = tk.StringVar(value=today)

        tk.Label(filter_frame, text="User ID", font=LABEL_FONT).grid(row=0, column=0, padx=(0, 5))
        user_entry = tk.Entry(filter_frame, textvariable=user_var, width=18, font=ENTRY_FONT)
        user_entry.grid(row=0, column=1, padx=(0, 15))
        tk.Label(filter_frame, text="From (YYYY-MM-DD)", font=LABEL_FONT).grid(row=0, column=2, padx=(0, 5))
        from_entry = tk.Entry(filter_frame, textvariable=from_var, width=12, font=ENTRY_FONT)
        from_entry.grid(row=0, column=3, padx=(0, 15))
        tk.Label(filter_frame, text="To", font=LABEL_FONT).grid(row=0, column=4, padx=(0, 5))
        to_entry = tk.Entry(filter_frame, textvariable=to_var, width=12, font=ENTRY_FONT)
        to_entry.grid(row=0, column=5, padx=(0, 15))

        table_frame = tk.Frame(dialog)
        table_frame.pack(fill="both", expand=True, padx=15, pady=5)
        columns = (
            ("no", "No", 60, "center"),
            ("user_id", "User ID", 220, "w"),
            ("department", "Department", 180, "w"),
            ("entry", "Entry", 190, "center"),
            ("exit", "Exit", 190, "center"),
            ("duration", "Duration", 110, "center"),
            ("status", "Status", 130, "center"),
        )
        tree = ttk.Treeview(table_frame, columns=[column[0] for column in columns], show="headings")
        for key, heading, width, anchor in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, anchor=anchor)
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        bottom_frame = tk.Frame(dialog)
        bottom_frame.pack(fill="x", padx=15, pady=15)
        summary_label = tk.Label(bottom_frame, text="", font=LABEL_FONT)
        summary_label.pack(side="left")

        records = []

        def search(event=None):
            success, message, data = attendanceSystem.get_attendance_records(user_var.get(), from_var.get(), to_var.get())
            if not success:
                messagebox.showerror("Attendance Records", message, parent=dialog)
                return
            records[:] = data
            tree.delete(*tree.get_children())
            for index, record in enumerate(records, start=1):
                tree.insert(
                    "",
                    "end",
                    values=(
                        index,
                        record["user_id"],
                        record["department"],
                        attendanceSystem.format_datetime(record["entry"]),
                        attendanceSystem.format_datetime(record["exit"]) or "-",
                        record["duration"] or "-",
                        record["status"],
                    ),
                )
            checked_in = sum(1 for record in records if record["exit"] is None)
            user_count = len({record["user_id"] for record in records})
            summary_label.config(text=f"{len(records)} records   |   {user_count} users   |   {checked_in} still checked in")

        def show_today():
            current = datetime.now().strftime(attendanceSystem.DATE_FORMAT)
            from_var.set(current)
            to_var.set(current)
            search()

        def show_all_dates():
            from_var.set("")
            to_var.set("")
            search()

        def export():
            if not records:
                messagebox.showinfo("Export CSV", "There are no records to export.", parent=dialog)
                return
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Export Attendance Records",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"attendance_{datetime.now():%Y%m%d_%H%M%S}.csv",
            )
            if not path:
                return
            exported, export_message = attendanceSystem.export_attendance_csv(records, path)
            if exported:
                messagebox.showinfo("Export CSV", export_message, parent=dialog)
            else:
                messagebox.showerror("Export CSV", export_message, parent=dialog)

        tk.Button(filter_frame, text="Search", width=10, font=LABEL_FONT, command=search).grid(row=0, column=6, padx=3)
        tk.Button(filter_frame, text="Today", width=10, font=LABEL_FONT, command=show_today).grid(row=0, column=7, padx=3)
        tk.Button(filter_frame, text="All Dates", width=10, font=LABEL_FONT, command=show_all_dates).grid(row=0, column=8, padx=3)
        tk.Button(bottom_frame, text="Close", width=14, font=LABEL_FONT, command=lambda: self.close_dialog(dialog)).pack(side="right", padx=5)
        tk.Button(bottom_frame, text="Export CSV", width=14, font=LABEL_FONT, command=export).pack(side="right", padx=5)
        for entry in (user_entry, from_entry, to_entry):
            entry.bind("<Return>", search)

        dialog.protocol("WM_DELETE_WINDOW", lambda: self.close_dialog(dialog))
        self.show_dialog(dialog, min_width=1150, min_height=620)
        search()


def prepare_environment():
    print("[1/4] Checking installed packages...")
    missing = []
    for module in ("cv2", "numpy", "PIL", "sklearn", "matplotlib", "torch", "facenet_pytorch"):
        try:
            __import__(module)
        except Exception as error:
            missing.append(f"{module} ({error})")
    if missing:
        print("Missing or broken packages: " + ", ".join(missing))
        return 1
    try:
        __import__("face_recognition")
    except (Exception, SystemExit) as error:
        print(f"Warning: face_recognition is unavailable, so the Hybrid Approach engine cannot be used ({error}).")

    print("[2/4] Preparing the database...")
    success, message = attendanceSystem.initialise_system()
    if not success:
        print(message)
        return 1

    print("[3/4] Downloading and loading the recognition model...")
    try:
        attendanceSystem.get_engine().load_model()
    except Exception as error:
        print(error)
        return 1

    print("[4/4] Preparing the trained data...")
    if attendanceSystem.check_data_exists():
        print("Trained data already exists.")
    elif attendanceSystem.count_trainable_users() == 0:
        print(
            "No face images found yet. Start the application, add a user so that "
            f"{attendanceSystem.CAPTURE_TARGET} face images are captured, then click 'Train Model'."
        )
    else:
        print("Training the model. This may take several minutes...")
        success, message = attendanceSystem.train_model(print)
        print(message)
        if not success:
            return 1
    attendanceSystem.shutdown()
    return 0


def main():
    os.chdir(BASE_DIR)
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    root = tk.Tk()
    root.withdraw()
    success, message = attendanceSystem.initialise_system()
    if not success:
        messagebox.showerror("Database Initialization Error", message, parent=root)
        root.destroy()
        return 1
    attendanceSystem.preload_model()
    root.deiconify()
    FaceRecognitionApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    if "--setup" in sys.argv[1:]:
        os.chdir(BASE_DIR)
        raise SystemExit(prepare_environment())
    try:
        exit_code = main()
    except Exception:
        log_exception(*sys.exc_info())
        try:
            error_root = tk.Tk()
            error_root.withdraw()
            messagebox.showerror(
                "Unexpected Error",
                f"{sys.exc_info()[1]}\n\nDetails were saved to:\n{ERROR_LOG_PATH}",
                parent=error_root,
            )
            error_root.destroy()
        except tk.TclError:
            pass
        exit_code = 1
    raise SystemExit(exit_code)

import tkinter as tk
from tkinter import ttk
import os

class ChoiceWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Settings")
        self.root.minsize(500, 300)
        
        # Language options
        self.languages = {"en": "English", "pl": "Polski"}
        self.voice_languages = ["English", "Polish", "German", "French"]  # Example languages
        self.modes = {"quiz": "Quiz Mode", "learning": "Learning Mode"}
        self.directions = {"left": "Left", "right": "Right"}
        
        self.translations = {
            "en": {"language": "Language:", "sound": "Sound:", "voice": "Voice Language:",
                    "file": "Select File:", "direction": "Direction:", "mode": "Mode:", "ok": "OK"},
            "pl": {"language": "Język:", "sound": "Dźwięk:", "voice": "Język głosu:",
                    "file": "Wybierz plik:", "direction": "Kierunek:", "mode": "Tryb:", "ok": "OK"}
        }
        
        self.selected_language = tk.StringVar(value="en")
        self.sound_on = tk.BooleanVar(value=True)
        self.selected_voice_language = tk.StringVar(value=self.voice_languages[0])
        self.selected_direction = tk.StringVar(value="left")
        self.selected_mode = tk.StringVar(value="quiz")
        self.selected_file = tk.StringVar()
        
        # UI Elements
        self.create_widgets()
        self.update_labels()
        
        self.root.mainloop()

    def create_widgets(self):
        # Language Selection
        self.lang_label = tk.Label(self.root)
        self.lang_dropdown = ttk.Combobox(self.root, values=list(self.languages.values()), state="readonly",
                                          textvariable=self.selected_language)
        self.lang_dropdown.bind("<<ComboboxSelected>>", self.change_language)
        
        # Sound Toggle
        self.sound_label = tk.Label(self.root)
        self.sound_button = ttk.Checkbutton(self.root, variable=self.sound_on, text="")
        
        # Voice Language
        self.voice_label = tk.Label(self.root)
        self.voice_dropdown = ttk.Combobox(self.root, values=self.voice_languages, state="readonly",
                                           textvariable=self.selected_voice_language)
        
        # File Selection
        self.file_label = tk.Label(self.root)
        self.file_dropdown = ttk.Combobox(self.root, values=self.get_files(), state="readonly",
                                          textvariable=self.selected_file)
        
        # Direction Selection
        self.direction_label = tk.Label(self.root)
        self.direction_dropdown = ttk.Combobox(self.root, values=list(self.directions.values()), state="readonly",
                                               textvariable=self.selected_direction)
        
        # Mode Selection
        self.mode_label = tk.Label(self.root)
        self.mode_dropdown = ttk.Combobox(self.root, values=list(self.modes.values()), state="readonly",
                                          textvariable=self.selected_mode)
        
        # OK Button
        self.ok_button = tk.Button(self.root, command=self.on_ok)
        
        # Layout
        self.lang_label.pack()
        self.lang_dropdown.pack()
        self.sound_label.pack()
        self.sound_button.pack()
        self.voice_label.pack()
        self.voice_dropdown.pack()
        self.file_label.pack()
        self.file_dropdown.pack()
        self.direction_label.pack()
        self.direction_dropdown.pack()
        self.mode_label.pack()
        self.mode_dropdown.pack()
        self.ok_button.pack(pady=10)

    def get_files(self):
        folder = "do_nauki"
        return os.listdir(folder) if os.path.exists(folder) else []

    def change_language(self, event=None):
        lang_code = list(self.languages.keys())[list(self.languages.values()).index(self.selected_language.get())]
        self.selected_language.set(lang_code)
        self.update_labels()

    def update_labels(self):
        lang = self.selected_language.get()
        self.lang_label.config(text=self.translations[lang]["language"])
        self.sound_label.config(text=self.translations[lang]["sound"])
        self.voice_label.config(text=self.translations[lang]["voice"])
        self.file_label.config(text=self.translations[lang]["file"])
        self.direction_label.config(text=self.translations[lang]["direction"])
        self.mode_label.config(text=self.translations[lang]["mode"])
        self.ok_button.config(text=self.translations[lang]["ok"])

    def on_ok(self):
        print("Selected Language:", self.selected_language.get())
        print("Sound:", "On" if self.sound_on.get() else "Off")
        print("Voice Language:", self.selected_voice_language.get())
        print("Selected File:", self.selected_file.get())
        print("Direction:", self.selected_direction.get())
        print("Mode:", self.selected_mode.get())
        self.root.destroy()

if __name__ == "__main__":
    ChoiceWindow()

#!/usr/bin/env python

#spolszczyć: stronę do nauki i tryb


# import sys
import time
import random
import os
import csv
import subprocess
from unidecode import unidecode
from datetime import datetime
import Levenshtein
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, font, Tk, ttk
import pyttsx3 #to speak without internet connection
from gtts import gTTS #to speak with internet connection using google tts
import tempfile
import platform
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame


# Global variables
#j_en = False

class Config:
    def __init__(self):
        self.captions_language = 'pl'
        self.voice_language = 'pl'
        self.selected_mode='quiz'
        self.selected_direction='left'
        self.main_folder_path = ""
        self.file_name = ""
        self.use_google_tts = True
        self.sound = True
        self.font_sizes = {
            "default": 20,
            "big": 20,
            "remaining": 14,
            "result": 14,
            "label": 14,
            "list": 12,
        }
        self.system_name = "Linux"
        self.vocabulary_folder_path = os.path.join(self.main_folder_path, "do_nauki")
        self.vocabulary_files_w_lines = []
        self.init_font_name()
        self.file_list()
        
    def init_font_name(self):
        # Tymczasowo inicjalizuj tkinter, aby sprawdzić dostępność czcionki
        temp_root = Tk()
        temp_root.withdraw()  # Ukryj tymczasowe okno
        # Sprawdź dostępność czcionki i ustaw zmienną globalną
        self.font_name = "Ubuntu" if "Ubuntu" in font.families() else "Arial"
        # Zamknij tymczasowe okno
        temp_root.destroy()
        
    def file_list(self):
        self.vocabulary_files_w_lines = []
        for file_name in os.listdir(self.vocabulary_folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.vocabulary_folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    liczba_linii = sum(1 for _ in f)
                self.vocabulary_files_w_lines.append((file_name, liczba_linii))
                # self.vocabulary_files.append(file_name)

class WordSet:
    def __init__(self, file_name, config: Config):
        self.config = config
        # self.file_name = config.file_name
        self.file_name=file_name
        self.font = config.font_name
        self.words = []  # List of dictionaries for word pairs  (left:"", right:"", repetitions:1, errors:0)
        self.load_words()

    def load_words(self):
        """Loads words from a file into a list of dictionaries."""
        with open(self.file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):  # Skip comments
                    continue
                try:
                    pl, en = line.split(';')  # Split into Polish and English words
                    if self.config.direction !="Right":
                        pl, en = en, pl
                    self.words.append({
                        'left': pl,
                        'right': en,
                        'repetitions': 1,  # Initial repetition count
                        'errors': 0       # Initial error count
                    })
                except ValueError:
                    print(f"Skipping malformed line: {line}")

    def get_random_pair_idx(self):
        """Returns a random word pair based on repetition counts."""
        total_repetitions = sum(word['repetitions'] for word in self.words)
        if total_repetitions == 0:
            return None  # No words left to quiz
        # Randomly select a word pair based on repetition counts
        random_value = random.randint(0, total_repetitions - 1)
        for i, entry in enumerate(self.words):
            random_value -= entry['repetitions']
            if random_value < 0:
                return i
        return None  # Fallback (should not be reached)

    def get_most_difficult_words(self):
        """Returns most difficult word pairs based on error counts."""
        sorted_words = sorted(self.words, key=lambda x: x['errors'], reverse=True)
        return [w for w in sorted_words if w['errors'] > 0]


    def update_repetition(self, idx, correct):
        """Updates the repetition count for a word pair based on whether the answer was correct."""
        if idx< len(self.words):
            if correct:
                self.words[idx]['repetitions'] = max(0, self.words[idx]['repetitions'] - 1)  # Decrease repetitions if correct
            elif self.config.selected_mode == 'quiz':
                self.words[idx]['repetitions'] = max(0, self.words[idx]['repetitions'] - 1)  # Decrease repetitions if correct
                self.words[idx]['errors'] += + 1  # increase errors if not correct
            else:
                self.words[idx]['repetitions'] += 2  # increase repetitions if not correct
                self.words[idx]['errors'] += + 1  # increase errors if not correct

    
class ChoiceWindow:
    def __init__(self, config: Config):
        self.config = config
        self.font = config.font_name
        self.root = tk.Tk()
        self.root.title("Settings")
        self.root.minsize(500, 300)
        
        # Jeśli mamy już listę plików, używamy jej, w przeciwnym razie ustawiamy pustą
        self.vocabulary_files_w_lines = config.vocabulary_files_w_lines if config.vocabulary_files_w_lines else []
        # self.vocabulary_files = config.vocabulary_files if config.vocabulary_files else []
        
        # Ustawienia interfejsu
        self.languages = {"en": "English", "pl": "Polski"}
        self.voice_languages = ["de", "en", "es", "fi", "fr", "hr", "hu", "it", "ja", "ko", "no", "pl", "pt", "ru", "se", "zh"]
        self.modes = {
            "en": {"quiz": "Quiz Mode", "learning": "Learning Mode"},
            "pl": {"quiz": "Quiz", "learning": "Nauka"}
        }
        self.directions = {
            "en": {"left": "Left", "right": "Right"},
            "pl": {"left": "Lewa", "right": "Prawa"}
        }
        self.translations = {
            "en": {"language": "Interface language:", "sound": "Sound:", "voice": "Voice Language:",
                   "file": "Select File:", "direction": "Side to learn:", "mode": "Mode:", "ok": "OK", "lines": "lines", "example": "Example pair:", "left": "left", "right": "right"},
            "pl": {"language": "Język interfejsu:", "sound": "Dźwięk:", "voice": "Język głosu:",
                   "file": "Wybierz plik:", "direction": "Strona do nauki:", "mode": "Tryb:", "ok": "OK", "lines": "linii", "example": "Przykładowa para:", "left": "lewa", "right": "prawa"}
        }
        
        # Zmienne tkinter
        self.selected_language = tk.StringVar(value="en")
        self.sound_on = tk.BooleanVar(value=True)
        self.selected_voice_language = tk.StringVar(value=self.voice_languages[0])
        self.selected_direction = tk.StringVar(value="left")
        self.selected_mode = tk.StringVar(value="quiz")
        self.selected_file = tk.StringVar()
        
        self.create_widgets()
        self.update_labels()
        
        self.root.mainloop()

    def create_widgets(self):
        # Configure the style for the combobox and dropdown list
        style = ttk.Style()
        style.configure("TCombobox",
                        font=(self.font, self.config.font_sizes['list']),
                        foreground="blue",
                        background="white")

        label_width=40
        # Język
        self.lang_label = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.lang_dropdown = ttk.Combobox(self.root, values=list(self.languages.values()),
                                          state="readonly", textvariable=self.selected_language, width=label_width, font=(self.font, self.config.font_sizes['list']))
        self.lang_dropdown.bind("<<ComboboxSelected>>", self.change_language)
        lang = self.selected_language.get()
        # Dźwięk
        self.sound_label = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.sound_button = ttk.Checkbutton(self.root, variable=self.sound_on, text="")
        # Język głosu
        self.voice_label = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.voice_dropdown = ttk.Combobox(self.root, values=self.voice_languages, state="readonly",
                                           textvariable=self.selected_voice_language, width=label_width, font=(self.font, self.config.font_sizes['list']))
        # Plik
        self.file_label = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.file_dropdown = ttk.Combobox(self.root, values=self.get_files(), state="readonly",
                                          textvariable=self.selected_file, width=label_width, font=(self.font, self.config.font_sizes['list']))
        # Bind the callback function to the selection event
        # cut (lines) in selected_file
        self.file_dropdown.bind("<<ComboboxSelected>>", self.on_file_selected)
        # Kierunek
        self.example_pair = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.direction_label = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.direction_dropdown = ttk.Combobox(self.root, values=list(self.directions[lang].values()), state="readonly",
                                               textvariable=self.selected_direction, width=label_width, font=(self.font, self.config.font_sizes['list']))
        # Tryb
        self.mode_label = tk.Label(self.root, font=(self.font, self.config.font_sizes['list']))
        self.mode_dropdown = ttk.Combobox(self.root, values=list(self.modes[lang].values()), state="readonly",
                                          textvariable=self.selected_mode, width=label_width, font=(self.font, self.config.font_sizes['list']))
        # Przycisk OK
        self.ok_button = tk.Button(self.root, command=self.on_ok, font=(self.font, self.config.font_sizes['list']))
        
        # Layout
        self.lang_label.pack(pady=5, padx=20)
        self.lang_dropdown.pack(pady=5, padx=20)
        self.sound_label.pack(pady=5)
        self.sound_button.pack(pady=5)
        self.voice_label.pack(pady=5)
        self.voice_dropdown.pack(pady=5)
        self.file_label.pack(pady=5)
        self.file_dropdown.pack(pady=5)
        self.example_pair.pack(pady=5)
        self.direction_label.pack(pady=5)
        self.direction_dropdown.pack(pady=5)
        self.mode_label.pack(pady=5)
        self.mode_dropdown.pack(pady=5)
        self.ok_button.pack(pady=10)
        
    def on_file_selected(self, event=None):
        current_value = self.selected_file.get()
        lang = self.selected_language.get()
        if ' ' in current_value:
            new_value = current_value.split(' ', 1)[0]
            self.selected_file.set(new_value)
        # Fallback text if file reading fails or the line isn't in the expected format.
        example_text = f'{self.translations[lang]["left"]} - {self.translations[lang]["right"]}'
        #DEBUG
        file_path = os.path.join(str(self.config.main_folder_path), "do_nauki", self.selected_file.get())
        try:
            # Open the file. Adjust the path if needed.
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Skip comment lines (lines starting with '#' after stripping leading whitespace)
                    if not line.lstrip().startswith("#"):
                        line = line.strip()
                        if ";" in line:
                            # Split at the first semicolon into left and right parts
                            left, right = line.split(";", 1)
                            # For both left and right, cut off everything to the right of the first '|'
                            left = left.split("|", 1)[0].strip() if "|" in left else left.strip()
                            right = right.split("|", 1)[0].strip() if "|" in right else right.strip()
                            example_text = f"{left} - {right}"
                        break
        except Exception as e:
            print("did not open the file with example", e)
            pass
    
        self.example_pair.config(text=f'{self.translations[lang]["example"]} {example_text}')


    def change_language(self, event=None):
        # Aktualizujemy wyświetlaną wartość na skrót językowy
        lang_code = [code for code, name in self.languages.items() if name == self.selected_language.get()]
        if lang_code:
            self.selected_language.set(lang_code[0])
        self.update_labels()
        self.on_file_selected()
        
    def get_files(self):
        lang = self.selected_language.get()  # Możesz dostosować filtrowanie do języka, jeśli pliki są jakoś oznaczone
        return [f"{nazwa} ({liczba} {self.translations[lang]['lines']})" for nazwa, liczba in self.vocabulary_files_w_lines]

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


def popraw(zly, dobry):
    # Funkcja do zaznaczania błędów
    wynik = ""
    for i in range(len(dobry)):
        if i < len(zly) and zly[i] == dobry[i]:
            wynik += dobry[i]
        else:
            wynik += dobry[i].upper()    
    return wynik

def wypisz_najtrudniejsze(najtrudniejsze_list, config: Config):
    """Wyświetla okno z najtrudniejszymi parami i zapisuje wynik do pliku CSV."""
    if not najtrudniejsze_list:
        if config.sound:
            sound_file = os.path.join(config.main_folder_path, "dzwiek", "koniec.wav")
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        messagebox.showinfo("Info", "Well done!" if config.captions_language == "en" else "Dobra robota!")
        return

    result_window = tk.Tk()
    title = "Most Difficult Pairs" if config.captions_language == "en" else "Najtrudniejsze pary wyrazów"
    result_window.title(title)
    
    text_widget = tk.Text(result_window, wrap=tk.WORD, font=(config.font_name, config.font_sizes["result"]))
    text_widget.pack(fill=tk.BOTH, expand=True)
    
    # Extract data
    polski = [w['left'] for w in najtrudniejsze_list]
    angielski = [w['right'] for w in najtrudniejsze_list]
    errors = [w['errors'] for w in najtrudniejsze_list]
    
    # Sort by descending errors and then by word
    sorted_indices = sorted(range(len(errors)), key=lambda i: (-errors[i], polski[i]))
    
    for i in sorted_indices:
        if errors[i] > 0:
            text_widget.insert(tk.END, f'{polski[i]} - {angielski[i]} : {errors[i]}\n')
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(config.main_folder_path, "najtrudniejsze")
    base_name = config.file_name.rsplit('.csv', 1)[0]  
    filename = os.path.join(base_path, f"{base_name}_najtrudniejsze_{date_str}.csv")
    if config.selected_mode=='quiz':
        filename = os.path.join(base_path, f"{base_name}_sprawdzian_najtrudniejsze_{date_str}.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["1 język", "2 język", "Liczba błędów"])
        for i in sorted_indices:
            if errors[i] > 0:
                writer.writerow([polski[i], angielski[i], errors[i]])
                
    info_text = (f'\nThe most difficult pairs were written to {filename}'
                 if config.captions_language == "en"
                 else f'\nNajtrudniejsze pary wyrazów zostały zapisane do pliku {filename}')
    text_widget.insert(tk.END, info_text)
    
    close_button = tk.Button(result_window, text="Close" if config.captions_language == "en" else "Zamknij",
                             command=result_window.destroy)
    close_button.pack(pady=10, padx=10)
    
    if config.sound:
        sound_file = os.path.join(config.main_folder_path, "dzwiek", "koniec.wav")
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    result_window.mainloop()

# Normalize a word for comparison
def normalize(word):
    return unidecode(word).strip().lower()

# Znalezienie najbliższego dopasowania
def find_closest_match(user_input, solution):
    alternatives = solution.split('|')  # Split alternatives by "|"
    user_normalized = normalize(user_input)
    alternatives_normalized = [normalize(alt) for alt in alternatives]
    distances = [Levenshtein.distance(user_normalized, alt) for alt in alternatives_normalized]
    min_distance = min(distances)
    closest_index = distances.index(min_distance)
    return alternatives[closest_index], min_distance

    
def speak(text, voice_language, config: Config):
    """Odtwarza podany tekst korzystając z Google TTS lub espeak-ng."""
    text = text.split("|")[0]
    if config.use_google_tts:
        try:
            tts = gTTS(text, lang=voice_language)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                tts.save(temp_audio.name)
                temp_audio.close()
                pygame.mixer.music.load(temp_audio.name)
                pygame.mixer.music.play()
                os.remove(temp_audio.name)
        except Exception as e:
            print(f"Błąd podczas syntezowania mowy (Google TTS): {e}")
    else:
        try:
            subprocess.run(["espeak-ng", "-v", voice_language, text], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Błąd podczas syntezowania mowy (espeak-ng): {e}")


def nauka(wordset: WordSet, config: Config):
    """
    Main function for the learning/test session.
    
    Parameters:
        wordset (WordSet): The set of words to study.
        config (Config): Configuration settings.
        test_mode (bool): Whether to run in test mode.
    """
    try:
        # Extract word data from wordset
        words = wordset.words
        if not words:
            messagebox.showinfo("Info", "No words available!" if config.captions_language == "en" else "Brak słów!")
            return

        # Initialize variables
        remaining = len(words)
        current_index = 0
        false_previous = False

        # Set up GUI
        root = tk.Tk()
        root.withdraw()

        # Start window
        start_window = tk.Toplevel(root)
        start_window.title("Start")
        start_label = tk.Label(start_window, text="Let's start!" if config.captions_language == "en" else "Zaczynamy!",
                               font=(config.font_name, 24))
        start_label.pack(pady=50, padx=50)

        def start_main_app():
            if config.sound:
                sound_file = os.path.join(config.main_folder_path, "dzwiek", "poczatek.wav")
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
                time.sleep(4)  # Wait for sound to play
            start_window.destroy()
            root.deiconify()
            update_question()

        start_window.after(300, start_main_app)

        # Main window setup
        root.title("Test" if config.selected_mode=='quiz' else "Learning")
        question_label = tk.Label(root, text="", font=(config.font_name, config.font_sizes["remaining"]))
        question_label.pack(pady=1)
        question_label2 = tk.Label(root, text="", font=(config.font_name, config.font_sizes["big"]))
        question_label2.pack(pady=10)

        entry = tk.Entry(root, font=(config.font_name, 20))
        entry.pack(pady=10, padx=10)

        result_label = tk.Label(root, text="", font=(config.font_name, config.font_sizes["remaining"]))
        result_label.pack(pady=10)
        result_label2 = tk.Label(root, text="", font=(config.font_name, config.font_sizes["big"]))
        result_label2.pack(pady=10)

        remaining_label = tk.Label(root, text="", font=(config.font_name, config.font_sizes["list"]))
        remaining_label.pack(pady=10)

        def update_question():
            nonlocal current_index
            remaining = sum(word['repetitions'] for word in wordset.words)
            remaining_words = sum(1 for word in wordset.words if word['repetitions'] > 0)
            if remaining == 0:
                result_label.config(text="No more words to quiz!" if config.captions_language == "en" else "Brak słów do quizu!")
                root.destroy()
                most_difficult = wordset.get_most_difficult_words()
                wypisz_najtrudniejsze(most_difficult, config)
                return

            current_word = words[current_index]
            question = current_word['left']
            correct_answer = current_word['right']

            if config.sound:
                # Synchronizowanie dźwięku z aktualizacją GUI
                root.after(500, lambda: speak(question, config.voice_language, config))  # Delikatne opóźnienie, aby dać czas na zaktualizowanie GUI
            # if dzwiek:
            #     speak(polski[i],voice_language)

            # Clear previous input and update display
            entry.delete(0, tk.END)
            question_label.config(text="Translate the word:" if config.captions_language == "en" else "Podaj tłumaczenie wyrazu:")
            question_label2.config(text=question, fg="blue")
            remaining_label.config(
                text=f"{remaining} questions and {remaining_words} unique pairs remain.\n Write \"q!\" to abort and see result." 
                     if config.captions_language == "en"
                     else f"{remaining} pytań oraz {remaining_words} różnych par do końca.\nWpisz \"q!\" żeby przerwać i zobaczyć wynik."
            )       
            entry.focus_set()


        def on_submit(event=None):
            nonlocal current_index, remaining, false_previous
            user_answer = entry.get().strip().lower()
            entry.delete(0, tk.END)

            if not user_answer:
                user_answer=""
                # result_label.config(text="Please enter an answer." if config.captions_language == "en" else "Proszę wprowadzić odpowiedź.")
                # return

            if user_answer == "q!":
                root.destroy()
                most_difficult = wordset.get_most_difficult_words()
                wypisz_najtrudniejsze(most_difficult, config)
                return

            current_word = words[current_index]
            correct_answer = current_word['right']
            angielski_normalized, min_distance = find_closest_match(user_answer, correct_answer.strip().lower())
            
            result_color="black"
            result_color2="red"
            if min_distance == 0:
                false_previous=False
                feedback = "Correct!" if config.captions_language == "en" else "Dobrze!"
                feedback2 =" "
                result_color = "green"
                # Update wordset data
                wordset.update_repetition(current_index, correct=True)
            else:
                false_previous=True
                feedback = "Incorrect." if config.captions_language == "en" else "Błędnie."
                result_color = "red"
                # Update wordset data
                wordset.update_repetition(current_index, correct=False)
                
                if not config.selected_mode=='quiz':
                    false_previous=True
                poprawione = popraw(user_answer, angielski_normalized)
                if user_answer == "":
                    poprawione = correct_answer
                feedback = f"the correct answer is:" if config.captions_language == "en" else f"prawidłowa odpowiedź to:"
                feedback2 =f"{poprawione}"
                result_color="black"
                # result_label.config(text="the correct answer is:" if j_en else "prawidłowa odpowiedź to:", fg="black")
                # result_label2.config(text=f"{poprawione}" if j_en else f"{poprawione}", fg="red")
                # powtorz[i] += 2
                # if sprawdzian:
                    # powtorz[i] = 0
                # najtrudniejsze[i] += 1
    
            # remaining = sum(powtorz)
            # remaining_words = sum(1 for j in powtorz if j != 0)
        
        
            
            # Normalize and compare answers
            # normalized_user = normalize(user_answer)
            # normalized_correct = normalize(correct_answer)


            result_label.config(text=feedback, fg=result_color)
            result_label2.config(text=feedback2, fg=result_color2)
            if config.sound:
                sound_file = os.path.join(config.main_folder_path, "dzwiek", "prawidlowa.wav" if min_distance == 0 else "bledna.wav")
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
            if config.selected_mode=='quiz' or not false_previous:
                current_index = wordset.get_random_pair_idx()
            false_previous=False
            remaining = sum(word['repetitions'] for word in wordset.words)
            update_question()

        entry.bind("<Return>", on_submit)
        submit_button = tk.Button(root, text="Submit" if config.captions_language == "en" else "Zatwierdź", command=on_submit)
        submit_button.pack(pady=10, padx=10)

        if config.sound:
            pygame.mixer.init()

        root.mainloop()

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        if config.sound:
            sound_file = os.path.join(config.main_folder_path, "dzwiek", "błąd.wav")
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        messagebox.showerror("Error", error_msg)

def main():
    # Initialize the Config object
    config = Config()
    # Initialize Pygame mixer
    if config.sound:
        pygame.mixer.init()
    
    
    # Set the main folder path
    config.main_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(config.main_folder_path)
    
    # Set up the vocabulary folder path
    config.vocabulary_folder_path = os.path.join(config.main_folder_path, "do_nauki")
    config.file_list()  # Populate vocabulary_files_w_lines
    
    # Create the ChoiceWindow to get user settings
    choice_window = ChoiceWindow(config)
    choice_window.root.mainloop()
    
    # Retrieve settings from ChoiceWindow
    settings = {
        'language': choice_window.selected_language.get(),
        'sound': choice_window.sound_on.get(),
        'voice_language': choice_window.selected_voice_language.get(),
        'file': choice_window.selected_file.get(),
        'direction': choice_window.selected_direction.get(),
        'mode': choice_window.selected_mode.get()
    }
    
    # Update Config with selected settings
    config.captions_language = settings['language']
    config.voice_language = settings['voice_language']
    config.sound = settings['sound']
    config.direction = settings['direction']
    config.selected_mode = settings['mode']
    config.file_name = settings['file']
    
    # Select the vocabulary file
    file_name = settings['file']
    file_path = os.path.join(config.vocabulary_folder_path, file_name)
    
    # Load the word set
    wordset = WordSet(file_path, config)
    
    # Start the learning/quiz session
    nauka(wordset, config)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
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
from tkinter import filedialog, messagebox, simpledialog, font, Tk 
#import threading

# Global variables
j_en = False
nazwa_pliku = ""
polski = []
angielski = []
powtorz = []
najtrudniejsze = []
dzwiek = False
font_size=20
font_size_remaining=14
font_size_result=14

# Tymczasowo inicjalizuj tkinter, aby sprawdzić dostępność czcionki
temp_root = Tk()
temp_root.withdraw()  # Ukryj tymczasowe okno
# Sprawdź dostępność czcionki i ustaw zmienną globalną
font_name = "Ubuntu" if "Ubuntu" in font.families() else "Arial"
# Zamknij tymczasowe okno
temp_root.destroy()


def lista_plikow(ścieżka):
    pliki = []
    for nazwa in os.listdir(ścieżka):
        if nazwa.endswith('.csv'):
            pełna_ścieżka = os.path.join(ścieżka, nazwa)
            with open(pełna_ścieżka, 'r', encoding='utf-8') as f:
                liczba_linii = sum(1 for _ in f)
            pliki.append((nazwa, liczba_linii))
    return pliki

    
class ChoiceWindow:
    def __init__(self, title, options, prompt_text):
        self.selected_index = None
        self.options = options
        # Główne okno
        self.root = tk.Tk()
        self.root.title(title)
        # Ustawienie minimalnej szerokości okna na 400 pikseli
        self.root.minsize(500, 1)  # Szerokość: 400, wysokość: 1 (domyślna)
        # Etykieta z promptem
        self.label = tk.Label(self.root, text=prompt_text)
        self.label.pack(pady=10)
        # Listbox z opcjami
        self.listbox = tk.Listbox(self.root)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Dodanie opcji do Listbox
        for option in options:
            self.listbox.insert(tk.END, option)
        # Przycisk potwierdzający wybór
        self.select_button = tk.Button(self.root, text="OK", command=self.on_select)
        self.select_button.pack(pady=10)
        # Powiązanie klawiszy z funkcjami
        self.root.bind("<Up>", self.on_key)
        self.root.bind("<Down>", self.on_key)
        self.root.bind("<Return>", self.on_key)
        # Domyślne zaznaczenie pierwszego elementu
        self.listbox.selection_set(0)
        self.listbox.activate(0)

    def on_select(self, event=None):
        """Obsługa wyboru opcji."""
        selected_index = self.listbox.curselection()
        if selected_index:
            self.selected_index = selected_index[0]
            self.root.destroy()

    def on_key(self, event):
        """Obsługa klawiszy strzałek i Enter."""
        if event.keysym == "Up":  # Strzałka w górę
            current_index = self.listbox.curselection()
            if current_index:
                new_index = max(0, current_index[0] - 1)
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(new_index)
                self.listbox.activate(new_index)
        elif event.keysym == "Down":  # Strzałka w dół
            current_index = self.listbox.curselection()
            if current_index:
                new_index = min(len(self.options) - 1, current_index[0] + 1)
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(new_index)
                self.listbox.activate(new_index)
        elif event.keysym == "Return":  # Enter
            self.on_select()

    def show(self):
        """Wyświetla okno i zwraca wybraną opcję."""
        self.root.mainloop()
        if self.selected_index is not None:
            return self.options[self.selected_index]
        return None

class CustomAskYesNo(ChoiceWindow):
    def __init__(self, title, message, j_en, default_yes=True):
        # Ustawienie opcji w zależności od języka
        if j_en:
            options = ["Yes", "No"] if default_yes else ["No", "Yes"]
        else:
            options = ["Tak", "Nie"] if default_yes else ["Nie", "Tak"]
        super().__init__(title, options, message)
        # Ustawienie domyślnego przycisku
        if default_yes:
            self.listbox.selection_set(0)  # Domyślnie zaznacz "No" lub "Nie"
        else:
            self.listbox.selection_set(1)  # Domyślnie zaznacz "Yes" lub "Tak"
    def show(self):
        """Wyświetla okno i zwraca True (Yes/Tak) lub False (No/Nie)."""
        super().show()
        if self.selected_index is not None:
            return self.options[self.selected_index] in ["Yes", "Tak"]
        return False

# Funkcja do wyboru języka
def wybierz_jezyk():
    jezyki = ["PL", "en"]
    choice_window = ChoiceWindow(
        title="Wybierz język / Set the language",
        options=jezyki,
        prompt_text="Wybierz język / Choose the language:"
    )
    wybrany_jezyk = choice_window.show()
    return wybrany_jezyk.lower() == "en"


def wybierz_plik(pliki, j_en):
    # Przygotowanie opcji w zależności od języka
    if j_en:
        options = [f"{nazwa} ({liczba} lines)" for nazwa, liczba in pliki]
        title = "Choose a file to learn"
        prompt_text = "Choose a file:"
    else:
        options = [f"{nazwa} ({liczba} linii)" for nazwa, liczba in pliki]
        title = "Wybierz plik do nauki"
        prompt_text = "Wybierz plik:"

    # Tworzenie okna wyboru
    choice_window = ChoiceWindow(
        title=title,
        options=options,
        prompt_text=prompt_text
    )

    # Wyświetlenie okna i pobranie wyboru
    wybrany_plik = choice_window.show()
    if wybrany_plik:
        return wybrany_plik.split(" ")[0]  # Zwraca nazwę pliku (pierwszy element)
    return None


def zaimportuj(nazwa_pliku):
    with open(nazwa_pliku, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=";")
        zaimportowana_lista = [row for row in reader if not row[0].startswith("#")]
    polski, angielski = map(list, zip(*zaimportowana_lista))
    return polski, angielski


def losuj_numer(tablica_powtorzen):
    # Check if there are any words left to quiz
    suma_powtorzen = sum(tablica_powtorzen)
    if suma_powtorzen == 0:
        return None  # No words left to quiz
    # Randomly select a word to quiz
    tmp = random.randint(0, suma_powtorzen - 1)
    for i, wartosc in enumerate(tablica_powtorzen):
        tmp -= wartosc
        if tmp < 0:
            return i

def popraw(zly, dobry):
    # Funkcja do zaznaczania błędów
    wynik = ""
    for i in range(len(dobry)):
        if i < len(zly) and zly[i] == dobry[i]:
            wynik += dobry[i]
        else:
            wynik += dobry[i].upper()    
    return wynik

def wypisz_najtrudniejsze(sprawdzian=False, dzwiek=False):
    if sum(najtrudniejsze) == 0:
        messagebox.showinfo("Info", "Well done!" if j_en else "Dobra robota!")
        return

    # Create a new window to display the most difficult pairs
    result_window = tk.Tk()
    result_window.title("Most Difficult Pairs" if j_en else "Najtrudniejsze pary wyrazów")

    # Create a text widget to display the pairs
    text_widget = tk.Text(result_window, wrap=tk.WORD, font=(font_name, font_size_result))
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Sort and display the most difficult pairs
    sorted_indices = sorted(
        range(len(najtrudniejsze)), 
        key=lambda i: (-najtrudniejsze[i], polski[i])
    )
    
    for i in sorted_indices:
        if najtrudniejsze[i] > 0:
            text_widget.insert(tk.END, f'{polski[i]} - {angielski[i]} : {najtrudniejsze[i]}\n')

    # Save to CSV (unchanged)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'najtrudniejsze/{nazwa_pliku}_najtrudniejsze_{date_str}.csv'
    if sprawdzian:
        filename = f'najtrudniejsze/{nazwa_pliku}_sprawdzian_najtrudniejsze_{date_str}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["1 język", "2 język", "Liczba powtórzeń"])
        for i in sorted_indices:
            if najtrudniejsze[i] > 0:
                writer.writerow([polski[i], angielski[i], najtrudniejsze[i]])

    text_widget.insert(tk.END, f'\nThe most difficult pairs were written to {filename}' if j_en else f'\nNajtrudniejsze pary wyrazów zostały zapisane do pliku {filename}')

    # Add a close button
    close_button = tk.Button(result_window, text="Close" if j_en else "Zamknij", command=result_window.destroy)
    close_button.pack(pady=10)
    
    
    if dzwiek: 
        subprocess.call("cvlc --play-and-exit dzwiek/koniec.wav 2> /dev/null", shell=True)

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


def nauka(polski, angielski, powtorz, dzwiek, sprawdzian=False):
    def on_submit(event=None):
        nonlocal i, remaining, remaining_words, false_previous
        wyraz = entry.get()
        entry.delete(0, tk.END)

        if wyraz == "q!":
            root.destroy()
            wypisz_najtrudniejsze(sprawdzian, dzwiek)
            return

        wyraz_normalized = unidecode(wyraz).strip().lower()
        angielski_wyraz = unidecode(angielski[i]).strip().lower()

        angielski_normalized, min_distance = find_closest_match(wyraz_normalized, angielski_wyraz)
        if min_distance == 0:
            false_previous=False
            result_label.config(text="correct!" if j_en else "dobrze!", fg="green")
            result_label2.config(text="", fg="green")
            powtorz[i] = max(0, powtorz[i] - 1)
            if dzwiek:
                subprocess.call("cvlc --play-and-exit dzwiek/prawidlowa.wav 2> /dev/null", shell=True)
        else:
            if not sprawdzian:
                false_previous=True
            poprawione = popraw(wyraz, angielski_normalized)
            if wyraz == "":
                poprawione = angielski_wyraz
            result_label.config(text=f"the correct answer is:" if j_en else f"prawidłowa odpowiedź to:", fg="black")
            result_label2.config(text=f"{poprawione}" if j_en else f"{poprawione}", fg="red")
            powtorz[i] += 2
            if sprawdzian:
                powtorz[i] = 0
            najtrudniejsze[i] += 1
            if dzwiek:
                subprocess.call("cvlc --play-and-exit dzwiek/bledna.wav 2> /dev/null", shell=True)

        remaining = sum(powtorz)
        remaining_words = sum(1 for j in powtorz if j != 0)

        if remaining == 0:
            result_label.config(text="No more words to quiz!" if j_en else "Brak słów do quizu!")
            root.destroy()
            wypisz_najtrudniejsze(sprawdzian, dzwiek)
            return

        update_question()

    def update_question():
        nonlocal i, false_previous
        if not false_previous:
            i = losuj_numer(powtorz)
        if i is None:
            result_label.config(text="No more words to quiz!" if j_en else "Brak słów do quizu!")
            return

        question_label.config(text=f"Translate the word:\n{polski[i]}" if j_en else f"Podaj tłumaczenie wyrazu:\n{polski[i]}")
        remaining_label.config(
    text=f"{remaining} questions and {remaining_words} unique pairs remain.\nClick write \"q!\" to abort and see result." 
         if j_en 
         else f"{remaining} pytań oraz {remaining_words} różnych par do końca.\nWpisz \"q!\" żeby przerwać i zobaczyć wynik."
)

        entry.focus_set()

    # Inicjalizujemy GŁÓWNE okno (jedno!)
    root = tk.Tk()
    root.withdraw()  # Ukrywamy na początku, aby nie pojawiało się od razu

    # Tworzymy okno startowe
    start_window = tk.Toplevel(root)
    start_window.title("Start")
    start_label = tk.Label(start_window, text="let's start!" if j_en else "zaczynamy!", font=(font_name, 24))
    start_label.pack(pady=50)

    
    # Po 3 sekundach zamykamy okno startowe i pokazujemy główne
    def start_main_app():
        if dzwiek:
            subprocess.call("cvlc --play-and-exit dzwiek/poczatek.wav 2> /dev/null", shell=True)
        start_window.destroy()
        root.deiconify()  # Pokazujemy główne okno

    start_window.after(500, start_main_app)
    
    # Tworzymy widżety w GŁÓWNYM oknie, ale są ukryte
    root.title("Test" if sprawdzian else "Learning" if j_en else "Sprawdzian" if sprawdzian else "Nauka")

    font_obj = font.Font(family=font_name, size=font_size)
    text_width = font_obj.measure("Podaj tłumaczenie wyrazu: pomarańcza123456")
    root.minsize(text_width + 50, 200)

    question_label = tk.Label(root, text="", font=(font_name, font_size))
    question_label.pack(pady=10)

    entry = tk.Entry(root, font=(font_name, font_size))
    entry.pack(pady=10)
    entry.bind("<Return>", on_submit)

    submit_button = tk.Button(root, text="Submit" if j_en else "Zatwierdź", command=on_submit)
    submit_button.pack(pady=10)

    result_label = tk.Label(root, text="", font=(font_name, font_size_result))
    result_label.pack(pady=10)
    result_label2 = tk.Label(root, text="", font=(font_name, font_size))
    result_label2.pack(pady=10)

    remaining_label = tk.Label(root, text="", font=(font_name, font_size_remaining))
    remaining_label.pack(pady=10)

    false_previous=False
    i = losuj_numer(powtorz)
    if i is None:
        result_label.config(text="No more words to quiz!" if j_en else "Brak słów do quizu!")
    else:
        remaining = sum(powtorz)
        remaining_words = sum(1 for j in powtorz if j != 0)
        update_question()

    entry.focus_set()
    root.mainloop()

    
def main():
    global j_en, polski, angielski, powtorz, najtrudniejsze, dzwiek

    do_nauki_folder = "do_nauki"
    pliki_do_nauki = lista_plikow(do_nauki_folder)
    
    j_en = wybierz_jezyk()
    #j_en = simpledialog.askstring("Language", "Wybierz język/Set the language [PL,en]:", initialvalue="PL").lower() == "en"
    nazwa_pliku = wybierz_plik(pliki_do_nauki, j_en)

    polski, angielski = zaimportuj(os.path.join(do_nauki_folder, nazwa_pliku))
    powtorz = [1] * len(polski)
    najtrudniejsze = [0] * len(polski)

    if os.name == 'nt':
        messagebox.showinfo("Info", "sound is not supported on Windows" if j_en else "dźwięk nie jest obsługiwany pod Windows")
        dzwiek = False
    else:
        #dzwiek = messagebox.askyesno("Sound" if j_en else "Dźwięk", "Do you want to turn the sound on?" if j_en else "czy włączyć dźwięk?")
        dzwiek = CustomAskYesNo(
            title="Sound" if j_en else "Dźwięk",
            message="Do you want to turn the sound on?" if j_en else "Czy włączyć dźwięk?",
            j_en=j_en,
            default_yes=True  # Możesz dostosować domyślny przycisk (True/False)
        ).show()

    # Pytanie o kierunek nauki
    czy_polskie_slowka = CustomAskYesNo(
        title="Direction" if j_en else "Kierunek",
        message=f"Do you want to be asked about this direction: {angielski[0]} -> ?" if j_en else f"Czy chcesz być pytany w tym kierunku: {angielski[0]} -> ?",
        j_en=j_en,
        default_yes=True  # Możesz dostosować domyślny przycisk (True/False)
    ).show()
    
    # Pytanie o kierunek nauki
    czy_sprawdzian = CustomAskYesNo(
        title="Choose mode" if j_en else "Wybierz rodzaj",
        message="Do you want to do a quiz instead of learning?" if j_en 
                else "Czy chcesz zrobić sprawdzian zamiast nauki?",
        j_en=j_en,
        default_yes=True  # Możesz dostosować domyślny przycisk (True/False)
    ).show()
    

    if czy_polskie_slowka:
        nauka(angielski, polski, powtorz, dzwiek, czy_sprawdzian)
    else:
        nauka(polski, angielski, powtorz, dzwiek, czy_sprawdzian)

if __name__ == "__main__":
    main()
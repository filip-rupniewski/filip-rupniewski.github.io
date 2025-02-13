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
from tkinter import filedialog, messagebox, simpledialog, font 
#import threading

# Global variables
j_en = False
nazwa_pliku = ""
polski = []
angielski = []
powtorz = []
najtrudniejsze = []
dzwiek = False

def lista_plikow(ścieżka):
    pliki = []
    for nazwa in os.listdir(ścieżka):
        if nazwa.endswith('.csv'):
            pełna_ścieżka = os.path.join(ścieżka, nazwa)
            with open(pełna_ścieżka, 'r', encoding='utf-8') as f:
                liczba_linii = sum(1 for _ in f)
            pliki.append((nazwa, liczba_linii))
    return pliki

def wybierz_plik(pliki):
    def on_select():
        global nazwa_pliku
        selected_index = listbox.curselection()
        if selected_index:
            nazwa_pliku = pliki[selected_index[0]][0]
            root.destroy()
    def on_key(event):  # Funkcja obsługująca klawisze
        if event.keysym == "Up":  # Strzałka w górę
            current_index = listbox.curselection()
            if current_index:
                new_index = max(0, current_index[0] - 1)
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(new_index)
                listbox.activate(new_index)
        elif event.keysym == "Down":  # Strzałka w dół
            current_index = listbox.curselection()
            if current_index:
                new_index = min(len(pliki) - 1, current_index[0] + 1)
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(new_index)
                listbox.activate(new_index)
        elif event.keysym == "Return":  # Enter
            on_select()

    root = tk.Tk()
    root.title("Choose a file to learn" if j_en else "Wybierz plik do nauki")
    # Ustawienie szerokości i wysokości okna
    root.minsize(400, 100)  # Minimalna szerokość 400, minimalna wysokość 100
    listbox = tk.Listbox(root)
    listbox.pack(fill=tk.BOTH, expand=True)

    for idx, (nazwa, liczba) in enumerate(pliki):
        listbox.insert(tk.END, f"{idx + 1}: {nazwa} ({liczba} lines)" if j_en else f"{idx + 1}: {nazwa} ({liczba} linii)")

    select_button = tk.Button(root, text="Select" if j_en else "Wybierz", command=on_select)
    select_button.pack()
    
    # Powiązanie klawiszy z funkcją on_key
    root.bind("<Up>", on_key)
    root.bind("<Down>", on_key)
    root.bind("<Return>", on_key)

    # Ustawienie domyślnego zaznaczenia pierwszego elementu
    listbox.selection_set(0)
    listbox.activate(0)

    root.mainloop()

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
    text_widget = tk.Text(result_window, wrap=tk.WORD, font=("Arial", 12))
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
        nonlocal i, remaining, remaining_words
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
            result_label.config(text="correct!" if j_en else "dobrze!", fg="green")
            powtorz[i] = max(0, powtorz[i] - 1)
            if dzwiek:
                subprocess.call("cvlc --play-and-exit dzwiek/prawidlowa.wav 2> /dev/null", shell=True)
        else:
            poprawione = popraw(wyraz, angielski_normalized)
            if wyraz == "":
                poprawione = angielski_wyraz
            result_label.config(text=f"the correct answer is: {poprawione}" if j_en else f"prawidłowa odpowiedź to: {poprawione}", fg="red")
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
        nonlocal i
        i = losuj_numer(powtorz)
        if i is None:
            result_label.config(text="No more words to quiz!" if j_en else "Brak słów do quizu!")
            return

        question_label.config(text=f"Translate the word: {polski[i]}" if j_en else f"Podaj tłumaczenie wyrazu: {polski[i]}")
        remaining_label.config(text=f"{remaining} questions and {remaining_words} unique pairs remain." if j_en else f"{remaining} pytań oraz {remaining_words} różnych par do końca.")

        entry.focus_set()

    # Inicjalizujemy GŁÓWNE okno (jedno!)
    root = tk.Tk()
    root.withdraw()  # Ukrywamy na początku, aby nie pojawiało się od razu

    # Tworzymy okno startowe
    start_window = tk.Toplevel(root)
    start_window.title("Start")
    start_label = tk.Label(start_window, text="let's start!" if j_en else "zaczynamy!", font=("Arial", 24))
    start_label.pack(pady=50)

    if dzwiek:
        subprocess.call("cvlc --play-and-exit dzwiek/poczatek.wav 2> /dev/null", shell=True)

    # Po 3 sekundach zamykamy okno startowe i pokazujemy główne
    def start_main_app():
        start_window.destroy()
        root.deiconify()  # Pokazujemy główne okno

    start_window.after(1000, start_main_app)

    # Tworzymy widżety w GŁÓWNYM oknie, ale są ukryte
    root.title("Test" if sprawdzian else "Learning" if j_en else "Sprawdzian" if sprawdzian else "Nauka")

    font_obj = font.Font(family="Arial", size=14)
    text_width = font_obj.measure("Podaj tłumaczenie wyrazu: pomarańcza123456")
    root.minsize(text_width + 50, 200)

    question_label = tk.Label(root, text="", font=("Arial", 14))
    question_label.pack(pady=10)

    entry = tk.Entry(root, font=("Arial", 14))
    entry.pack(pady=10)
    entry.bind("<Return>", on_submit)

    submit_button = tk.Button(root, text="Submit" if j_en else "Zatwierdź", command=on_submit)
    submit_button.pack(pady=10)

    result_label = tk.Label(root, text="", font=("Arial", 14))
    result_label.pack(pady=10)

    remaining_label = tk.Label(root, text="", font=("Arial", 12))
    remaining_label.pack(pady=10)

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

    j_en = simpledialog.askstring("Language", "Wybierz język/Set the language [PL,en]:", initialvalue="PL").lower() == "en"

    wybierz_plik(pliki_do_nauki)

    polski, angielski = zaimportuj(os.path.join(do_nauki_folder, nazwa_pliku))
    powtorz = [1] * len(polski)
    najtrudniejsze = [0] * len(polski)

    if os.name == 'nt':
        messagebox.showinfo("Info", "sound is not supported on Windows" if j_en else "dźwięk nie jest obsługiwany pod Windows")
        dzwiek = False
    else:
        dzwiek = messagebox.askyesno("Sound", "Do you want to turn the sound on?" if j_en else "czy włączyć dźwięk?")

    kierunek = simpledialog.askstring("Direction", "Choose translation direction: Which column do you want to learn [1/2, default 2]?" if j_en else "wybierz kierunek tłumaczenia: Której kolumny chcesz się uczyć [1/2, domyślnie 2]?", initialvalue="2")

    sprawdzian = simpledialog.askstring("Mode", "Choose 1 for learning and 2 for test [1/2, default 1]?" if j_en else "wybierz 1 dla nauki i 2 dla sprawdzianu [1/2, domyślnie 1]?", initialvalue="1") == "2"

    if kierunek == "1":
        nauka(angielski, polski, powtorz, dzwiek, sprawdzian)
    else:
        nauka(polski, angielski, powtorz, dzwiek, sprawdzian)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
import sys
import time
import random
import os
import csv
import subprocess
from unidecode import unidecode
from datetime import datetime

# Function to list available files for learning
def list_files(path):
    files = []
    for name in os.listdir(path):
        if name.endswith('.csv'):
            full_path = os.path.join(path, name)
            with open(full_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            files.append((name, line_count))
    return files

# Function to select a file from the listed options
def select_file(files):
    print("Choose a file to learn:" if j_en else "Wybierz plik do nauki:")
    for idx, (name, count) in enumerate(files):
        print(f"{idx + 1}: {name} (number of lines: {count})" if j_en else f"{idx + 1}: {name} (liczba linii: {count})")
    while True:
        choice = (input("Write a number of a file: ") if j_en else input("Wpisz numer pliku: "))
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1][0]
        else:
            print("Please write a correct number." if j_en else "Proszę wpisać prawidłowy numer.")

# Function to import word pairs from a CSV file
def import_words(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=";")
        imported_list = [row for row in reader if not row[0].startswith('#')]
    polish, english = map(list, zip(*imported_list))
    return polish, english

# Function to save difficult words for further review
def save_difficult_words(polish, english, difficult, file_name):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'difficult_words/{file_name}_difficult_{date_str}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Polish", "English", "Repetitions"])
        for i, count in enumerate(difficult):
            if count > 0:
                writer.writerow([polish[i], english[i], count])
    print(f'Difficult word pairs were saved to {filename}' if j_en else f'Najtrudniejsze pary wyrazów zapisano do pliku {filename}')

# Function to highlight incorrect characters in the answer
def correct(incorrect, correct):
    result = ""
    for i in range(len(correct)):
        if i < len(incorrect) and incorrect[i] == correct[i]:
            result += correct[i]
        else:
            result += correct[i].upper()
    return result

# Main learning function
def learn(polish, english, repetitions, sound):
    if j_en:
        print("Let's start!")
    else: 
        print("Zaczynamy!")
    if sound: 
        subprocess.call("cvlc --play-and-exit sound/start.wav 2> /dev/null", shell=True)
    prev_wrong_index = None
    while any(repetitions):
        os.system('cls' if os.name == 'nt' else 'clear')
        if prev_wrong_index is not None:
            i = prev_wrong_index
            prev_wrong_index = None
        else:
            i = random.choices(range(len(repetitions)), weights=repetitions, k=1)[0]
        prompt = f"{polish[i]}: "
        answer = input(prompt)
        if answer == "q!":
            break
        answer_normalized = unidecode(answer).strip().lower()
        correct_normalized = unidecode(english[i]).strip().lower()
        if answer_normalized == correct_normalized or any(ans.strip().lower() == answer_normalized for ans in english[i].split('|')):
            print("Correct!" if j_en else "Dobrze!")
            repetitions[i] = max(0, repetitions[i] - 1)
            if sound: 
                subprocess.call("cvlc --play-and-exit sound/correct.wav 2> /dev/null", shell=True)
        else:
            corrected = correct(answer, english[i])
            print(f"{' ' * (13 + len(polish[i]))}Correct answer: {corrected}" if j_en else f"{' ' * (2 + len(polish[i]))}Prawidłowa odpowiedź: {corrected}")
            repetitions[i] += 2
            difficult[i] += 1
            if sound: 
                subprocess.call("cvlc --play-and-exit sound/wrong.wav 2> /dev/null", shell=True)
            prev_wrong_index = i
            input("[Enter]")
    if j_en:
        print("End of learning!")
    else:
        print("Koniec nauki!")
    save_difficult_words(polish, english, difficult, file_name)

# Setting learning parameters and running the program
learn_folder = "do_nauki"
learn_files = list_files(learn_folder)
j_en = input("Set the language [PL,en]: ").lower() == "en"
file_name = select_file(learn_files)
polish, english = import_words(os.path.join(learn_folder, file_name))
repetitions = [1] * len(polish)
difficult = [0] * len(polish)
if os.name == 'nt': 
    print("Sound is not supported on Windows" if j_en else "Dźwięk nie jest obsługiwany pod Windows")
    sound = 0
else:
    sound = input("Turn on sound [Y/n]? " if j_en else "Czy włączyć dźwięk [T/n]? ").lower() != "n"
direction = input("Choose translation direction [1/2, default 2]? " if j_en else "Wybierz kierunek tłumaczenia [1/2, domyślnie 2]? ")
if direction == "1":
    learn(english, polish, repetitions, sound)
else:
    learn(polish, english, repetitions, sound)

#!/usr/bin/env python
import sys

import time
import random
import os
import csv
import subprocess
from unidecode import unidecode  # Import the unidecode function
from datetime import datetime  # Import datetime for date handling

def lista_plikow(ścieżka):
    # Zwraca listę plików CSV z folderu oraz ich liczby linii
    pliki = []
    for nazwa in os.listdir(ścieżka):
        if nazwa.endswith('.csv'):
            pełna_ścieżka = os.path.join(ścieżka, nazwa)
            with open(pełna_ścieżka, 'r', encoding='utf-8') as f:
                liczba_linii = sum(1 for _ in f)
            pliki.append((nazwa, liczba_linii))
    return pliki

def wybierz_plik(pliki):
    # Wyświetla pliki i pozwala użytkownikowi wybrać jeden
    print("Choose a file to learn:" if j_en else "Wybierz plik do nauki:")
    for idx, (nazwa, liczba) in enumerate(pliki):
        print(f"{idx + 1}: {nazwa} (number of lines: {liczba})" if j_en else f"{idx + 1}: {nazwa} (liczba linii: {liczba})")
    
    while True:
        wybor = (input("Write a number of a file: ") if j_en else input("Wpisz numer pliku: "))
        if wybor.isdigit() and 1 <= int(wybor) <= len(pliki):
            return pliki[int(wybor) - 1][0]
        else:
            print("Please write a correct number." if j_en else "Proszę wpisać prawidłowy numer.")


def zaimportuj(nazwa_pliku):
    # Wczytywanie bazy danych - dwóch kolumn do nauki: polski, angielski
    with open(nazwa_pliku, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=";")
        zaimportowana_lista = list(reader)
    polski, angielski = map(list, zip(*zaimportowana_lista))
    return polski, angielski

def countdown(t):
    print("drukowanie")
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
    print('wydrukowano!\n')

def losuj_numer(tablica_powtorzen):
    # Losuje słowo do sprawdzenia
    suma_powtorzen = sum(tablica_powtorzen)
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

def wypisz_najtrudniejsze():
    # Wypisuje najtrudniejsze wyrazy
    print("\n")
    if sum(najtrudniejsze) == 0:
        print("Well done!" if j_en else "Dobra robota!")
        return
    if j_en:
        print("\nthe most difficult pairs of words:")
    else:
        print("\nnajtrudniejsze pary wyrazów:")
    for i in range(len(najtrudniejsze)):
        if najtrudniejsze[i] > 0:
            print(f'{polski[i]} - {angielski[i]}')
    # Zapisuje najtrudniejsze wyrazy do pliku CSV
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format current date with hour, minute, and second
    filename = f'najtrudniejsze/{nazwa_pliku}_najtrudniejsze_{date_str}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["1 język", "2 język", "Liczba powtórzeń"])
        for i in range(len(najtrudniejsze)):
            if najtrudniejsze[i] > 0:
                writer.writerow([polski[i], angielski[i], najtrudniejsze[i]])
    print(f'The most difficult pairs were written to {filename}' if j_en else f'Najtrudniejsze pary wyrazów zostały zapisane do pliku {filename}')

def nauka(polski, angielski, powtorz, dzwiek):
    if j_en:
        print("let's start!")
    else: 
        print("zaczynamy!")
    
    if dzwiek: 
        subprocess.call("cvlc --play-and-exit dzwiek/poczatek.wav 2> /dev/null", shell=True)
    
    prev_wrong_index = None  # Variable to store the index of the previous wrong answer
    
    while any(powtorz):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Choose the word to ask
        if prev_wrong_index is not None:
            i = prev_wrong_index  # Prioritize asking the previously incorrect word
            prev_wrong_index = None  # Reset after asking the wrong word again
        else:
            i = losuj_numer(powtorz)  # Normal random selection if no previous wrong word
        remaining = sum(powtorz)
        
        komunikat = str(powtorz)

        # Input section
        if j_en:
            komunikat += f"\n{remaining} questions remain.\n" + "To finish studying, write \"q!\" and press ENTER.\n" + f"Write the translation of the word {polski[i]}: "
        else:
            komunikat += f"\n{remaining} pytań do końca.\n" + "Żeby zakończyć naukę, wpisz \"q!\" i wciśnij ENTER.\n" + f"Podaj tłumaczenie wyrazu {polski[i]}: "
            
        wyraz = input(komunikat)

        if wyraz == "q!":
            break
        
        # Normalize both user input and the correct answer
        wyraz_normalized = unidecode(wyraz).strip().lower()
        angielski_normalized = unidecode(angielski[i]).strip().lower()

        if wyraz_normalized == angielski_normalized:
            print("correct!" if j_en else "dobrze!")
            powtorz[i] -= 1
            if dzwiek: 
                subprocess.call("cvlc --play-and-exit dzwiek/prawidlowa.wav 2> /dev/null", shell=True)
        else:
            poprawione = popraw(wyraz, angielski[i])
            print(f"{' ' * (11+len(polski[i]))}the correct answer is: {poprawione}" if j_en else f"{' ' * (2+len(polski[i]))}prawidłowa odpowiedź to: {poprawione}")
            powtorz[i] += 2
            najtrudniejsze[i] += 1
            if dzwiek: 
                subprocess.call("cvlc --play-and-exit dzwiek/bledna.wav 2> /dev/null", shell=True)
            prev_wrong_index = i  # Mark the current word as wrong to ask it again next
            input("[Enter]")
        sys.stdout.flush()  # Ensures that the message is written out
    
    if j_en:
        print("end of the learning!")
    else:
        print("koniec nauki!")
    
    wypisz_najtrudniejsze()  # Call the function to also save difficult words to a file
    
    if dzwiek: 
        subprocess.call("cvlc --play-and-exit dzwiek/koniec.wav 2> /dev/null", shell=True)


# Main logic
do_nauki_folder = "do_nauki"
pliki_do_nauki = lista_plikow(do_nauki_folder)

# Language selection
j_en = input("Wybierz język/Set the language [PL,en]: ").lower() == "en"

# Let the user choose a file to learn from
nazwa_pliku = wybierz_plik(pliki_do_nauki)

# Import the selected file
polski, angielski = zaimportuj(os.path.join(do_nauki_folder, nazwa_pliku))
powtorz = [1] * len(polski)
najtrudniejsze = [0] * len(polski)

# Sound handling
if os.name == 'nt': 
    print("sound is not supported on Windows" if j_en else "dźwięk nie jest obsługiwany pod Windows")
    dzwiek = 0
else:
    dzwiek = input("Do you want to turn the sound on [Y/n]? " if j_en else "czy włączyć dźwięk [T/n]? ").lower() != "n"

# Translation direction
kierunek = input("Choose translation direction: Which column do you want to learn [1/2, default 2]? " if j_en else "wybierz kierunek tłumaczenia: Której kolumny chcesz się uczyć [1/2, domyślnie 2]? ")

if kierunek == "1":
    nauka(angielski, polski, powtorz, dzwiek)
else:
    nauka(polski, angielski, powtorz, dzwiek)
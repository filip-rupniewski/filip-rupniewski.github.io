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
    
# def losuj_numer(tablica_powtorzen):
#     # Create a list of non-zero indices and their corresponding weights
#     non_zero_indices = [i for i in range(len(tablica_powtorzen)) if tablica_powtorzen[i] > 0]
#     if not non_zero_indices:
#         return None  # No words left to repeat
#     # Create a list of weights based on the values in tablica_powtorzen
#     weights = [tablica_powtorzen[i] for i in non_zero_indices]
#     # Choose a word based on weighted random selection
#     result = random.choices(non_zero_indices, weights=weights, k=1)[0]
#     print("losowanie" + str(result))
#     return result

def popraw(zly, dobry):
    # Funkcja do zaznaczania błędów
    wynik = ""
    for i in range(len(dobry)):
        if i < len(zly) and zly[i] == dobry[i]:
            wynik += dobry[i]
        else:
            wynik += dobry[i].upper()    
    return wynik

def wypisz_najtrudniejsze(sprawdzian=False):
    # Wypisuje najtrudniejsze wyrazy
    print("\n")
    if sum(najtrudniejsze) == 0:
        print("Well done!" if j_en else "Dobra robota!")
        return
    if j_en:
        print("\nthe most difficult pairs of words:")
    else:
        print("\nnajtrudniejsze pary wyrazów:")

    # Sortowanie malejące na podstawie wartości w tablicy 'najtrudniejsze', a potem alfabetyczne według polskich słów
    sorted_indices = sorted(
        range(len(najtrudniejsze)), 
        key=lambda i: (-najtrudniejsze[i], polski[i])
    )
    
    # Wypisywanie najtrudniejszych par
    for i in sorted_indices:
        if najtrudniejsze[i] > 0:
            print(f'{polski[i]} - {angielski[i]} : {najtrudniejsze[i]}')

    # Zapis do pliku CSV
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format current date with hour, minute, and second
    filename = f'najtrudniejsze/{nazwa_pliku}_najtrudniejsze_{date_str}.csv'
    if sprawdzian:
        filename = f'najtrudniejsze/{nazwa_pliku}_sprawdzian_najtrudniejsze_{date_str}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["1 język", "2 język", "Liczba powtórzeń"])
        
        # Zapisanie najtrudniejszych par do pliku CSV
        for i in sorted_indices:
            if najtrudniejsze[i] > 0:
                writer.writerow([polski[i], angielski[i], najtrudniejsze[i]])
    
    print(f'The most difficult pairs were written to {filename}' if j_en else f'Najtrudniejsze pary wyrazów zostały zapisane do pliku {filename}')


def nauka(polski, angielski, powtorz, dzwiek, sprawdzian=False):
    #main function which ask for translations
    #in case 'sprawdzian'==TRUE it ask about each pair only once
    if j_en:
        if sprawdzian:
            print("Test")
        print("let's start!")
    else: 
        if sprawdzian:
            print("Sprawdzian")
        print("zaczynamy!")
    
    if dzwiek: 
        subprocess.call("cvlc --play-and-exit dzwiek/poczatek.wav 2> /dev/null", shell=True)
    
    prev_wrong_index = None  # Variable to store the index of the previous wrong answer
    
    while any(powtorz):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Choose the word to ask
        if prev_wrong_index is not None and sprawdzian==False:
            i = prev_wrong_index  # Prioritize asking the previously incorrect word
            prev_wrong_index = None  # Reset after asking the wrong word again
        else:
            i = losuj_numer(powtorz)  # Normal random selection if no previous wrong word
        remaining = sum(powtorz)
        
        remaining_words = 0
        for j in powtorz:
            if j !=0:
                remaining_words+=1
        
        komunikat = str(powtorz)

        # Input section
        if j_en:
            komunikat += f"\n{remaining} questions and {remaining_words} unique pairs remain.\n" + "To finish studying, write \"q!\" and press ENTER.\n" + f"Write the translation of the word\n{' ' * 34}{polski[i]}: "
        else:
            komunikat += f"\n{remaining} pytań oraz {remaining_words} różnych par do końca.\n" + "Żeby zakończyć naukę, wpisz \"q!\" i wciśnij ENTER.\n" + f"Podaj tłumaczenie wyrazu\n{' ' * 25}{polski[i]}: "
            
        wyraz = input(komunikat)

        if wyraz == "q!":
            break
        
        # Normalize both user input and the correct answer
        wyraz_normalized = unidecode(wyraz).strip().lower()
        angielski_normalized = unidecode(angielski[i]).strip().lower()

        if wyraz_normalized == angielski_normalized:
            print("correct!" if j_en else "dobrze!")
            powtorz[i] = max(0, powtorz[i] - 1)  # Ensure the value doesn't go below zero
            if dzwiek: 
                subprocess.call("cvlc --play-and-exit dzwiek/prawidlowa.wav 2> /dev/null", shell=True)
        else:
            poprawione = popraw(wyraz, angielski[i])
            print(f"{' ' * (13+len(polski[i]))}the correct answer is: {poprawione}" if j_en else f"{' ' * (2+len(polski[i]))}prawidłowa odpowiedź to: {poprawione}")
            powtorz[i] += 2
            if sprawdzian:
                powtorz[i]=0
            najtrudniejsze[i] += 1
            if dzwiek: 
                subprocess.call("cvlc --play-and-exit dzwiek/bledna.wav 2> /dev/null", shell=True)
            prev_wrong_index = i  # Mark the current word as wrong to ask it again next
            input("[Enter]")
        sys.stdout.flush()  # Ensures that the message is written out
    
    if j_en:
        if sprawdzian:
            print("end of the test!")
        else:
            print("end of the learning!")
    else:
        if sprawdzian:
            print("koniec sprawdzianu!")
        else:
            print("koniec nauki!")
    
    wypisz_najtrudniejsze(sprawdzian)  # Call the function to also save difficult words to a file
    
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

# Choosing mode: learning or test
sprawdzian = input("Choose 1 for learning and 2 for test [1/2, default 1]? " if j_en else "wybierz 1 dla nauki i 2 dla sprawdzianu [1/2, domyślnie 1]? ")
sprawdzian = True if sprawdzian == "2" else False
    
if kierunek == "1":
    nauka(angielski, polski, powtorz, dzwiek, sprawdzian)
else:
    nauka(polski, angielski, powtorz, dzwiek, sprawdzian)

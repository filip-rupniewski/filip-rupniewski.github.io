#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import time

def ile_stron(plik):
    """Get the number of pages in the PDF file."""
    cmd = f"pdftk {plik} dump_data output | grep -i NumberOfPages"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error while retrieving page count:", result.stderr.decode())
        sys.exit(1)
    napis = result.stdout.decode()
    return int(napis.split()[1])

def countdown(t):
    """Display a countdown timer."""
    print("drukowanie")
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
    print('wydrukowano!\n')

def dodaj_puste(plik, pocz, liczba):
    """Add blank pages to the PDF to ensure a multiple of 4 pages."""
    k = liczba % 4
    cmd = f"pdftk {plik} cat {pocz}-{pocz + liczba - 1} output tymczasowe_drukowanie_zmiana0.pdf"
    subprocess.call(cmd, shell=True) 
    pocz = 1
    if k % 4 != 0:
        subprocess.call("echo '' | ps2pdf -sPAPERSIZE=a5 - pusta_strona000_a5.pdf", shell=True)
        
        while 0 < k < 4:
            cmd = f"pdftk tymczasowe_drukowanie_zmiana0.pdf pusta_strona000_a5.pdf cat output tymczasowe_drukowanie_zmiana1.pdf"
            subprocess.call(cmd, shell=True) 
            cmd = "pdftk tymczasowe_drukowanie_zmiana1.pdf cat output tymczasowe_drukowanie_zmiana0.pdf"
            subprocess.call(cmd, shell=True) 
            k += 1
            liczba += 1
            subprocess.call("rm tymczasowe_drukowanie_zmiana1.pdf", shell=True)
    return "tymczasowe_drukowanie_zmiana0.pdf", 1, liczba

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <pdf_file>")
        sys.exit(1)

    plik = f'"{sys.argv[1]}"'
    liczba_stron_dokumentu = ile_stron(plik)
    print(f"Ten dokument ma {liczba_stron_dokumentu} stron")

    n = 0
    while n == 0:
        pocz = int(input("Podaj strone poczatkowa: "))
        n = int(input("Podaj liczbe stron: "))
        if pocz + n - 1 > liczba_stron_dokumentu or pocz < 1 or n <= 0: 
            print("Liczba stron do druku wykracza poza liczbę stron dokumentu")
            n = 0

    k = int((n // 4) * 4)
    print(pocz, pocz + n - 1, k)
    nazwa = plik
    plik, pocz, n = dodaj_puste(plik, pocz, n)

    broszura_strony = ""
    for i in range(0, int(n / 4)):
        print(i)
        print(f"{pocz - 1 + int(n / 2) - 2 * i} {pocz - 1 + int(n / 2) + 2 * i + 1} {pocz - 1 + int(n / 2) + 2 * i + 2} {pocz - 1 + int(n / 2) - 2 * i - 1} ")
        broszura_strony += f"{pocz - 1 + int(n / 2) - 2 * i} {pocz - 1 + int(n / 2) + 2 * i + 1} {pocz - 1 + int(n / 2) + 2 * i + 2} {pocz - 1 + int(n / 2) - 2 * i - 1} "

    print("Kolejnosc stron to: ", broszura_strony)
    print(f"Plik do zrobienia broszury z {nazwa} to tymczasowe_drukowanie_zmiana3.pdf")

    cmd = f"pdftk tymczasowe_drukowanie_zmiana0.pdf cat {broszura_strony} output tymczasowe_drukowanie_zmiana3.pdf"
    subprocess.call(cmd, shell=True) 
    subprocess.call("pdfcrop --margins 15 tymczasowe_drukowanie_zmiana3.pdf tymczasowe_drukowanie_zmiana4.pdf > /dev/null", shell=True) 
    subprocess.call("evince tymczasowe_drukowanie_zmiana4.pdf 2> /dev/null", shell=True) 

    drukowac = "n"
#    drukowac = input("Czy wydrukowac? (y/n): ").lower()
    if drukowac == "n":
        print("Nie drukowac")
    else:
        print("Drukowac")
        print("\n lp -d doktoranci3piętro -o number-up=2 -o fit-to-page -o sides=two-sided-short-edge tymczasowe_drukowanie_zmiana4.pdf")
        print("\n rm tymczasowe_drukowanie_zmiana4.pdf")
        subprocess.call("lp -d doktoranci3piętro -o number-up=2 -o fit-to-page -o sides=two-sided-short-edge tymczasowe_drukowanie_zmiana4.pdf", shell=True) 
        countdown(n + 8 * n // 4)

    # Clean up temporary files
    subprocess.call("rm pusta_strona000_a5.pdf", shell=True) 
    subprocess.call("rm tymczasowe_drukowanie_zmiana0.pdf", shell=True) 
    subprocess.call("rm tymczasowe_drukowanie_zmiana3.pdf", shell=True) 
    subprocess.call("rm tymczasowe_drukowanie_zmiana4.pdf", shell=True) 

if __name__ == "__main__":
    main()

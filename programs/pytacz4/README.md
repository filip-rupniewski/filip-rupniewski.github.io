# Pytacz 3.0 (EN)

**Pytacz 3.0** helps you learn vocabulary.

The program asks you to translate words. If you answer a question correctly, the occurrence count for that question decreases by one. If you answer incorrectly, the occurrence count increases by two. Initially, each question starts with an occurrence count of 1.

You can see both the number of questions left and the number of unique questions remaining. When all occurrence counts reach 0, your learning session is over – you've mastered the vocabulary!

At the end of the session, the word pairs you struggled with the most are saved in the `"najtrudniejsze"` folder.

## Demo

You can watch a demonstration of how the program works. The video files are located in the `demo` folder. You can watch the english demo also [here](https://mega.nz/file/P51BkbKK#QeXFqUP9Gp2iOnp35FOTg4F8sIcvKGZINshEqhpfHic).


## Vocabulary

Each file in the `do_nauki` folder consists of two columns of words. By default, these are Polish words and their English translations (separated by semicolons). You can add or modify these files as needed. You can also add comments to the files by starting each line with a `#`.

## GUI

There are text-based and graphic versions of the program. In text-only mode, the current occurrence count for each question is displayed in a table at the top.  
The graphic version is recommended, as it offers the ability to read the asked words aloud. However, it is currently only supported on Linux (or Mac OS).

## Required Packages

- VLC (for sound, if desired)
- ffmpeg (for sound, if desired in windows) 
- Python 3
- Python libraries (installable via pip):
    - `unidecode`
    - `Levenshtein`
    - `pygame`
    - `pyttsx3` (for text-to-speech conversion, required only for the GUI version and without internet access)
    - `gTTS` (for text-to-speech conversion, required only for the GUI version)

for windows: 
You can install Python from windows store. Then ppen Command Prompt (`Win + R → type cmd → Enter`) and write: 
    ```Command Prompt
    pip install unidecode Levenshtein pygame pyttsx3 gTTS
    ```
for linux:
    ```bash
    sudo apt install -y python3
    pip install unidecode Levenshtein pygame pyttsx3 gTTS
    ```


## To Run the Program

1. Open the terminal or Command prompt in case of windows. To do it in windows type: `Win + R → type cmd → Enter`).
2. Navigate to the Pytacz folder. (on windows type `cd "Downloads\pytacz4"` if you downloaded all the files to `Downlads` folder)
2. Run one of the following commands, depending on your desired mode:

    ```bash
    python3 pytacz.py
    ```

    ```Command prompt (windows)
    python pytacz.py
    ```

    Or, for the GUI version:

    ```bash
    python3 pytacz_gui3.4.py
    ```

    ```Command prompt (windows)
    python pytacz_gui3.4.py
    ```

If you'd like to run the program inside a Python environment on Linux, in my case it is myenv, you can simply execute:

    ```bash
    bash run_pytacz_gui.sh
    ```
You can always change the corresponding name in run_pytacz_gui.sh to agree with your environment name.
    
# Pytacz 3.0 (PL)

**Pytacz 3.0** pomaga w nauce słownictwa.

Program prosi o tłumaczenie słów. Jeśli odpowiedź jest poprawna, liczba wystąpień danego pytania zmniejsza się o jeden. Jeśli odpowiedź jest niepoprawna, liczba wystąpień zwiększa się o dwa. Początkowo każde pytanie zaczyna się z liczbą wystąpień równą 1.

Pod pytaniem wyświetla się zarówno liczbę pozostałych pytań, jak i liczbę unikalnych pytań, które zostały. Gdy wszystkie liczby wystąpień osiągną 0, Twoja sesja nauki jest zakończona – nauczyłeś się słówek!

Pary słów, z którymi miałeś największe trudności, zostaną zapisane w folderze `"najtrudniejsze"`.

## Demo

Filmy demonstracyjne przedstawiające działanie programu znajdują się w folderze `demo`.  
Możesz także obejrzeć polskie demo w przeglądarce [tutaj](https://mega.nz/file/C90wBSTa#skcDnw5jHAjXC4mK3yImJaZngIzHJ01-vX7L3ADx78I).

Zmiany:

## Słownictwo

Każdy plik w folderze `do_nauki` składa się z dwóch kolumn słów. Domyślnie są to polskie słowa i ich angielskie tłumaczenia (oddzielone średnikami). Możesz dodać lub zmodyfikować te pliki w zależności od potrzeb. Możesz również dodawać komentarze do plików, zaczynając każdy wiersz od `#`.

## GUI

Program oferuje wersję tekstową oraz graficzną. W trybie tylko tekstowym liczba wystąpień dla każdego pytania jest wyświetlana w tabeli na górze.  
Zaleca się korzystanie z wersji graficznej, ponieważ oferuje ona możliwość odczytu na głos zadanych słów. Jednak jest ona obecnie obsługiwana tylko w systemie Linux (lub Mac OS).

## Wymagane pakiety

- VLC (do obsługi dźwięku, jeśli jest to wymagane)
- Python 3
- Biblioteki Pythona (do zainstalowania za pomocą pip):
    - `unidecode`
    - `python-Levenshtein`
    - `pygame`
    - `pyttsx3` (do konwersji tekstu na mowę, wymagane tylko dla wersji GUI i tylko w przypadku braku dostępu do internetu)
    - `gTTS` (do konwersji tekstu na mowę, wymagane tylko dla wersji GUI)


Windows: 
Można zainstalować Pythona z windows store. Następnie otwórz Wiersz poleceń (`Win + R → type cmd → Enter`) i wpisz: 
    ```Wiersz poleceń
    pip install unidecode Levenshtein pygame pyttsx3 gTTS
    ```
Linux:
    ```bash
    sudo apt install -y python3
    pip install unidecode Levenshtein pygame pyttsx3 gTTS
    ```


## Uruchomienie programu

1. W terminalu przejdź do folderu Pytacz.
2. Uruchom jedną z poniższych komend, w zależności od wybranego trybu:

    Tryb tekstowy:

    ```bash
    python3 pytacz.py
    ```

    ```Command prompt (windows)
    python pytacz.py
    ```

    Dla wersji GUI:

    ```bash
    python3 pytacz_gui3.4.py
    ```
    
    ```Command prompt (windows)
    python pytacz_gui3.4.py
    ```
    
Jeśli chcesz uruchomić program w oddzielnym środowisku Python pod linuxem, w tym przypadku `myenv`, po prostu wykonaj:

    ```bash
    bash run_pytacz_gui.sh
    ```
    
lub zmień `myenv` na odpowiednią nazwę w `run_pytacz_gui.sh`. 

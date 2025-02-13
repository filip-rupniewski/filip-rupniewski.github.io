# Pytacz 3.0

**Pytacz 3.0** helps you learn vocabulary.

The program asks you to translate words. If you answer a question correctly, the occurrence count for that question decreases by one. If you answer incorrectly, the occurrence count increases by two. Initially, each question starts with an occurrence count of 1.

You can see both the number of questions left and the number of unique questions remaining. When all occurrence counts reach 0, your learning session is over – you've mastered the vocabulary!

At the end of the session, the word pairs you struggled with the most are saved in the `"najtrudniejsze"` folder.

## Vocabulary

Each file in the `/do_nauki` folder consists of two columns of words. By default, these are Polish words and their English translations (separated by semicolons). You can add or modify these files as needed. You can also add comments to the files by starting each line with a `#`.

## GUI

There are text-based and graphic versions of the program. In text-only mode, the current occurrence count for each question is displayed in a table at the top.  
The graphic version is recommended, as it offers the ability to read the asked words aloud. However, it is currently only supported on Linux.

## Required Packages

- VLC (for sound, if desired)
- Python 3
- Python libraries (installable via pip):
    - `unidecode`
    - `python-Levenshtein`
    - `pyttsx3` (for text-to-speech conversion, required only for the GUI version)

## To Run the Program

1. In the terminal, navigate to the Pytacz folder.
2. Run one of the following commands, depending on your desired mode:

    ```bash
    python3 pytacz.py
    ```

    Or, for the GUI version:

    ```bash
    python3 pytacz_gui3.0.py
    ```

If you'd like to run the program inside a Python environment (e.g., myenv), you can simply execute:

    ```bash
    ./run_pytacz_gui.sh
    ```


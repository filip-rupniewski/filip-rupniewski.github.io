Pytacz 4.0 helps you learn vocabulary.

	The program asks for the translation of words. If you answer a question correctly, the occurrence of that question decreases by one. If you answer incorrectly, the occurrence increases by two. Initially, each question starts with an occurrence count of 1. The current number of occurrences for each question is displayed in a table at the top. When all numbers reach 0, your learning session is over – you’ve mastered the vocabulary! 
At the end of the session, the pairs of words you struggled with the most are saved in the "najtrudniejsze" folder.

Vocabulary:
	Each file in the /do_nauki folder consists of two columns of words. By default, these are Polish words and their English translations (separated by semicolons). You can add or modify these files as needed.

Required packages:
	VLC (for sound, if desired)
	Python 3
	unidecode Python library (installable via pip)

To run the program:
	In the terminal, navigate to the Pytacz folder and run:
python3 pytacz.py
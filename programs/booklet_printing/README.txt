PDF Booklet Printing Script

This script allows you to print a PDF file as a booklet on Linux. It crops margins, rearranges pages, and opens the file in a PDF viewer. You can then simply click print with the options for double-sided printing and short-edge binding.

Important Notes
Due to the cropping of the margins, pages with a small amount of text may change orientation from vertical to horizontal. As a result, if automatic rotation and resizing are enabled in the printer settings, these pages may be rotated unexpectedly.

Required Programs and Libraries
To run this script, you need to have the following programs and libraries installed:
pdfcrop
evince
pdftk
python3
ps2pdf

You can install each of them (on debian-based distributions, e.g. Ubuntu) by typing the following commands in the terminal:
sudo apt-get install texlive-extra-utils  # for pdfcrop
sudo apt-get install evince
sudo apt-get install pdftk
sudo apt-get install python3

Usage
To use the script, run the following command in the terminal in the folder with a script:

python3 drukowanieP6c.py your_file.pdf

Replace your_file.pdf with the path to the PDF file you want to print.
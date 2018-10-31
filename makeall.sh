#!/usr/bin/env zsh
pythonw task.py
cd report
latexmk --xelatex report.tex
open report.pdf

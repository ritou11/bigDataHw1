#!/usr/bin/env zsh
cd report/
rm meta/*
rm figure/*.png
latexmk -C report.tex

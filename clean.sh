#!/usr/bin/env zsh
rm report/*.log
rm report/figure/*.png
cd report/
latexmk -C report.tex

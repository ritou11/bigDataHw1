#!/usr/bin/env zsh
cd report/
rm meta/*
rm meta/fig/*
rm figure/*.png
latexmk -C report.tex

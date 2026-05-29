# Compile the AffBench paper.
# Requires: pdflatex, bibtex, and the ECCV 2026 style files in this directory.
# Download eccv2026.sty from https://eccv.ecva.net/Conferences/2026

PDF = main.pdf
SRC = main.tex
BIB = refs.bib

.PHONY: all clean view

all: $(PDF)

$(PDF): $(SRC) $(BIB)
	pdflatex $(SRC)
	bibtex main
	pdflatex $(SRC)
	pdflatex $(SRC)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz *.toc

view: $(PDF)
	xdg-open $(PDF) 2>/dev/null || open $(PDF)

# Makefile for paper/
PDF  = main.pdf
SRC  = main.tex
BIB  = references.bib
STY  = iclr2025_conference.sty

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC) $(BIB) $(STY)
	pdflatex -interaction=nonstopmode $(SRC)
	bibtex main
	pdflatex -interaction=nonstopmode $(SRC)
	pdflatex -interaction=nonstopmode $(SRC)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz $(PDF)

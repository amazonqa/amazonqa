all: compiletex compilebib linkbib linkreferences

.PHONY: clean

clean:
	rm *.aux *.bbl *.blg *.log *.out 2>/dev/null

compiletex:
	pdflatex conll2018

compilebib:
	bibtex conll2018

linkbib:
	pdflatex conll2018

linkreferences:
	pdflatex conll2018

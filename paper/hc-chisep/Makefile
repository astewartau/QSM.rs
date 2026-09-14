# HC-ChiSep manuscript
#
#   make          build hc-chisep.pdf
#   make clean    remove build artefacts
#   make final    build with all draft markers hidden (\DRAFTfalse)

TEX = hc-chisep

all: $(TEX).pdf

figures:
	python3 scripts/fig1_concept.py
	python3 scripts/fig2_pipeline.py

$(TEX).pdf: $(TEX).tex $(TEX).bib
	pdflatex -interaction=nonstopmode $(TEX) </dev/null >/dev/null
	bibtex $(TEX) >/dev/null || true
	pdflatex -interaction=nonstopmode $(TEX) </dev/null >/dev/null
	pdflatex -interaction=nonstopmode $(TEX) </dev/null | grep -E "^(!|LaTeX Warning: (Citation|Reference))" || true

final:
	sed -i 's/^\\DRAFTtrue$$/\\DRAFTfalse/' $(TEX).tex
	$(MAKE) all
	sed -i 's/^\\DRAFTfalse$$/\\DRAFTtrue/' $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).spl $(TEX).pdf

.PHONY: all figures final clean

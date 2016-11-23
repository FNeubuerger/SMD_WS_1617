all: Sheet.pdf

wait: FORCE | build
		 TEXINPUTS="$(call translate,build:)" \
		 BIBINPUTS=build: \
		 max_print_line=1048576 \
	 latexmk \
		 --lualatex \
		 --output-directory=build \
		 --interaction=nonstopmode \
		 --halt-on-error \
		 --pvc \
	 main.tex

Sheet.pdf: FORCE | build
	  TEXINPUTS="$(call translate,build:)" \
		BIBINPUTS=build: \
		max_print_line=1048576 \
	latexmk \
		--lualatex \
		--output-directory=build \
		--interaction=nonstopmode \
		--halt-on-error \
	main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean

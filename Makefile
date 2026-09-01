.PHONY: paper pdf watch clean

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PAPER_DIR := $(ROOT_DIR)/paper
PAPER_BUILD_DIR := $(PAPER_DIR)/build
PAPER_MAIN_PDF := $(PAPER_DIR)/main.pdf
PAPER_TEX := main.tex
LATEXMK ?= latexmk
LATEXMK_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error -pdf

paper:
	@cd $(PAPER_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) $(PAPER_TEX)
	@cp -f $(PAPER_BUILD_DIR)/main.pdf $(PAPER_MAIN_PDF)
	@echo "PDF generated: $(PAPER_MAIN_PDF)"

pdf: paper
	@echo "Alias: 'make pdf' is the same as 'make paper'."

watch:
	@cd $(PAPER_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) -pvc $(PAPER_TEX)

clean:
	@cd $(PAPER_DIR) && $(LATEXMK) -C $(PAPER_TEX) || true
	@rm -rf $(PAPER_BUILD_DIR)
	@rm -f $(PAPER_MAIN_PDF)
	@echo "Clean complete: paper build artifacts removed."

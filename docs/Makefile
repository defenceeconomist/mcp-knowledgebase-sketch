SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	@rm -rf "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

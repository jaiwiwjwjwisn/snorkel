# Minimal makefile for Sphinx documentation
#

# This makefile is a minimal configuration for building documentation
# using the Sphinx documentation generator.

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=    # Optional arguments to pass to Sphinx build process
SPHINXBUILD   ?= sphinx-build   # Path to the Sphinx executable
SOURCEDIR     = .   # Directory containing the source documentation files
BUILDDIR      = _build  # Directory for the built documentation files

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


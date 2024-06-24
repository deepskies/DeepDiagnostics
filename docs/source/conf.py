import sys

sys.path.append("../src/deepdiagnostics")

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "deepdiagnostics"
copyright = "2024, Becky Nevin, M Voetberg, Brian Nord"
author = "Becky Nevin, M Voetberg, Brian Nord"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    'sphinxcontrib.bibtex'
]
bibtex_bibfiles = ['ref.bib']
napoleon_use_param = True
autodoc_default_options = {
    "members": True,
}
autodoc_typehints = "description"
autoclass_content = "class"
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pyramid"
html_static_path = ["_static"]

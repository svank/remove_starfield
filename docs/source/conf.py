# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import version as get_version

from packaging.version import Version

sys.path.insert(0, os.path.abspath("../.."))


project = "remove_starfield"
copyright = "2024, Sam Van Kooten"
author = "Sam Van Kooten"

# The full version, including alpha/beta/rc tags
release: str = get_version("remove_starfield")
version: str = release
_version = Version(release)
if _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = []

default_role = "any"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = True
html_static_path = ["_static"]
html_css_files = ['css/custom.css']
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/svank/remove_starfield",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "show_nav_level": 1,
    "show_toc_level": 3,
}
html_context = {
    "github_user": "svank",
    "github_repo": "remove_starfield",
    "github_version": "main",
    "doc_path": "docs/source/",
}


autoapi_dirs = ["../../remove_starfield"]
autoapi_ignore = ["*tests*"]
autoapi_options = [ 'members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members']

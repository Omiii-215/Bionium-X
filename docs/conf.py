import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Bionium-X'
copyright = '2026, Bionium-X Team'
author = 'Bionium-X Team'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

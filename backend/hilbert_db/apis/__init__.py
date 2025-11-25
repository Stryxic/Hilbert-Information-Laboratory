"""
Hilbert DB - API Layer

This package defines the HTTP-facing functional API for:

    - graphs
    - elements
    - molecules
    - stability metrics
    - search

Each module defines pure function signatures that the web layer can
bind to (FastAPI, Flask, etc.).
"""

# Import modules so that module-level __all__ can be accessed
from . import graph_api
from . import elements_api
from . import molecules_api
from . import stability_api
from . import search_api

# Re-export all API functions and dataclasses
from .graph_api import *
from .elements_api import *
from .molecules_api import *
from .stability_api import *
from .search_api import *

# Build a combined __all__
__all__ = (
    graph_api.__all__
    + elements_api.__all__
    + molecules_api.__all__
    + stability_api.__all__
    + search_api.__all__
)

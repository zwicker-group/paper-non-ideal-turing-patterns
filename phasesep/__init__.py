"""
python code for studying phase separation using numerical simulations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.fields import *
from pde.grids import *
from pde.solvers import *
from pde.storage import *
from pde.trackers import *
from pde.visualization import *

from .free_energies import *
from .pdes import *
from .reactions import *

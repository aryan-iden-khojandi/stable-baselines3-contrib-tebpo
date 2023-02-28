import os

from sb3_contrib.ars import ARS
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.tqc import TQC
from sb3_contrib.trpo import TRPO, TRPO_ANALYSIS
from sb3_contrib.tebpo import TEBPO, TEBPO_MC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

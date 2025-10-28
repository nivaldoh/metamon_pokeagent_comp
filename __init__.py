# Bridge the repository's top-level 'metamon' package to the inner package
# so that absolute imports like `from metamon import SUPPORTED_BATTLE_FORMATS`
# resolve correctly even when this repo's folder shadows an installed package.

from .metamon import *  # noqa: F401,F403


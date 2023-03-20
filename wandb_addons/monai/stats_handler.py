from typing import Any, Callable, TYPE_CHECKING

from monai.config import IgniteInfo
from monai.utils import optional_import, min_version

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine",
        IgniteInfo.OPT_IMPORT_VERSION,
        min_version,
        "Engine",
        as_type="decorator",
    )


class WandbStatsHandler:
    def __init__(self):
        pass

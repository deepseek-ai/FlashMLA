"""Shared helpers of the phase-3 prototypes."""
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def build_dir(name):
    """Build directory of an inline extension: $FLASHMLA_P3_BUILD/<name> (default: <system temp dir>/flashmla_p3/<name>)."""
    d = os.path.join(os.environ.get("FLASHMLA_P3_BUILD", os.path.join(tempfile.gettempdir(), "flashmla_p3")), name)
    os.makedirs(d, exist_ok=True)
    return d

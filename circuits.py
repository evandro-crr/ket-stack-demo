"""Ket circuit compilation, run in a separate worker process by app.py.

ket and Qiskit are imported only inside the worker (see `_load_ket`), so the
Streamlit server process can import this module without loading them. That
way a crash or memory spike in their native code can't take the app down.
"""

from __future__ import annotations

import io


def _load_ket():
    """Bind the ket names used by `diffusion` as module globals."""
    global CZ, H, X, around, cat
    from ket import CZ, H, X, around, cat


def diffusion(qubits: Quant):
    with around(cat(H, X), qubits):
        CZ(*qubits)


def draw_svg(
    num_qubits: int,
    qpu_size: int | None = None,
    decompose: bool = False,
    coupling_graph: list[tuple[int, int]] | None = None,
    gateset_name: str | None = None,
) -> str:
    """Draw the diffusion circuit at one compilation stage, as SVG markup.

    SVG instead of PNG: the native-gate diagrams are tens of thousands of pixels
    tall, and rasterizing them takes gigabytes of memory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ket import gateset, qulib

    _load_ket()
    native = getattr(gateset, gateset_name)() if gateset_name else None
    fig = qulib.draw(
        diffusion,
        num_qubits,
        qpu_size=qpu_size,
        decompose=decompose,
        coupling_graph=coupling_graph,
        native_gate_set=native,
        fold=60,
        output="mpl",
    )
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

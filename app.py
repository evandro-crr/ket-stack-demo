import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from ket import *
from ket import gateset
import sympy as sp
from sympy import pi, I, exp, cos, sin, Matrix, eye
from sympy.physics.quantum import TensorProduct as kron

st.set_page_config(
    page_title="Ket compilation example",
    page_icon=":material/device_hub:",
    layout="wide",
)

# ── Sympy gate matrices ────────────────────────────────────────────────────────
I_mat = eye(2)
sp_X = Matrix([[0, 1], [1, 0]])
sp_Y = Matrix([[0, -I], [I, 0]])
sp_Z = Matrix([[1, 0], [0, -1]])


def Rx(t):
    return cos(t / 2) * I_mat - I * sin(t / 2) * sp_X


def Ry(t):
    return cos(t / 2) * I_mat - I * sin(t / 2) * sp_Y


def Rz(t):
    return Matrix([[exp(-I * t / 2), 0], [0, exp(I * t / 2)]])


def Rzz(t):
    return Matrix(
        [
            [exp(-I * t / 2), 0, 0, 0],
            [0, exp(I * t / 2), 0, 0],
            [0, 0, exp(I * t / 2), 0],
            [0, 0, 0, exp(-I * t / 2)],
        ]
    )


def GPi(p):
    return Matrix([[0, exp(-I * p)], [exp(I * p), 0]])


def GPi2(p):
    return (1 / sp.sqrt(2)) * Matrix([[1, -I * exp(-I * p)], [-I * exp(I * p), 1]])


sp_SX = (1 / 2) * Matrix([[1 + I, 1 - I], [1 - I, 1 + I]])
sp_iSWAP = Matrix([[1, 0, 0, 0], [0, 0, I, 0], [0, I, 0, 0], [0, 0, 0, 1]])
sp_CZ = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
sp_ECR = (eye(4) - I * kron(sp_X, sp_Y)) / sp.sqrt(2)
sp_MS = (eye(4) - I * kron(sp_X, sp_X)) / sp.sqrt(2)

theta_sym, phi_sym = sp.symbols("theta phi", real=True)

GATESET_MATRICES = {
    "RzGPiGPi2MS": [
        (r"R_z(\theta)", lambda: Rz(theta_sym)),
        (r"GPi(\phi)", lambda: GPi(phi_sym)),
        (r"GPi_2(\phi)", lambda: GPi2(phi_sym)),
        (r"MS", lambda: sp_MS),
    ],
    "RzRxISWAP": [
        (r"R_z(\theta)", lambda: Rz(theta_sym)),
        (r"R_x(\theta)", lambda: Rx(theta_sym)),
        (r"iSWAP", lambda: sp_iSWAP),
    ],
    "RzSXCZ": [
        (r"R_z(\theta)", lambda: Rz(theta_sym)),
        (r"\sqrt{X}", lambda: sp_SX),
        (r"CZ", lambda: sp_CZ),
    ],
    "RzSXECR": [
        (r"R_z(\theta)", lambda: Rz(theta_sym)),
        (r"\sqrt{X}", lambda: sp_SX),
        (r"ECR", lambda: sp_ECR),
    ],
    "RzSXXRzz": [
        (r"R_z(\theta)", lambda: Rz(theta_sym)),
        (r"\sqrt{X}", lambda: sp_SX),
        (r"X", lambda: sp_X),
        (r"R_{zz}(\theta)", lambda: Rzz(theta_sym)),
    ],
}


# ── Ket diffusion circuit ──────────────────────────────────────────────────────
def diffusion(qubits: Quant):
    with around(cat(H, X), qubits):
        CZ(*qubits)


def generate_grid_edges(n: int, m: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(n):
        for c in range(m):
            node = r * m + c
            if c < m - 1:
                edges.append((node, node + 1))
            if r < n - 1:
                edges.append((node, node + m))
    return edges


import io


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def cached_high_level(num_qubits: int) -> bytes:
    return _fig_to_bytes(qulib.draw(diffusion, num_qubits, output="mpl", fold=60))


@st.cache_data(show_spinner=False)
def cached_decomposed(num_qubits: int, qpu_size: int) -> bytes:
    return _fig_to_bytes(
        qulib.draw(
            diffusion,
            num_qubits,
            decompose=True,
            fold=60,
            qpu_size=qpu_size,
            output="mpl",
        )
    )


@st.cache_data(show_spinner=False)
def cached_mapped(num_qubits: int, qpu_size: int, coupling_graph: tuple) -> bytes:
    return _fig_to_bytes(
        qulib.draw(
            diffusion,
            num_qubits,
            qpu_size=qpu_size,
            coupling_graph=list(coupling_graph),
            fold=60,
            output="mpl",
        )
    )


@st.cache_data(show_spinner=False)
def cached_native(
    num_qubits: int, qpu_size: int, coupling_graph: tuple, gateset_name: str
) -> bytes:
    gs_cls = getattr(gateset, gateset_name)
    return _fig_to_bytes(
        qulib.draw(
            diffusion,
            num_qubits,
            qpu_size=qpu_size,
            coupling_graph=list(coupling_graph),
            native_gate_set=gs_cls(),
            fold=60,
            output="mpl",
        )
    )


# ── Sidebar: all controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/tune: Configuration")

    st.subheader("Circuit")
    num_qubits = st.slider("Number of qubits", min_value=2, max_value=7, value=7)

    st.subheader("QPU")
    col_n, col_m = st.columns(2)
    grid_n = col_n.number_input("Rows", min_value=1, max_value=10, value=4)
    grid_m = col_m.number_input("Columns", min_value=1, max_value=10, value=4)
    qpu_size = int(grid_n * grid_m)
    st.caption(f":material/memory: **{qpu_size} qubits** on the QPU")

    st.subheader("Native gateset")
    selected_gateset = st.selectbox(
        "Gateset",
        list(GATESET_MATRICES.keys()),
        label_visibility="collapsed",
    )

    st.space("small")
    run_button = st.button(
        ":material/play_arrow: Update circuits",
        type="primary",
        use_container_width=True,
    )

    if qpu_size < num_qubits:
        st.warning(
            f"QPU has only {qpu_size} qubits but circuit needs {num_qubits}. "
            "Mapping will be skipped.",
            icon=":material/warning:",
        )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Ket compilation example", anchor=False)

coupling_graph = generate_grid_edges(grid_n, grid_m)

top_left, top_right = st.tabs(
    [
        ":material/device_hub: QPU coupling graph",
        f":material/calculate: {selected_gateset} matrices",
    ]
)

# QPU graph
with top_left:
    with st.container(border=True):
        st.subheader(":material/device_hub: QPU coupling graph", anchor=False)
        fig_graph, ax = plt.subplots(figsize=(4, 4))
        fig_graph.patch.set_alpha(0)
        ax.set_facecolor("none")
        G = nx.Graph()
        G.add_nodes_from(range(qpu_size))
        G.add_edges_from(coupling_graph)
        pos = {r * grid_m + c: (c, -r) for r in range(grid_n) for c in range(grid_m)}
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_color="#4F8BF9",
            font_color="white",
            node_size=600,
            font_weight="bold",
            edge_color="#888",
            width=2,
        )
        st.pyplot(fig_graph, width="stretch")

# Native gateset matrices
with top_right:
    with st.container(border=True):
        st.subheader(f":material/calculate: {selected_gateset} matrices", anchor=False)
        entries = GATESET_MATRICES[selected_gateset]
        for i in range(0, len(entries), 2):
            pair = entries[i : i + 2]
            cols = st.columns(2)
            for col, (label, mat_fn) in zip(cols, pair):
                with col:
                    st.latex(f"{label} = {sp.latex(mat_fn())}")

st.space("small")

# Diffusion code
with st.expander(":material/code: Diffusion function", expanded=True):
    st.code(
        "def diffusion(qubits: Quant):\n"
        "    with around(cat(H, X), qubits):\n"
        "        CZ(*qubits)",
        language="python",
    )

st.space("small")

# ── Detect settings drift and invalidate stale results ────────────────────────
current_config = {
    "num_qubits": num_qubits,
    "grid_n": int(grid_n),
    "grid_m": int(grid_m),
    "selected_gateset": selected_gateset,
}

if "circuit_results" in st.session_state:
    stored_config = st.session_state.circuit_results.get("config", {})
    if stored_config != current_config:
        del st.session_state.circuit_results

# ── Circuit compilation (triggered by button) ──────────────────────────────────
if run_button:
    c_coupling = generate_grid_edges(grid_n, grid_m)
    results = {"config": current_config}
    c_coupling_tuple = tuple(c_coupling)

    with st.status(":material/sync: Compiling circuits…", expanded=True) as status:
        st.write(":material/bolt: High-level circuit…")
        results["img_high"] = cached_high_level(num_qubits)

        st.write(":material/compress: Decomposed circuit…")
        results["img_dec"] = cached_decomposed(num_qubits, qpu_size)

        if qpu_size >= num_qubits:
            st.write(":material/map: Mapped circuit…")
            try:
                results["img_map"] = cached_mapped(
                    num_qubits, qpu_size, c_coupling_tuple
                )
            except Exception as e:
                results["map_error"] = str(e)

            st.write(":material/memory: Native gate circuit…")
            try:
                results["img_native"] = cached_native(
                    num_qubits, qpu_size, c_coupling_tuple, selected_gateset
                )
            except Exception as e:
                results["native_error"] = str(e)
        else:
            results["qpu_size_error"] = True

        status.update(
            label=":material/check_circle: Compilation complete",
            state="complete",
            expanded=False,
        )

    st.session_state.circuit_results = results
    st.toast("Circuits updated!", icon=":material/check:")

# ── Display stored results ─────────────────────────────────────────────────────
if "circuit_results" in st.session_state:
    res = st.session_state.circuit_results

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            ":material/bolt: High-level",
            ":material/compress: Decomposed",
            ":material/map: Mapped",
            ":material/memory: Native gates",
        ]
    )

    with tab1:
        st.image(res["img_high"])

    with tab2:
        st.image(res["img_dec"])

    with tab3:
        if res.get("qpu_size_error"):
            st.error(
                "QPU too small to map this circuit. Increase QPU size in the sidebar.",
                icon=":material/error:",
            )
        elif "map_error" in res:
            st.error(f"Mapping failed: {res['map_error']}", icon=":material/error:")
        elif "img_map" in res:
            st.image(res["img_map"])

    with tab4:
        if res.get("qpu_size_error"):
            st.error(
                "QPU too small — native gate compilation skipped.",
                icon=":material/error:",
            )
        elif "native_error" in res:
            st.error(
                f"Compilation failed: {res['native_error']}", icon=":material/error:"
            )
        elif "img_native" in res:
            st.image(res["img_native"])
else:
    st.info(
        "Configure the parameters in the sidebar and click **Update circuits** to compile.",
        icon=":material/info:",
    )

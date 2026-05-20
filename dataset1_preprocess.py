import numpy as np
import h5py
import xml.etree.ElementTree as ET
import os
import pickle


ENERGY_RANGE = {
    "photon": (256, 4194304),
    "pion":   (256, 4194304),
}

TARGET_ALPHA = 10


def parse_binning_xml(xml_path: str, particle: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for bin_node in root.findall("Bin"):
        if bin_node.attrib["name"] == particle:
            lay_ids     = []
            lay_r_edges = []
            lay_alphas  = []
            for i, layer in enumerate(bin_node.findall("Layer")):
                r_edges = list(map(float, layer.attrib["r_edges"].split(",")))
                n_alpha = int(layer.attrib["n_bin_alpha"])
                lay_ids.append(i)
                lay_r_edges.append(r_edges)
                lay_alphas.append(n_alpha)
            return lay_ids, lay_r_edges, lay_alphas

    raise ValueError(f"particle '{particle}' not found in {xml_path}")


def get_valid_layers(lay_ids, lay_r_edges):
    """Valid layers: layers with n_r > 0 (i.e. r_edges count > 1)"""
    return [i for i, r in zip(lay_ids, lay_r_edges) if len(r) > 1]


def get_all_voxel_counts(xml_path: str, particle: str):
    _, lay_r_edges, lay_alphas = parse_binning_xml(xml_path, particle)
    return [
        lay_alphas[i] * (len(lay_r_edges[i]) - 1)
        for i in range(len(lay_alphas))
    ]


def get_max_r_bins(lay_r_edges: list):
    """Get the max radial bin count across all valid layers"""
    max_r = 0
    for r_edges in lay_r_edges:
        n_r = len(r_edges) - 1
        if n_r > max_r:
            max_r = n_r
    return max_r


def build_weight_mats(lay_r_edges: list, cache_path: str = "data/weight_mats.pkl"):
    """Build radial area weight matrices (original method)"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] load weight_mats from {cache_path}")
        return data["all_r_edges"], data["weight_mats"]

    print("[INFO] building weight_mats...")

    all_edges = set()
    for r_edges in lay_r_edges:
        all_edges.update(r_edges)
    all_r_edges = sorted(all_edges)
    M = len(all_r_edges) - 1

    weight_mats = []
    for r_edges in lay_r_edges:
        n_r = len(r_edges) - 1
        W   = np.zeros((M, n_r), dtype=np.float32)
        for src in range(n_r):
            r_lo = r_edges[src]
            r_hi = r_edges[src + 1]
            for dst in range(M):
                lo = all_r_edges[dst]
                hi = all_r_edges[dst + 1]
                overlap = max(0.0, min(r_hi, hi) - max(r_lo, lo))
                if overlap > 0:
                    W[dst, src] = overlap / (r_hi - r_lo)
        weight_mats.append(W)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({"all_r_edges": all_r_edges, "weight_mats": weight_mats}, f)
    print(f"[INFO] saved weight_mats -> {cache_path}; r_edges: {all_r_edges}")

    return all_r_edges, weight_mats


def build_mask_info(lay_r_edges: list):
    """
    Build info needed for mask method.
    Returns:
        max_r_bins: max radial bin count
        mask_list: per-layer mask array, shape=(max_r_bins,), 1=valid, 0=masked
    """
    max_r_bins = get_max_r_bins(lay_r_edges)
    mask_list = []
    
    for r_edges in lay_r_edges:
        n_r = len(r_edges) - 1
        mask = np.zeros(max_r_bins, dtype=np.float32)
        # First n_r bins are the valid region
        mask[:n_r] = 1.0
        mask_list.append(mask)
    
    print(f"[INFO] mask method: max_r_bins={max_r_bins}")
    return max_r_bins, mask_list


def resample_alpha(layer: np.ndarray, src_alpha: int, dst_alpha: int) -> np.ndarray:
    """Angular resampling (unchanged)"""
    if src_alpha == dst_alpha:
        return layer

    src_edges = np.linspace(0, 1, src_alpha + 1)
    dst_edges = np.linspace(0, 1, dst_alpha + 1)

    W_a = np.zeros((dst_alpha, src_alpha), dtype=np.float32)
    for src in range(src_alpha):
        a_lo   = src_edges[src]
        a_hi   = src_edges[src + 1]
        da_src = a_hi - a_lo
        for dst in range(dst_alpha):
            lo      = dst_edges[dst]
            hi      = dst_edges[dst + 1]
            overlap = max(0.0, min(a_hi, hi) - max(a_lo, lo))
            if overlap > 0:
                W_a[dst, src] = overlap / da_src

    return np.einsum('nsm,ds->ndm', layer, W_a).astype(np.float32)


def normalize_energy(energies: np.ndarray, particle: str) -> np.ndarray:
    e_min, e_max = ENERGY_RANGE[particle]
    log_e   = np.log10(energies)
    log_min = np.log10(e_min)
    log_max = np.log10(e_max)
    return (log_e - log_min) / (log_max - log_min + 1e-9)



'''
class Dataset1Preprocessor:

    def __init__(self, hdf5_path, xml_path, particle,
                 weight_cache="data/weight_mats.pkl"):
        self.hdf5_path    = hdf5_path
        self.xml_path     = xml_path
        self.particle     = particle
        self.weight_cache = weight_cache
        self.e_min, self.e_max = ENERGY_RANGE[particle]

        self.lay_ids, self.lay_r_edges, self.lay_alphas = parse_binning_xml(
            xml_path, particle
        )
        self.all_r_edges, self.weight_mats = build_weight_mats(
            self.lay_r_edges, cache_path=weight_cache
        )

        # Valid layers: n_r > 0 (r_edges count > 1)
        self.valid_layer_indices = [
            i for i, r in enumerate(self.lay_r_edges) if len(r) > 1
        ]
        self.n_valid_layers = len(self.valid_layer_indices)
        self.M = len(self.all_r_edges) - 1

        # Output grid: valid layers × fixed alpha × unified radial
        self.volume_size = (self.n_valid_layers, TARGET_ALPHA, self.M)
        print(f"[INFO] output grid: "
              f"{self.n_valid_layers} valid layers × "
              f"{TARGET_ALPHA} alpha_bins × "
              f"{self.M} r_bins")

    def normalize_energy(self, energies: np.ndarray) -> np.ndarray:
        return normalize_energy(energies, self.particle)

    def process(self, max_samples: int = None):
        with h5py.File(self.hdf5_path, "r") as f:
            showers  = f["showers"][:max_samples].astype(np.float32)
            energies = f["incident_energies"][:max_samples].astype(np.float32)

        energies = self.normalize_energy(energies)

        all_counts = get_all_voxel_counts(self.xml_path, self.particle)
        bin_starts = [sum(all_counts[:lid]) for lid in self.lay_ids]
        bin_ends   = [
            bin_starts[i] + self.lay_alphas[i] * (len(self.lay_r_edges[i]) - 1)
            for i in range(len(self.lay_ids))
        ]

        N = showers.shape[0]
        n_valid = self.n_valid_layers
        M       = self.M

        # Output volume: [N, n_valid_layers, TARGET_ALPHA, M]
        volume = np.zeros((N, n_valid, TARGET_ALPHA, M), dtype=np.float32)

        for out_idx, i in enumerate(self.valid_layer_indices):
            n_a   = self.lay_alphas[i]
            n_r   = len(self.lay_r_edges[i]) - 1

            # Extract raw layer data: [N, n_a, n_r]
            layer = showers[:, bin_starts[i]:bin_ends[i]].reshape(N, n_a, n_r)

            # Radial interpolation to unified grid: [N, n_a, M]
            layer = np.einsum('nar,mr->nam', layer, self.weight_mats[i])

            # Angular resample to TARGET_ALPHA: [N, TARGET_ALPHA, M]
            layer = resample_alpha(layer, n_a, TARGET_ALPHA)  # [N, TARGET_ALPHA, M]

            volume[:, out_idx] = layer  # Direct assignment, no transpose needed

        grid = np.log10(volume + 1.0)
        self.vmax = grid.max()
        grid = grid / (self.vmax + 1e-9) * 2 - 1

        # Output: [N, 1, n_valid_layers, TARGET_ALPHA, M]
        return (
            grid.reshape(N, 1, n_valid, TARGET_ALPHA, M).astype(np.float32),
            energies.astype(np.float32),
        )
'''
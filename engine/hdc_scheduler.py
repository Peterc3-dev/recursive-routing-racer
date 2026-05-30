"""
Hyperdimensional Computing scheduler for R.A.G-Race-Router.

Replaces the neural policy gradient scheduler with geometric
field resolution via codebook similarity search.

Instead of: if gpu_temp > 75: route_to_npu()
This: encode state as hypervector -> find nearest codebook match -> snap to decision

The codebook is built from experience, not trained via backpropagation.
Each good routing decision becomes a stored pattern.
"""
import torch
import numpy as np
import os
import json
from typing import Dict, Tuple

try:
    import torchhd
    HDC_AVAILABLE = True
except ImportError:
    HDC_AVAILABLE = False

DIMENSIONS = 10000  # Hypervector dimensionality
DEVICES = ['cpu', 'gpu', 'npu']


class HdcScheduler:
    def __init__(self, dimensions: int = DIMENSIONS):
        self.d = dimensions
        self.codebook = []  # List of (state_hv, device, metadata)
        self._codebook_dirty = True
        self.save_path = os.path.expanduser("~/.rag-race-router/hdc_codebook.json")

        if not HDC_AVAILABLE:
            print("[HDC] torchhd not available, falling back to cosine similarity")

        # Basis hypervectors for encoding continuous values
        # Each feature gets a random basis vector
        self.basis = {
            'gpu_temp': self._random_hv(),
            'gpu_util': self._random_hv(),
            'gpu_vram': self._random_hv(),
            'cpu_load': self._random_hv(),
            'npu_available': self._random_hv(),
            'op_size': self._random_hv(),
            'op_type_matmul': self._random_hv(),
            'op_type_attention': self._random_hv(),
            'op_type_embed': self._random_hv(),
            'op_type_normalize': self._random_hv(),
            'op_type_conv': self._random_hv(),
            'op_type_other': self._random_hv(),
        }

        # Device label hypervectors
        self.device_hvs = {
            'cpu': self._random_hv(),
            'gpu': self._random_hv(),
            'npu': self._random_hv(),
        }

        # Try to load existing codebook
        self._load()

    def _random_hv(self) -> torch.Tensor:
        """Generate a random bipolar hypervector."""
        if HDC_AVAILABLE:
            return torchhd.random(1, self.d).squeeze(0)
        else:
            return torch.sign(torch.randn(self.d))

    def _bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bind two hypervectors (XOR-like for bipolar)."""
        if HDC_AVAILABLE:
            return torchhd.bind(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
        else:
            return a * b  # Element-wise multiply for bipolar = XOR equivalent

    def _bundle(self, vectors: list) -> torch.Tensor:
        """Bundle multiple hypervectors (majority vote)."""
        if HDC_AVAILABLE:
            stacked = torch.stack(vectors)
            return torchhd.multiset(stacked)
        else:
            stacked = torch.stack(vectors)
            return torch.sign(stacked.sum(dim=0))

    def _similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two hypervectors."""
        if HDC_AVAILABLE:
            return torchhd.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        else:
            return torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()

    def _quantize_value(self, value: float, basis: torch.Tensor, levels: int = 10) -> torch.Tensor:
        """Encode a continuous value by rotating a basis vector."""
        # Thermometer encoding: number of permutations proportional to value
        level = int(value * levels)
        hv = basis.clone()
        for _ in range(level):
            hv = torch.roll(hv, 1)  # Permutation = rotation
        return hv

    def encode_state(self, metrics: Dict) -> torch.Tensor:
        """Encode system state + op info as a single hypervector."""
        components = []

        # Continuous metrics (0-1 normalized)
        components.append(self._bind(
            self.basis['gpu_temp'],
            self._quantize_value(metrics.get('gpu_temp', 0) / 100.0, self.basis['gpu_temp'])
        ))
        components.append(self._bind(
            self.basis['gpu_util'],
            self._quantize_value(metrics.get('gpu_util', 0) / 100.0, self.basis['gpu_util'])
        ))
        components.append(self._bind(
            self.basis['cpu_load'],
            self._quantize_value(metrics.get('cpu_load', 0) / 100.0, self.basis['cpu_load'])
        ))
        components.append(self._bind(
            self.basis['npu_available'],
            self._quantize_value(1.0 if metrics.get('npu_available', False) else 0.0, self.basis['npu_available'])
        ))
        components.append(self._bind(
            self.basis['op_size'],
            self._quantize_value(min(metrics.get('op_size', 0) / 1e6, 1.0), self.basis['op_size'])
        ))

        # Op type (categorical -- use the matching basis vector)
        op_type = metrics.get('op_type', 'other')
        op_key = f'op_type_{op_type}' if f'op_type_{op_type}' in self.basis else 'op_type_other'
        components.append(self.basis[op_key])

        # Bundle all components into one state vector
        return self._bundle(components)

    def _batch_similarity(self, state_hv: torch.Tensor) -> Tuple[str, float]:
        """Vectorized codebook lookup via single matmul.

        Phase 2 optimization: pre-normalize codebook, then cosine similarity
        is a matrix-vector product. ~200us vs 5221us for the per-entry loop.
        """
        if not hasattr(self, '_codebook_matrix') or self._codebook_dirty:
            hvs = torch.stack([e['hv'].float() for e in self.codebook])
            norms = torch.norm(hvs, dim=1, keepdim=True).clamp(min=1e-8)
            self._codebook_matrix = hvs / norms
            self._codebook_devices = [e['device'] for e in self.codebook]
            self._codebook_dirty = False

        q = state_hv.float().unsqueeze(0)
        q = q / torch.norm(q).clamp(min=1e-8)

        # Single matmul: [1, D] @ [D, N] = [1, N] similarities
        similarities = (q @ self._codebook_matrix.T).squeeze(0)

        best_idx = torch.argmax(similarities).item()
        best_sim = similarities[best_idx].item()

        if best_sim < 0.1:
            return None, best_sim
        return self._codebook_devices[best_idx], best_sim

    def dispatch(self, metrics: Dict) -> Tuple[str, float]:
        """
        Find the best routing decision for the current state.
        Returns (device, confidence).

        This is the moment the magnets snap.
        """
        if not self.codebook:
            return self._cold_start(metrics)

        state_hv = self.encode_state(metrics)
        device, similarity = self._batch_similarity(state_hv)

        if device is None:
            return self._cold_start(metrics)

        return device, similarity

    def _cold_start(self, metrics: Dict) -> Tuple[str, float]:
        """Heuristic routing when codebook is empty or no match found."""
        op_type = metrics.get('op_type', 'other')
        gpu_temp = metrics.get('gpu_temp', 50)

        if op_type in ('matmul', 'attention', 'conv') and gpu_temp < 80:
            return 'gpu', 0.0
        elif op_type in ('embed', 'normalize', 'softmax'):
            return 'npu', 0.0
        else:
            return 'cpu', 0.0

    def record(self, metrics: Dict, device: str, latency_ms: float):
        """
        Record a routing decision and its outcome.
        Good outcomes (low latency) get stored as codebook entries.
        This is how the codebook learns -- through accumulation, not backprop.
        """
        state_hv = self.encode_state(metrics)

        entry = {
            'hv': state_hv,
            'device': device,
            'latency_ms': latency_ms,
            'metrics': {k: v for k, v in metrics.items() if k != 'hv'},
        }

        # Only store good outcomes (below median latency for this device)
        device_latencies = [e['latency_ms'] for e in self.codebook if e['device'] == device]
        if not device_latencies or latency_ms <= np.median(device_latencies) * 1.2:
            self.codebook.append(entry)
            self._codebook_dirty = True

            # Keep codebook bounded (oldest entries evicted)
            max_entries = 1000
            if len(self.codebook) > max_entries:
                self.codebook = self.codebook[-max_entries:]

        # Periodic save
        if len(self.codebook) % 10 == 0:
            self._save()

    def _save(self):
        """Save codebook to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        data = {
            'dimensions': self.d,
            'entries': len(self.codebook),
            'basis_keys': list(self.basis.keys()),
        }
        # Save basis vectors and codebook entries as torch tensors
        torch.save({
            'basis': self.basis,
            'device_hvs': self.device_hvs,
            'codebook': [(e['hv'], e['device'], e['latency_ms'], e['metrics'])
                        for e in self.codebook],
        }, self.save_path.replace('.json', '.pt'))

        # Save human-readable summary
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load codebook from disk."""
        pt_path = self.save_path.replace('.json', '.pt')
        if os.path.exists(pt_path):
            try:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
                self.basis = data['basis']
                self.device_hvs = data['device_hvs']
                self.codebook = [
                    {'hv': hv, 'device': dev, 'latency_ms': lat, 'metrics': met}
                    for hv, dev, lat, met in data['codebook']
                ]
                print(f"[HDC] Loaded codebook: {len(self.codebook)} entries")
            except Exception as e:
                print(f"[HDC] Failed to load codebook: {e}")

    def show(self) -> str:
        """Display codebook statistics."""
        if not self.codebook:
            return "HDC Codebook: Empty (cold start mode)"

        device_counts = {}
        device_latencies = {}
        for entry in self.codebook:
            dev = entry['device']
            device_counts[dev] = device_counts.get(dev, 0) + 1
            device_latencies.setdefault(dev, []).append(entry['latency_ms'])

        lines = [
            f"HDC Codebook: {len(self.codebook)} entries, {self.d} dimensions",
            "",
            f"{'Device':<8} {'Entries':<10} {'Avg Latency':<15} {'Share':<10}",
            f"{'------':<8} {'-------':<10} {'-----------':<15} {'-----':<10}",
        ]
        for dev in DEVICES:
            count = device_counts.get(dev, 0)
            if count > 0:
                avg_lat = np.mean(device_latencies[dev])
                share = count / len(self.codebook) * 100
                lines.append(f"{dev:<8} {count:<10} {avg_lat:<15.2f}ms {share:<10.1f}%")

        return "\n".join(lines)

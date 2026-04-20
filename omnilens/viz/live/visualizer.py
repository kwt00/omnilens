"""LiveVisualizer — interactive browser-based model analysis.

Provides three analysis panels:
  1. Activation heatmap — see which layers/tokens are active
  2. Attention explorer — inspect attention patterns per head/layer
  3. Feature dashboard — SAE feature activations per token (when SAE attached)

Usage:
    from omnilens import TappedModel
    from omnilens.viz.live import LiveVisualizer

    model = TappedModel.from_pretrained("gpt2")
    viz = LiveVisualizer(model)
    viz.start()  # opens browser at localhost:8765

    # Optional: attach an SAE for feature analysis
    viz.attach_sae(sae, hook_point="layers.6.residual.block_out")
"""

from __future__ import annotations

import threading
import webbrowser
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel
    from omnilens.methods.sae import SAE


class LiveVisualizer:
    """Interactive browser-based model analysis tool."""

    def __init__(self, model: TappedModel) -> None:
        if not hasattr(model, "model") or not isinstance(model.model, nn.Module):
            raise TypeError(
                f"Expected a TappedModel, got {type(model).__name__}"
            )
        self._model = model
        self._server_thread: threading.Thread | None = None
        self._port: int = 8765

    def attach_sae(self, sae: SAE, hook_point: str) -> None:
        """Attach an SAE for feature analysis in the dashboard."""
        from omnilens.viz.live.api.routes_analysis import set_sae
        set_sae(sae, hook_point)

    def start(self, port: int = 8765, open_browser: bool = True) -> None:
        """Start the visualization server in a background thread."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "LiveVisualizer requires fastapi and uvicorn. "
                "Install with: pip install omnilens[live]"
            )

        from omnilens.viz.live.api.routes_analysis import set_model
        set_model(self._model)

        from omnilens.viz.live.config import settings
        settings.port = port
        self._port = port

        def run_server():
            from omnilens.viz.live.server import app
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        url = f"http://localhost:{port}"
        print(f"OmniLens Live at {url}")

        if open_browser:
            webbrowser.open(url)

    def stop(self) -> None:
        """Stop the visualization server."""
        pass  # daemon thread dies with the process

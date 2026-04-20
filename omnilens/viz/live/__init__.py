"""Live architecture visualization for any PyTorch model.

Usage with TappedModel:
    from omnilens import TappedModel
    from omnilens.viz.live import LiveVisualizer

    model = TappedModel.from_pretrained("meta-llama/Llama-3.1-8B")
    viz = LiveVisualizer(model)
    viz.start()  # opens browser

Usage with raw nn.Module:
    import torch.nn as nn
    from omnilens.viz.live import LiveVisualizer

    model = nn.TransformerEncoder(...)
    viz = LiveVisualizer(model, sample_input=torch.randn(1, 64))
    viz.start()
"""

from omnilens.viz.live.visualizer import LiveVisualizer

__all__ = ["LiveVisualizer"]

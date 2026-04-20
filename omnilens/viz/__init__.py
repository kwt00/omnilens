def __getattr__(name):
    if name == "LiveVisualizer":
        from omnilens.viz.live import LiveVisualizer
        return LiveVisualizer
    raise AttributeError(f"module 'omnilens.viz' has no attribute {name!r}")

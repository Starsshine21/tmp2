def evaluate_dexjoco_openpi(*args, **kwargs):
    """Load the simulator evaluator only when the CLI invokes it."""
    from .eval_dexjoco_openpi import main

    return main(*args, **kwargs)

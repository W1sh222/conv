"""Layout-only QA fixture; these masks are not experimental results."""
import importlib.util
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent / "mask_layout_qa"
root.mkdir(exist_ok=True)
(root / "experiment.json").write_text(json.dumps({"arguments": {"layer": 14, "head": 8}}))
np.savez(root / "block_map.npz", selected_mask=np.tri(256, dtype=bool))
spec = importlib.util.spec_from_file_location("mask_plot", Path(__file__).parents[1] / "experiments2/replot_mask_overview.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for location in ("upper-right", "bottom"):
    module.draw(root, root / f"layout_only_{location}.png", 120, 1.4, location)
print(root)

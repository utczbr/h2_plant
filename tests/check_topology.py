"""Check if topology has duplicate connections"""

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.config.plant_config import ConnectionConfig

# Create topology
topology = [
    ConnectionConfig('electrolyzer', 'h2_out', 'lp_storage', 'h2_in', 'hydrogen'),
    ConnectionConfig('lp_storage', 'h2_out', 'compressor', 'h2_in', 'hydrogen'),
    ConnectionConfig('compressor', 'h2_out', 'hp_storage', 'h2_in', 'hydrogen')
]

print(f"Number of connections: {len(topology)}")
for i, conn in enumerate(topology):
    print(f"{i}: {conn.source_id} → {conn.target_id}")

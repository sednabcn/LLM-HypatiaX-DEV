#!/bin/bash
set -euo pipefail

pip install img2pdf --break-system-packages -q 2>&1 | tail -3

INPUT="hypatiax_algorithm1_routing_cascade_v2.png"
OUTPUT="hypatiax_algorithm1_routing_cascade_v2.pdf"

python3 -c "
import img2pdf

input_path = '${INPUT}'
output_path = '${OUTPUT}'

with open(output_path, 'wb') as f:
    f.write(img2pdf.convert(input_path))

print(f'Wrote {output_path}')
"

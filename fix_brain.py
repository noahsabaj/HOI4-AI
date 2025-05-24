# fix_brain.py - Quick fix to expose features
import os
import sys

# Read original brain
brain_path = "src/legacy/ai/hoi4_brain.py"
with open(brain_path, 'r') as f:
    content = f.read()

# Add features attribute before return
old_line = "        return {"
new_lines = """        # Store features for hybrid model
        self.features = x

        return {"""

# Replace
content = content.replace(old_line, new_lines)

# Save
with open(brain_path, 'w') as f:
    f.write(content)

print("✅ Fixed brain to expose features!")
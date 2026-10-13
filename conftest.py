"""Make the repository root importable so `import groundingdino...` works
when running `pytest` from anywhere in the tree."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

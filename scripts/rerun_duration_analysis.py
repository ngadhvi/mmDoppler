import nbformat

nb_path = "notebooks/02_analyze.ipynb"
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

code = """# Re-run Duration Statistics with Time-Gap Segmentation
print("Computing activity duration statistics (Time-Gap Aware)...")

# Re-initialize analyzer to pick up new code
from helper.analysis_utils import ClassDistributionAnalyzer
dist_analyzer = ClassDistributionAnalyzer(micro_df)

duration_stats = dist_analyzer.compute_duration_stats(fps=2.0)
print(duration_stats)
"""

if nb.cells:
    # Update the last cell again
    nb.cells[-1].source = code
    nb.cells[-1].outputs = []
    nb.cells[-1].execution_count = None

with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print("Updated 02_analyze.ipynb to re-run the duration analysis.")

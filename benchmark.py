# 29/12/2022 - major refactor (copied from Neural Filtering)
from ransac_comparison import run_comparison

def benchmark(ransac_func, matches, query_images_names, Ks, no_iterations, val_idx=None):
    images_data = run_comparison(ransac_func, matches, query_images_names, Ks, no_iterations, val_idx=val_idx)
    return images_data
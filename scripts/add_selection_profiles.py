import argparse
import h5py
import numpy as np
import os


def main(args):
    verbose = args.verbose
    fp_max = args.fp_max
    fn_max = args.fn_max
    fps = np.arange(0, fp_max + 1)
    fns = np.arange(0, fn_max + 1)
    n_repeats = args.n_repeats
    seed = args.seed

    # File path to folder or file
    file_path = args.file_path
    if os.path.isfile(file_path):
        paths = [file_path]
    elif os.path.isdir(file_path):
        paths = [os.path.join(file_path, f)
                 for f in sorted(os.listdir(file_path))]
    else:
        raise ValueError("File path must be valid file or directory.")

    # Iterate over experiment files
    for path in paths:
        if verbose:
            print(f"Fitting to file {path}.")
        # Read in true parameters
        with h5py.File(path, 'r') as params:
            a_true = params['a_true'][:] != 0
            b_true = params['b_true'][:] != 0
        # Get selection / non-selection profiles
        nz_idx_a = np.argwhere(a_true).ravel()
        z_idx_a = np.setdiff1d(np.arange(a_true.size), nz_idx_a)
        nz_idx_b = np.argwhere(b_true).ravel()
        z_idx_b = np.setdiff1d(np.arange(b_true.size), nz_idx_b)
        rng = np.random.default_rng(seed)

        # For each file, iterate over FP and FN
        for fp in fps:
            for fn in fns:
                # Iterate over certain number of repeats
                for repeat in range(n_repeats):
                    # Coupling parameters
                    a_sel = np.copy(a_true)
                    fp_idx_a = rng.choice(z_idx_a, size=fp, replace=False)
                    fn_idx_a = rng.choice(nz_idx_a, size=fn, replace=False)
                    a_sel[fp_idx_a] = True
                    a_sel[fn_idx_a] = False
                    # Tuning parameters
                    b_sel = np.copy(b_true)
                    fp_idx_b = rng.choice(z_idx_b, size=fp, replace=False)
                    fn_idx_b = rng.choice(nz_idx_b, size=fn, replace=False)
                    b_sel[fp_idx_b] = True
                    b_sel[fn_idx_b] = False
                    # Write to file
                    with h5py.File(path, 'a') as params:
                        params[f'a_fp={fp}_fn={fn}_{repeat}'] = a_sel
                        params[f'b_fp={fp}_fn={fn}_{repeat}'] = b_sel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--fp_max', type=int, default=4)
    parser.add_argument('--fn_max', type=int, default=4)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2332)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)

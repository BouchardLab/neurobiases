import argparse
import glob
import numpy as np
import os


def main(args):
    exp_folder = args.exp_folder
    tag = args.tag
    script_path = args.script_path
    job_folder = args.job_folder

    # Experiment configuration
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    n_models = args.n_models
    n_datasets = args.n_datasets
    # Model hyperparameters
    n_coupling_locs = args.n_coupling_locs
    n_tuning_locs = args.n_tuning_locs
    coupling_locs = np.linspace(args.coupling_loc_min,
                                args.coupling_loc_max,
                                n_coupling_locs)
    tuning_locs = np.linspace(args.tuning_loc_min,
                              args.tuning_loc_max,
                              n_tuning_locs)
    # Create hyperparameters for CV fitting
    n_coupling_lambdas = args.n_coupling_lambdas
    coupling_lambdas = np.logspace(args.coupling_lambda_lower,
                                   args.coupling_lambda_upper,
                                   num=n_coupling_lambdas)
    n_tuning_lambdas = args.n_tuning_lambdas
    tuning_lambdas = np.logspace(args.tuning_lambda_lower,
                                 args.tuning_lambda_upper,
                                 num=n_tuning_lambdas)
    Ks = np.arange(args.max_K) + 1

    # Process coupling random states
    if args.coupling_rng == -1:
        coupling_rng = np.random.default_rng()
    else:
        coupling_rng = np.random.default_rng(args.coupling_rng)
    coupling_rngs = coupling_rng.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process tuning random states
    if args.tuning_rng == -1:
        tuning_rng = np.random.default_rng()
    else:
        tuning_rng = np.random.default_rng(args.tuning_rng)
    tuning_rngs = tuning_rng.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process dataset random states
    if args.dataset_rng == -1:
        dataset_rng = np.random.default_rng()
    else:
        dataset_rng = np.random.default_rng(args.dataset_rng)
    dataset_rngs = dataset_rng.integers(
        low=0,
        high=2**32 - 1,
        size=n_datasets)
    # Process fitter random state
    if args.fitter_rng == -1:
        fitter_rng = np.random.randint(low=0, high=2**32 - 1)
    else:
        fitter_rng = args.fitter_rng

    n_jobs = n_coupling_locs * n_tuning_locs * n_models * n_datasets
    n_batch_scripts = args.n_batch_scripts
    n_jobs_per_batch = int(np.ceil(n_jobs / n_batch_scripts))

    # Header of batch script
    batch_header = lambda tag: (
        "#!/bin/bash\n"
        f"#SBATCH -N {n_nodes}\n"
        "#SBATCH -C haswell\n"
        f"#SBATCH -q {qos}\n"
        "#SBATCH -J nb\n"
        f"#SBATCH --output=/global/homes/s/sachdeva/out/neurobiases/{tag}.o\n"
        f"#SBATCH --error=/global/homes/s/sachdeva/error/neurobiases/{tag}.o\n"
        "#SBATCH --mail-user=pratik.sachdeva@berkeley.edu\n"
        "#SBATCH --mail-type=ALL\n"
        f"#SBATCH -t {time}\n"
        f"#SBATCH --image=docker:pssachdeva/neuro2:latest\n\n"
    )

    counter = 0
    for ii, coupling_loc in enumerate(coupling_locs):
        for jj, tuning_loc in enumerate(tuning_locs):
            for kk, (coupling_rng, tuning_rng) in enumerate(zip(coupling_rngs, tuning_rngs)):
                for ll, dataset_rng in enumerate(dataset_rngs):
                    counter += 1
                    save_path = f"{save_folder}/{tag}_{kk}_{ii}_{jj}_{ll}.h5"
                    # Load the model configuration
                    with h5py.File(save_path, 'w') as params:
                        # Triangular model hyperparameters
                        params.attrs['N'] = N
                        params.attrs['M'] = M
                        params.attrs['K'] = K
                        params.attrs['D'] = D
                        params.attrs['corr_cluster'] = args.corr_cluster
                        params.attrs['corr_back'] = args.corr_back
                        params.attrs['coupling_distribution'] = args.coupling_distribution
                        params.attrs['coupling_sparsity'] = args.coupling_sparsity
                        params.attrs['coupling_loc'] = coupling_loc
                        params.attrs['coupling_scale'] = args.coupling_scale
                        params.attrs['coupling_rng'] = coupling_rng
                        params.attrs['tuning_distribution'] = args.tuning_distribution
                        params.attrs['tuning_sparsity'] = args.tuning_sparsity
                        params.attrs['tuning_loc'] = tuning_loc
                        params.attrs['tuning_scale'] = args.tuning_scale
                        params.attrs['tuning_rng'] = tuning_rng
                        # Random seeds
                        params.attrs['coupling_rng'] = coupling_rng
                        params.attrs['tuning_rng'] = tuning_rng
                        params.attrs['dataset_rng'] = dataset_rng
                        params.attrs['fitter_rng'] = fitter_rng
                        # Training hyperparameters
                        params['Ks'] = Ks
                        params['coupling_lambdas'] = coupling_lambdas
                        params['tuning_lambdas'] = tuning_lambdas
                        # Training settings
                        params.attrs['cv'] = args.cv
                        params.attrs['fine_sweep_frac'] = args.fine_sweep_frac
                        params.attrs['solver'] = args.solver
                        params.attrs['initialization'] = args.initialization
                        params.attrs['max_iter'] = args.max_iter
                        params.attrs['tol'] = args.tol

    # Find all configuration files matching the tag
    files = glob.glob(f"{exp_folder}/{tag}*.h5")
    n_files = len(files)

    # Handle NERSC issues
    n_nodes = args.n_nodes
    n_tasks = args.n_tasks
    time = args.time
    qos = args.qos
    n_confs = int(np.ceil(n_files / n_nodes))

    # Model arguments
    model_fit = args.model_fit
    warn = args.warn

    # Split files into groups assigned to each .conf file
    splits = (n_nodes * np.arange(n_confs))[1:]
    file_groups = np.array_split(files, splits)
    n_groups = len(file_groups)
    # Iterate over each conf file
    for group_idx, file_group in enumerate(file_groups):
        conf_path = os.path.join(job_folder, f"{tag}_{group_idx}.conf")
        n_files = len(file_group)
        batch_script += "echo '==============================================================='\n"
        batch_script += f"echo 'Job {group_idx + 1}/{n_groups}'\n"
        batch_script += "echo '==============================================================='\n"
        batch_script += f"srun -n {n_files} -c 64 --multi-prog {conf_path}\n"
        # Open conf file for current group
        with open(conf_path, 'w') as conf:
            # Iterate over files in group
            for idx, file_path in enumerate(file_group):
                command = (
                    f"srun -n {n_tasks} -c $OMP_NUM_THREADS shifter python "
                    f"{args.script_path} --file_path="
                    f"HOME=/global/homes/s/sachdeva/ mpirun -np {n_tasks} "
                    f"-launcher fork python -u {script_path} "
                    f"--file_path={file_path} "
                    f"--model_fit={model_fit} "
                )
                if warn:
                    command += " --warn"
                command += "\n"
                conf.write(command)

    # Write batch script
    batch_path = f"{job_folder}/{tag}.sh"
    with open(batch_path, 'w') as batch:
        batch.write(batch_script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--tag', type=str)
    # Fixed model hyperparameters
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--n_datasets', type=int, default=30)
    # Variable model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-2)
    parser.add_argument('--coupling_loc_max', type=float, default=2)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-2)
    parser.add_argument('--tuning_loc_max', type=float, default=2)
    # CV fitting hyperparameters
    parser.add_argument('--n_coupling_lambdas', type=int, default=25)
    parser.add_argument('--coupling_lambda_lower', type=float, default=-7)
    parser.add_argument('--coupling_lambda_upper', type=float, default=-2)
    parser.add_argument('--n_tuning_lambdas', type=int, default=25)
    parser.add_argument('--tuning_lambda_lower', type=float, default=-7)
    parser.add_argument('--tuning_lambda_upper', type=float, default=-2)
    parser.add_argument('--fine_sweep_frac', type=float, default=0.1)
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    # Model parameters
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', type=float, default=0.5)
    parser.add_argument('--coupling_scale', type=float, default=0.25)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', type=float, default=0.5)
    parser.add_argument('--tuning_scale', type=float, default=0.25)
    parser.add_argument('--corr_cluster', type=float, default=0.25)
    parser.add_argument('--corr_back', type=float, default=0.10)
    # Fitter arguments
    parser.add_argument('--solver', default='ow_lbfgs')
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    # Random seeds
    parser.add_argument('--coupling_rng', type=int, default=-1)
    parser.add_argument('--tuning_rng', type=int, default=-1)
    parser.add_argument('--dataset_rng', type=int, default=-1)
    parser.add_argument('--fitter_rng', type=int, default=-1)
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--exp_folder', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--script_path', type=str)
    parser.add_argument('--job_folder', type=str)
    # NERSC options
    parser.add_argument('--n_nodes', type=int, default=128)
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--model_fit', default='em')
    parser.add_argument('--time', default='00:30:00')
    parser.add_argument('--qos', default='debug')
    # Other options
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)

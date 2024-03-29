import argparse
import h5py
import neuropacks as packs
import numpy as np


def main(args):
    # Tag for the experiment
    tag = args.tag
    # Folder containing the model fits
    fit_folder = args.fit_folder
    # Folder containing the batch scripts
    batch_folder = args.batch_folder
    # Folder containing the outputs
    output_folder = args.output_folder
    # Folder containing the outputs
    script_path = args.script_path
    # Path to neural data
    data_path = args.data_path
    dataset = args.dataset

    # NERSC parameters
    n_nodes = args.n_nodes
    n_tasks = args.n_tasks
    n_batch_scripts = args.n_batch_scripts
    time = args.time
    qos = args.qos

    # Experiment configuration
    model_fit = args.model_fit
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

    # Handle dataset
    if dataset == 'pvc11':
        pack = packs.PVC11(data_path=data_path)
        # Get design matrix and stimuli
        X = pack.get_design_matrix(form='cosine2')
        Y = pack.get_response_matrix(transform='square_root')
    elif dataset == 'ecog':
        raise NotImplementedError()

    # Get number of jobs and batch scripts
    n_jobs = Y.shape[1]
    n_jobs_per_batch = int(np.ceil(n_jobs / n_batch_scripts))

    # Header of batch script
    def batch_header(out):
        return (
            "#!/bin/bash\n"
            f"#SBATCH -N {n_nodes}\n"
            "#SBATCH -C haswell\n"
            f"#SBATCH -q {qos}\n"
            "#SBATCH -J nb\n"
            f"#SBATCH --output={output_folder}/{out}_out.o\n"
            f"#SBATCH --error={output_folder}/{out}_error.o\n"
            "#SBATCH --mail-user=pratik.sachdeva@berkeley.edu\n"
            "#SBATCH --mail-type=ALL\n"
            f"#SBATCH -t {time}\n"
            f"#SBATCH --image=docker:pssachdeva/neuro:latest\n\n"
            "export OMP_NUM_THREADS=1\n\n")

    job_counter = 0
    batch_counter = 0

    for job in range(n_jobs):
        # Save path for current experiment
        save_path = f"{fit_folder}/{tag}_{dataset}_{job}.h5"
        # Load the model configuration
        with h5py.File(save_path, 'w') as params:
            params['X'] = X
            params['Y'] = np.delete(Y, job, axis=1)
            params['y'] = Y[:, job][..., np.newaxis]
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
        # Create SBATCH command
        command = (
            f"srun -n {n_tasks} -c $OMP_NUM_THREADS shifter python "
            f"{script_path} --file_path={save_path} "
            f"--model_fit={model_fit} &\n"
        )

        # If we have reset the job counter, create a new batch script
        if job_counter == 0:
            batch_script = batch_header(f"{tag}_{batch_counter}")
            batch_path = f"{batch_folder}/{tag}_{batch_counter}.sh"
            batch_counter += 1

        batch_script += command
        job_counter += 1
        if job_counter % args.n_simultaneous:
            command += "wait\n"
        # If we're at the last counter, write file
        if job_counter == n_jobs_per_batch:
            with open(batch_path, 'w') as batch:
                batch.write(batch_script)
            # Reset job counter
            job_counter = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    # Required arguments
    parser.add_argument('--tag', type=str)
    parser.add_argument('--fit_folder', type=str)
    parser.add_argument('--batch_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--script_path', type=str)
    parser.add_argument('--data_path', type=str)
    # NERSC arguments
    parser.add_argument('--n_nodes', type=int, default=64)
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--n_batch_scripts', type=int, default=100)
    parser.add_argument('--n_simultaneous', type=int, default=50)
    parser.add_argument('--time', type=str, default='00:30:00')
    parser.add_argument('--qos', type=str, default='debug')
    # Fixed model hyperparameters
    parser.add_argument('--model_fit', type=str, default='em')
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
    # Fitter arguments
    parser.add_argument('--solver', default='ow_lbfgs')
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    # Random seeds
    parser.add_argument('--fitter_rng', type=int, default=-1)
    args = parser.parse_args()

    main(args)

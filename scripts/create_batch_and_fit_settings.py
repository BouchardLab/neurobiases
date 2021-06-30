import argparse
import h5py
import numpy as np
import os


def main(args):
    # Folder containing the experiment parameters
    params_folder = args.params_folder
    # Experiment tag
    tag = args.tag
    if tag is None:
        tag = os.path.basename(params_folder).split('_')[0]
    # Name for group to store experiment results
    group_name = args.group
    # Folder containing the batch scripts
    batch_folder = args.batch_folder
    # Folder containing the outputs
    output_folder = args.output_folder
    # Path to the script
    script_path = args.script_path

    # Get all the experiments
    paths = [os.path.join(params_folder, f)
             for f in sorted(os.listdir(params_folder))]
    n_jobs = len(paths)

    # NERSC parameters
    run_nersc = not args.run_local
    n_nodes = args.n_nodes
    if run_nersc:
        n_tasks = args.n_tasks
        n_nodes_per_job = int(n_tasks / 32)
        n_batch_scripts = args.n_batch_scripts
        n_simultaneous = args.n_simultaneous
        time = args.time
        qos = args.qos
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
    else:
        n_simultaneous = n_jobs
        n_jobs_per_batch = n_jobs

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

    batch_counter = 0
    for job, path in enumerate(paths):
        # Load the model configuration
        with h5py.File(path, 'a') as exp:
            group = exp.create_group(group_name)
            # Training hyperparameters
            group['Ks'] = Ks
            group['coupling_lambdas'] = coupling_lambdas
            group['tuning_lambdas'] = tuning_lambdas
            # Training settings
            group.attrs['model'] = args.model
            group.attrs['criterion'] = args.criterion
            group.attrs['cv'] = args.cv
            group.attrs['fine_sweep_frac'] = args.fine_sweep_frac
            group.attrs['n_coupling_lambdas_fine'] = args.n_coupling_lambdas_fine
            group.attrs['n_tuning_lambdas_fine'] = args.n_tuning_lambdas_fine
            group.attrs['solver'] = args.solver
            group.attrs['initialization'] = args.initialization
            group.attrs['max_iter'] = args.max_iter
            group.attrs['tol'] = args.tol
            group.attrs['Psi_transform'] = args.Psi_transform

        # If we have reset the job counter, create a new batch script
        job_idx = job % n_jobs_per_batch

        if job_idx == 0:
            if run_nersc:
                batch_script = batch_header(f"{tag}_{batch_counter}")
                batch_path = f"{batch_folder}/{tag}_{batch_counter}.sh"
                batch_counter += 1
            else:
                batch_script = ""
                batch_path = f"{batch_folder}/{tag}.sh"

        # Create run command
        command = f"python -u {script_path} --file_path={path} --group={group_name}"
        if run_nersc:
            command = (
                f"srun -N {n_nodes_per_job} -n {n_tasks} "
                "-c $OMP_NUM_THREADS shifter {command} &")

            if (job + 1) % n_simultaneous == 0:
                command += "wait"
        else:
            command = f"mpirun -n {n_nodes} " + command
        command += "\n"
        batch_script += command

        # If we're at the last counter, write file
        if job_idx + 1 == n_jobs_per_batch:
            if run_nersc and ((job + 1) % n_simultaneous != 0):
                batch_script += "wait"

            with open(batch_path, 'w') as batch:
                batch.write(batch_script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    # Required arguments
    parser.add_argument('--params_folder', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--batch_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--script_path', type=str)
    # NERSC arguments
    parser.add_argument('--run_local', action='store_true')
    parser.add_argument('--n_nodes', type=int, default=64)
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--n_batch_scripts', type=int, default=100)
    parser.add_argument('--n_simultaneous', type=int, default=50)
    parser.add_argument('--time', type=str, default='00:30:00')
    parser.add_argument('--qos', type=str, default='debug')
    # Solver hyperparameters
    parser.add_argument('--model', type=str, default='tm')
    parser.add_argument('--n_coupling_lambdas', type=int, default=25)
    parser.add_argument('--n_coupling_lambdas_fine', type=int, default=10)
    parser.add_argument('--coupling_lambda_lower', type=float, default=-7)
    parser.add_argument('--coupling_lambda_upper', type=float, default=-2)
    parser.add_argument('--n_tuning_lambdas', type=int, default=25)
    parser.add_argument('--n_tuning_lambdas_fine', type=int, default=10)
    parser.add_argument('--tuning_lambda_lower', type=float, default=-7)
    parser.add_argument('--tuning_lambda_upper', type=float, default=-2)
    parser.add_argument('--fine_sweep_frac', type=float, default=0.1)
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--criterion', type=str, default='bic')
    # Fitter arguments
    parser.add_argument('--solver', default='ow_lbfgs')
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--Psi_transform', type=str, default='softplus')
    args = parser.parse_args()

    main(args)

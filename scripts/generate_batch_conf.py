import argparse
import glob
import numpy as np
import os


def main(args):
    exp_folder = args.exp_folder
    tag = args.tag
    script_path = args.script_path
    job_folder = args.job_folder

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

    # Header of batch script
    batch_script = (
        "#!/bin/bash\n"
        f"#SBATCH -N {n_nodes}\n"
        "#SBATCH -C haswell\n"
        f"#SBATCH -q {qos}\n"
        "#SBATCH -J nb\n"
        f"#SBATCH --output=/global/homes/s/sachdeva/out/neurobiases/{job_folder}.o\n"
        f"#SBATCH --output=/global/homes/s/sachdeva/error/neurobiases/{job_folder}.o\n"
        "#SBATCH --mail-user=pratik.sachdeva@berkeley.edu\n"
        "#SBATCH --mail-type=ALL\n"
        f"#SBATCH -t {time}\n"
        f"#SBATCH --image=docker:pssachdeva/neuro2:latest\n\n"
    )

    # Split files into groups assigned to each .conf file
    splits = (n_nodes * np.arange(n_confs))[1:]
    file_groups = np.array_split(files, splits)
    # Iterate over each conf file
    for group_idx, file_group in enumerate(file_groups):
        conf_path = os.path.join(job_folder, f"{tag}_{group_idx}.conf")
        batch_script += f"srun -n {n_nodes} -c 64 --multi-prog {conf_path}\n"
        # Open conf file for current group
        with open(conf_path, 'w') as conf:
            # Iterate over files in group
            for idx, file_path in enumerate(file_group):
                command = (
                    f"{idx} shifter -E -e "
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

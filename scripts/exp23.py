import h5py
import os
import subprocess


model = 'tm'
base_path = '/storage/fits/neurobiases/exp23'
paths = [os.path.join(base_path, f)
         for f in sorted(os.listdir(base_path))]

for path in paths:
    print(f'Fitting path {path}.')
    with h5py.File(path, 'r') as params:
        a_groups = [group for group in list(params) if group.startswith('a_fp')]
        b_groups = [group for group in list(params) if group.startswith('b_fp')]
    for a_group in a_groups:
        for b_group in b_groups:
            print(f'---> Group {a_group}, {b_group}.')
            subprocess.run([
                'mpirun',
                '-n',
                '4',
                'python',
                '-u',
                '/home/psachdeva/projects/neurobiases/scripts/estimation_from_file.py',
                f'--file_path={path}',
                f'--model={model}',
                f'--a_mask_group={a_group}',
                f'--b_mask_group={b_group}',
                f'--results_group={model}_{a_group}_{b_group}'
            ])

# global imports
import csv
import json
import math
import numpy as np
import os
import subprocess
import scipy.stats.qmc as qmc
import time

def getConfig():
    # The dsParameterBounds names need to match the NX parameter names in the part file as well as the journal file!
    config_file = os.path.join(os.getcwd(), 'config.json')
    try:
        # Open and read the JSON file
        with open(config_file, 'r') as file:
            config = json.load(file)
    except Exception as e:
        print("Configuration file not found. Make sure it is in the same folder as the main.py script")
    return config

def generateSamples(config, nSamples):
    """
    Generates Latin Hypercube samples for given parameters and saves each sample as a separate JSON file.
    
    Parameters:
    - parameters: dict, keys are parameter names and values are [min, max] lists
    - numSamples: int, number of samples to generate
    """
    # create one dictionary containing all parameters that have ranges
    parameters = config["dsParameterBounds"]

    # Generate input files for latin hypercube samples
    sampler = qmc.LatinHypercube(d=len(parameters)) 
    samples = sampler.random(nSamples)
    scaled_samples = qmc.scale(samples, [v[0] for v in parameters.values()], [v[1] for v in parameters.values()])
    sampleLst = []
    for i in range(nSamples):
        param_dict = {k:float(v) for k,v in zip(parameters.keys(), scaled_samples[i])}
        sampleLst.append(param_dict)
    return sampleLst

def build_command(sample, journalFile, freeCadExecpath):
    cmd = [freeCadExecpath, journalFile, json.dumps(sample),] 
    return cmd

def calculate_objective(sample, freeCAD_journal, freeCADExecPath):
    # create necessary folders
    # try processing the current sample geometry
    try:
        # run freecad journal, that updates the geometry with current parameters
        # and solves the fe simulation and returns the deformation energy
        cmd = build_command(sample, freeCAD_journal , freeCADExecPath)
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)# creationflags=subprocess.CREATE_NO_WINDOW)
        assert res.returncode == 0 , "freeCAD script routine failed"
        # retrieve the absorbed energy from the logged string
        # make sure the last thing you log in the full_journal_fc.py script is the 
        absorbed_energy = float(res.stdout.split('\n')[-3])
        max_stress = float(res.stdout.split('\n')[-2])
        if "Access violation" in res.stderr:
            raise Exception("Access violation error in freeCAD script, likely due to invalid geometry or failed meshing.")
    # catch exceptions and log failed samples. DB entry at index (sampleId) is invalid
    except Exception as e:
        print(f"Sample processing failed due to error: {e}")
        time.sleep(1)
        return None, None

    # return binary files for optional backward conversion to geometries
    return absorbed_energy, max_stress

def initialize_objective_log(csv_path, parameter_names):
    fieldnames = ["eval_id"] + parameter_names + ["objective_value", "valid", "best_objective_so_far"]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

def log_objective_evaluation(csv_path, eval_id, sample, parameter_names, objective_value, valid, best_objective_so_far):
    fieldnames = ["eval_id"] + parameter_names + ["objective_value", "valid", "best_objective_so_far"]
    row = {
        "eval_id": eval_id,
        "objective_value": objective_value if valid else "",
        "valid": valid,
        "best_objective_so_far": best_objective_so_far if best_objective_so_far is not None else "",
    }
    for name in parameter_names:
        row[name] = sample[name]

    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row)
    
def main():
    """
    In this example, a latin hypercube sampling is performed on the parameter bounds
    and the geometries are evaluated for there objective of energy absorbtion. 
    TODO:
    Your task is to implement an optimizer to find the optimal geometry maximizing the
    energy absorption capacity. This can include better sampling strategies, choosing 
    a suitable optimizer and handling parameter combinations that yield invalid geometries.

    Do not forget to fulfill the maximum stress constraint as well as the geometry 
    constraints listed in section 2 of the project description.

    """
    freeCADExecPath = r'C:/Program Files/FreeCAD 1.0/bin/FreeCADCmd.exe' # default on windows
    # mac should be something like: 
    # freeCADExecPath = r'/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd'
    # linux should be something like, but depends heavily on the installation method:
    # freeCADExecPath = '/usr/local/freecad/lib/freeCADCmd'
    
    # freecad scripts for geometry manipulation and fem simulation
    '''
    if the script does not work initially, copy the full_journal_fc.py file
    to the bin folder in the FreeCAD 1.0 directory,  i.e for windows default
    'C:/Program Files/FreeCAD 1.0/bin/full_journal_fc.py' and change the 
    freeCAD_journal variable below to this path.
    After running like that once, you should be able to copy the file back and
    run normally.
    '''
    freeCAD_journal = os.path.join(os.getcwd(), "full_journal_fc.py")

    # generates latin hypercube samples of the parameters
    nSamples = 15
    config = getConfig()
    # samples is a list of dictionary containing parameter:value pairs
    samples = generateSamples(config, nSamples)

    parameter_names = list(config["dsParameterBounds"].keys())
    objective_log_path = os.path.join(os.getcwd(), "objective_evaluations.csv")
    initialize_objective_log(objective_log_path, parameter_names)

    results = []
    best_objective_so_far = None
    for eval_id, sample in enumerate(samples, start=1):
        absorbed_energy, max_stress = calculate_objective(sample, freeCAD_journal, freeCADExecPath)
        valid = absorbed_energy is not None
        if valid:
            if best_objective_so_far is None:
                best_objective_so_far = absorbed_energy
            else:
                best_objective_so_far = max(best_objective_so_far, absorbed_energy)

        log_objective_evaluation(
            objective_log_path,
            eval_id,
            sample,
            parameter_names,
            absorbed_energy,
            valid,
            best_objective_so_far,
        )

        if absorbed_energy is not None:
            print(f"Sample {sample} -> Absorbed Energy: {absorbed_energy}, Max Stress: {max_stress}")
            sample['energy_objective'] = absorbed_energy
            sample['stress_constraint'] = max_stress
            results.append(sample)

    # each entry in the results list is a dictionary with cadparams:value pairs
    # as well as the objective:value pair corresponding to this geometry
    #print(results)
    # print objective value of one result
    idx = 1
    #print('objective value (total plastic deformation energy):', results[idx]['energy_objective'])
    #print('maximum stress value:', results[idx]['stress_constraint'])



if __name__ == "__main__":
    main()

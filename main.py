# global imports
import csv
import json
import math
import numpy as np
import os
import subprocess
import scipy.stats.qmc as qmc
import time

PLATE_SIZE = 60.0

MIN_EDGE_CLEARANCE = 2.0
MIN_CUTOUT_CLEARANCE = 1.0

CIRCLE_1_RADIUS = 9.0
CIRCLE_2_RADIUS = 6.0
CAPSULE_RADIUS = 7.5
CAPSULE_CENTERLINE_LENGTH = 25.0
STRESS_LIMIT = 275.0

OBJECTIVE_LOG_PARAMETERS = ["x1", "y1", "x2", "y2", "x3", "y3", "angle"]
OBJECTIVE_LOG_FIELDNAMES = [
    "eval_id",
    *OBJECTIVE_LOG_PARAMETERS,
    "objective_value",
    "max_stress",
    "geometry_valid",
    "stress_valid",
    "fem_valid",
    "failure_reason",
    "best_feasible_objective_so_far",
]

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

def _as_float(sample, name):
    try:
        return float(sample[name])
    except KeyError:
        raise ValueError(f"missing required geometry parameter '{name}'")
    except (TypeError, ValueError):
        raise ValueError(f"geometry parameter '{name}' is not numeric")

def _point_distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

def _point_to_segment_distance(point, segment_start, segment_end):
    px, py = point
    ax, ay = segment_start
    bx, by = segment_end
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return _point_distance(point, segment_start)

    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return _point_distance(point, closest)

def _circle_edge_clearance(center, radius):
    x, y = center
    return min(x - radius, y - radius, PLATE_SIZE - x - radius, PLATE_SIZE - y - radius)

def _capsule_centerline(sample):
    x3 = _as_float(sample, "x3")
    y3 = _as_float(sample, "y3")
    angle_degrees = _as_float(sample, "angle")

    # FreeCAD's sketch uses a clockwise-positive spreadsheet angle for this slot.
    theta = math.radians(-angle_degrees)
    half_length = CAPSULE_CENTERLINE_LENGTH / 2.0
    dx = half_length * math.cos(theta)
    dy = half_length * math.sin(theta)
    return (x3 - dx, y3 - dy), (x3 + dx, y3 + dy)

def _capsule_edge_clearance(segment_start, segment_end, radius):
    min_x = min(segment_start[0], segment_end[0])
    max_x = max(segment_start[0], segment_end[0])
    min_y = min(segment_start[1], segment_end[1])
    max_y = max(segment_start[1], segment_end[1])
    return min(min_x - radius, min_y - radius, PLATE_SIZE - max_x - radius, PLATE_SIZE - max_y - radius)

def _circle_cutout_clearance(center_a, radius_a, center_b, radius_b):
    return _point_distance(center_a, center_b) - radius_a - radius_b

def _circle_capsule_cutout_clearance(circle_center, circle_radius, segment_start, segment_end, capsule_radius):
    return _point_to_segment_distance(circle_center, segment_start, segment_end) - circle_radius - capsule_radius

def geometry_constraint_report(sample):
    try:
        circle_1_center = (_as_float(sample, "x1"), _as_float(sample, "y1"))
        circle_2_center = (_as_float(sample, "x2"), _as_float(sample, "y2"))
        capsule_start, capsule_end = _capsule_centerline(sample)
    except ValueError as exc:
        return False, str(exc)

    edge_clearances = [
        ("circle 1", _circle_edge_clearance(circle_1_center, CIRCLE_1_RADIUS)),
        ("circle 2", _circle_edge_clearance(circle_2_center, CIRCLE_2_RADIUS)),
        ("capsule", _capsule_edge_clearance(capsule_start, capsule_end, CAPSULE_RADIUS)),
    ]
    for cutout_name, clearance in edge_clearances:
        if clearance < MIN_EDGE_CLEARANCE:
            return False, (
                f"{cutout_name} edge clearance is {clearance:.3f} mm; "
                f"minimum is {MIN_EDGE_CLEARANCE:.3f} mm"
            )

    cutout_clearances = [
        (
            "circle 1 to circle 2",
            _circle_cutout_clearance(circle_1_center, CIRCLE_1_RADIUS, circle_2_center, CIRCLE_2_RADIUS),
        ),
        (
            "circle 1 to capsule",
            _circle_capsule_cutout_clearance(
                circle_1_center,
                CIRCLE_1_RADIUS,
                capsule_start,
                capsule_end,
                CAPSULE_RADIUS,
            ),
        ),
        (
            "circle 2 to capsule",
            _circle_capsule_cutout_clearance(
                circle_2_center,
                CIRCLE_2_RADIUS,
                capsule_start,
                capsule_end,
                CAPSULE_RADIUS,
            ),
        ),
    ]
    for cutout_pair, clearance in cutout_clearances:
        if clearance < MIN_CUTOUT_CLEARANCE:
            return False, (
                f"{cutout_pair} clearance is {clearance:.3f} mm; "
                f"minimum is {MIN_CUTOUT_CLEARANCE:.3f} mm"
            )

    return True, ""

def generate_geometry_valid_samples(config, nSamples, max_attempts=10000):
    valid_samples = []
    attempts = 0
    batch_size = max(50, nSamples * 5)

    while len(valid_samples) < nSamples and attempts < max_attempts:
        candidates = generateSamples(config, min(batch_size, max_attempts - attempts))
        for candidate in candidates:
            attempts += 1
            valid, _ = geometry_constraint_report(candidate)
            if valid:
                valid_samples.append(candidate)
                if len(valid_samples) == nSamples:
                    break

    if len(valid_samples) < nSamples:
        raise RuntimeError(
            f"Could only generate {len(valid_samples)} geometry-valid samples "
            f"after {attempts} attempts. Tighten config.json bounds or raise max_attempts."
        )

    print(f"Generated {len(valid_samples)} geometry-valid samples after {attempts} candidates.")
    return valid_samples

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
        res = subprocess.run(cmd, capture_output=True, text=True)# creationflags=subprocess.CREATE_NO_WINDOW)
        if "Access violation" in res.stderr:
            raise Exception("Access violation error in freeCAD script, likely due to invalid geometry or failed meshing.")
        if res.returncode != 0:
            details = (res.stderr or res.stdout).strip()
            if details:
                details = details.splitlines()[-1]
                raise Exception(f"FreeCAD exited with code {res.returncode}: {details}")
            raise Exception(f"FreeCAD exited with code {res.returncode}")
        # retrieve the absorbed energy from the logged string
        # make sure the last thing you log in the full_journal_fc.py script is the 
        output_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        absorbed_energy = float(output_lines[-2])
        max_stress = float(output_lines[-1])
    # catch exceptions and log failed samples. DB entry at index (sampleId) is invalid
    except Exception as e:
        failure_reason = str(e)
        print(f"Sample processing failed due to error: {failure_reason}")
        time.sleep(1)
        return None, None, failure_reason

    # return binary files for optional backward conversion to geometries
    return absorbed_energy, max_stress, ""

def initialize_objective_log(csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OBJECTIVE_LOG_FIELDNAMES)
        writer.writeheader()

def log_objective_evaluation(
    csv_path,
    eval_id,
    sample,
    objective_value,
    max_stress,
    geometry_valid,
    stress_valid,
    fem_valid,
    failure_reason,
    best_feasible_objective_so_far,
):
    row = {
        "eval_id": eval_id,
        "objective_value": objective_value if fem_valid else "",
        "max_stress": max_stress if fem_valid else "",
        "geometry_valid": geometry_valid,
        "stress_valid": stress_valid,
        "fem_valid": fem_valid,
        "failure_reason": failure_reason,
        "best_feasible_objective_so_far": (
            best_feasible_objective_so_far
            if best_feasible_objective_so_far is not None
            else ""
        ),
    }
    for name in OBJECTIVE_LOG_PARAMETERS:
        row[name] = sample.get(name, "")

    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OBJECTIVE_LOG_FIELDNAMES)
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
    max_candidate_attempts = 10000
    batch_size = max(50, nSamples * 5)
    config = getConfig()

    objective_log_path = os.path.join(os.getcwd(), "objective_evaluations.csv")
    initialize_objective_log(objective_log_path)

    results = []
    best_feasible_objective_so_far = None
    eval_id = 0
    candidate_attempts = 0
    feasible_samples_found = 0

    while feasible_samples_found < nSamples and candidate_attempts < max_candidate_attempts:
        samples = generateSamples(
            config,
            min(batch_size, max_candidate_attempts - candidate_attempts),
        )

        for sample in samples:
            eval_id += 1
            candidate_attempts += 1

            geometry_valid, geometry_failure_reason = geometry_constraint_report(sample)
            absorbed_energy = None
            max_stress = None
            fem_valid = False
            stress_valid = False
            failure_reason = ""

            if geometry_valid:
                absorbed_energy, max_stress, failure_reason = calculate_objective(sample, freeCAD_journal, freeCADExecPath)
                fem_valid = absorbed_energy is not None and max_stress is not None
                if fem_valid:
                    stress_valid = max_stress <= STRESS_LIMIT
                    if not stress_valid:
                        failure_reason = (
                            f"max_stress {max_stress:.6g} exceeds limit {STRESS_LIMIT:.6g}"
                        )
            else:
                print(f"Sample {sample} skipped: {geometry_failure_reason}")
                failure_reason = geometry_failure_reason

            feasible = geometry_valid and fem_valid and stress_valid
            if feasible:
                if best_feasible_objective_so_far is None:
                    best_feasible_objective_so_far = absorbed_energy
                else:
                    best_feasible_objective_so_far = max(best_feasible_objective_so_far, absorbed_energy)

            log_objective_evaluation(
                objective_log_path,
                eval_id,
                sample,
                absorbed_energy,
                max_stress,
                geometry_valid,
                stress_valid,
                fem_valid,
                failure_reason,
                best_feasible_objective_so_far,
            )

            if fem_valid:
                print(f"Sample {sample} -> Absorbed Energy: {absorbed_energy}, Max Stress: {max_stress}")
                sample['energy_objective'] = absorbed_energy
                sample['stress_constraint'] = max_stress
                if stress_valid:
                    feasible_samples_found += 1
                    results.append(sample)

            if feasible_samples_found >= nSamples:
                break

    if feasible_samples_found < nSamples:
        raise RuntimeError(
            f"Only found {feasible_samples_found} feasible candidates "
            f"after {candidate_attempts} logged candidates. Tighten config.json bounds "
            "or raise max_candidate_attempts."
        )

    # each entry in the results list is a dictionary with cadparams:value pairs
    # as well as the objective:value pair corresponding to this geometry
    #print(results)
    # print objective value of one result
    idx = 1
    #print('objective value (total plastic deformation energy):', results[idx]['energy_objective'])
    #print('maximum stress value:', results[idx]['stress_constraint'])



if __name__ == "__main__":
    main()

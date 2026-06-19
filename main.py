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
QUADRATIC_FEATURE_COUNT = (
    1
    + len(OBJECTIVE_LOG_PARAMETERS)
    + len(OBJECTIVE_LOG_PARAMETERS)
    + (len(OBJECTIVE_LOG_PARAMETERS) * (len(OBJECTIVE_LOG_PARAMETERS) - 1)) // 2
)
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

def get_config_int(config, name, default, minimum=1):
    value = config.get(name, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Configuration value '{name}' must be an integer")
    if value < minimum:
        raise ValueError(f"Configuration value '{name}' must be >= {minimum}")
    return value

def get_config_float(config, name, default, minimum=None):
    value = config.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Configuration value '{name}' must be numeric")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration value '{name}' must be >= {minimum}")
    return value

def parameter_bounds(config):
    bounds = config["dsParameterBounds"]
    missing_parameters = [
        name for name in OBJECTIVE_LOG_PARAMETERS if name not in bounds
    ]
    if missing_parameters:
        raise ValueError(
            "Configuration dsParameterBounds is missing: "
            + ", ".join(missing_parameters)
        )
    return bounds

def design_cache_key(sample, round_decimals):
    return tuple(round(_as_float(sample, name), round_decimals) for name in OBJECTIVE_LOG_PARAMETERS)

def normalize_sample(sample, bounds):
    normalized = []
    for name in OBJECTIVE_LOG_PARAMETERS:
        lower, upper = bounds[name]
        lower = float(lower)
        upper = float(upper)
        if upper <= lower:
            raise ValueError(f"Invalid bounds for '{name}': upper must exceed lower")
        normalized.append((2.0 * (_as_float(sample, name) - lower) / (upper - lower)) - 1.0)
    return np.array(normalized, dtype=float)

def quadratic_features_from_normalized(normalized_values):
    features = [1.0]
    features.extend(normalized_values)
    features.extend(value * value for value in normalized_values)
    for i in range(len(normalized_values)):
        for j in range(i + 1, len(normalized_values)):
            features.append(normalized_values[i] * normalized_values[j])
    return np.array(features, dtype=float)

def build_quadratic_feature_matrix(samples, bounds):
    feature_rows = []
    for sample in samples:
        normalized = normalize_sample(sample, bounds)
        feature_rows.append(quadratic_features_from_normalized(normalized))
    return np.vstack(feature_rows)

def fit_response_surface(feasible_results, bounds, ridge_lambda):
    if not feasible_results:
        return None

    x_matrix = build_quadratic_feature_matrix(feasible_results, bounds)
    y_vector = np.array(
        [float(result["energy_objective"]) for result in feasible_results],
        dtype=float,
    )

    regularization = np.eye(x_matrix.shape[1], dtype=float) * ridge_lambda
    regularization[0, 0] = 0.0
    normal_matrix = x_matrix.T @ x_matrix + regularization
    normal_rhs = x_matrix.T @ y_vector
    try:
        coefficients = np.linalg.solve(normal_matrix, normal_rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal_matrix, normal_rhs, rcond=None)[0]

    return {
        "bounds": bounds,
        "coefficients": coefficients,
        "training_count": len(feasible_results),
    }

def predict_response_surface(model, samples):
    x_matrix = build_quadratic_feature_matrix(samples, model["bounds"])
    return x_matrix @ model["coefficients"]

def generate_unique_geometry_valid_samples(
    config,
    nSamples,
    excluded_keys,
    round_decimals,
    max_attempts=100000,
):
    valid_samples = []
    seen_keys = set()
    attempts = 0
    batch_size = max(50, nSamples * 5)

    while len(valid_samples) < nSamples and attempts < max_attempts:
        candidates = generateSamples(config, min(batch_size, max_attempts - attempts))
        for candidate in candidates:
            attempts += 1
            candidate_key = design_cache_key(candidate, round_decimals)
            if candidate_key in excluded_keys or candidate_key in seen_keys:
                continue

            geometry_valid, _ = geometry_constraint_report(candidate)
            if not geometry_valid:
                continue

            seen_keys.add(candidate_key)
            valid_samples.append(candidate)
            if len(valid_samples) == nSamples:
                break

    if len(valid_samples) < nSamples:
        raise RuntimeError(
            f"Could only generate {len(valid_samples)} unique geometry-valid samples "
            f"after {attempts} candidates. Tighten config.json bounds, reduce candidate "
            "pool sizes, or raise max_candidate_attempts."
        )

    print(f"Generated {len(valid_samples)} unique geometry-valid samples after {attempts} candidates.")
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

def evaluate_and_log_candidate(
    csv_path,
    eval_id,
    sample,
    freeCAD_journal,
    freeCADExecPath,
    best_feasible_objective_so_far,
    best_feasible_design_so_far,
):
    geometry_valid, geometry_failure_reason = geometry_constraint_report(sample)
    absorbed_energy = None
    max_stress = None
    fem_valid = False
    stress_valid = False
    failure_reason = ""

    if geometry_valid:
        absorbed_energy, max_stress, failure_reason = calculate_objective(
            sample,
            freeCAD_journal,
            freeCADExecPath,
        )
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
    if feasible and (
        best_feasible_objective_so_far is None
        or absorbed_energy > best_feasible_objective_so_far
    ):
        best_feasible_objective_so_far = absorbed_energy
        best_feasible_design_so_far = {
            name: _as_float(sample, name) for name in OBJECTIVE_LOG_PARAMETERS
        }

    log_objective_evaluation(
        csv_path,
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

    evaluation = {
        **{name: _as_float(sample, name) for name in OBJECTIVE_LOG_PARAMETERS},
        "objective_value": absorbed_energy,
        "max_stress": max_stress,
        "geometry_valid": geometry_valid,
        "stress_valid": stress_valid,
        "fem_valid": fem_valid,
        "failure_reason": failure_reason,
        "feasible": feasible,
        "energy_objective": absorbed_energy,
        "stress_constraint": max_stress,
        "fem_attempted": geometry_valid,
    }
    return evaluation, best_feasible_objective_so_far, best_feasible_design_so_far
    
def main():
    """
    Run a budgeted response-surface optimization loop.

    Real FreeCAD FEM evaluations remain the source of truth. The quadratic
    response surface is used only to rank new geometry-valid candidates cheaply.
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

    config = getConfig()
    bounds = parameter_bounds(config)

    initial_doe_size = get_config_int(config, "initial_doe_size", 50)
    max_fem_evaluations = get_config_int(config, "max_fem_evaluations", 75)
    surrogate_candidate_pool_size = get_config_int(config, "surrogate_candidate_pool_size", 2000)
    surrogate_batch_size = get_config_int(config, "surrogate_batch_size", 5)
    max_candidate_attempts = get_config_int(config, "max_candidate_attempts", 100000)
    min_surrogate_training_points = get_config_int(
        config,
        "min_surrogate_training_points",
        max(8, len(OBJECTIVE_LOG_PARAMETERS) + 1),
    )
    cache_round_decimals = get_config_int(config, "cache_round_decimals", 6, minimum=0)
    rsm_ridge_lambda = get_config_float(config, "rsm_ridge_lambda", 1.0e-8, minimum=0.0)

    objective_log_path = os.path.join(os.getcwd(), "objective_evaluations.csv")
    initialize_objective_log(objective_log_path)

    feasible_results = []
    evaluated_design_keys = set()
    best_feasible_objective_so_far = None
    best_feasible_design_so_far = None
    eval_id = 0
    fem_evaluations = 0

    def evaluate_candidates(samples, phase_name):
        nonlocal eval_id
        nonlocal fem_evaluations
        nonlocal best_feasible_objective_so_far
        nonlocal best_feasible_design_so_far

        for sample in samples:
            if fem_evaluations >= max_fem_evaluations:
                break

            sample_key = design_cache_key(sample, cache_round_decimals)
            if sample_key in evaluated_design_keys:
                continue
            evaluated_design_keys.add(sample_key)

            eval_id += 1
            print(
                f"{phase_name} candidate {eval_id}: "
                f"FEM evaluation {fem_evaluations + 1}/{max_fem_evaluations}"
            )
            evaluation, best_feasible_objective_so_far, best_feasible_design_so_far = (
                evaluate_and_log_candidate(
                    objective_log_path,
                    eval_id,
                    sample,
                    freeCAD_journal,
                    freeCADExecPath,
                    best_feasible_objective_so_far,
                    best_feasible_design_so_far,
                )
            )

            if evaluation["fem_attempted"]:
                fem_evaluations += 1

            if evaluation["feasible"]:
                feasible_results.append(
                    {
                        **{name: evaluation[name] for name in OBJECTIVE_LOG_PARAMETERS},
                        "energy_objective": evaluation["energy_objective"],
                        "stress_constraint": evaluation["stress_constraint"],
                    }
                )

    initial_doe_count = min(initial_doe_size, max_fem_evaluations)
    if initial_doe_count > 0:
        print(f"Generating initial DOE with {initial_doe_count} geometry-valid samples.")
        initial_doe_samples = generate_unique_geometry_valid_samples(
            config,
            initial_doe_count,
            evaluated_design_keys,
            cache_round_decimals,
            max_attempts=max_candidate_attempts,
        )
        evaluate_candidates(initial_doe_samples, "DOE")

    while fem_evaluations < max_fem_evaluations:
        remaining_budget = max_fem_evaluations - fem_evaluations
        batch_count = min(surrogate_batch_size, remaining_budget)

        if len(feasible_results) >= min_surrogate_training_points:
            model = fit_response_surface(feasible_results, bounds, rsm_ridge_lambda)
            candidate_pool_count = max(surrogate_candidate_pool_size, batch_count)
            candidate_pool = generate_unique_geometry_valid_samples(
                config,
                candidate_pool_count,
                evaluated_design_keys,
                cache_round_decimals,
                max_attempts=max_candidate_attempts,
            )
            predictions = predict_response_surface(model, candidate_pool)
            ranked_indices = np.argsort(predictions)[::-1]
            selected_samples = [
                candidate_pool[index] for index in ranked_indices[:batch_count]
            ]
            print(
                "Fitted quadratic response surface with "
                f"{model['training_count']} feasible FEM points "
                f"({QUADRATIC_FEATURE_COUNT} features); evaluating top "
                f"{len(selected_samples)} predicted candidates."
            )
        else:
            print(
                f"Only {len(feasible_results)} feasible FEM points are available; "
                "using geometry-valid LHS exploration before fitting the response surface."
            )
            selected_samples = generate_unique_geometry_valid_samples(
                config,
                batch_count,
                evaluated_design_keys,
                cache_round_decimals,
                max_attempts=max_candidate_attempts,
            )

        fem_evaluations_before_batch = fem_evaluations
        evaluate_candidates(selected_samples, "RSM")
        if fem_evaluations == fem_evaluations_before_batch:
            raise RuntimeError(
                "No new FEM evaluations were completed in the latest iteration. "
                "Check candidate generation and cache settings."
            )

    if best_feasible_design_so_far is None:
        print(
            "Optimization finished without a feasible FEM result satisfying "
            f"max_stress <= {STRESS_LIMIT:.6g} MPa."
        )
    else:
        print("Best feasible design found:")
        for name in OBJECTIVE_LOG_PARAMETERS:
            print(f"  {name}: {best_feasible_design_so_far[name]}")
        print(f"Best feasible objective: {best_feasible_objective_so_far}")



if __name__ == "__main__":
    main()

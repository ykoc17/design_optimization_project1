# global imports
import argparse
import csv
import json
import math
import numpy as np
import os
import subprocess
import scipy.stats.qmc as qmc
import sys
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
    "srsm_iteration",
    "predicted_objective",
    "srsm_subspace_fraction",
    "best_feasible_objective_so_far",
]

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

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

def generateSamples(config, nSamples, bounds=None):
    """
    Generates Latin Hypercube samples for given parameters and saves each sample as a separate JSON file.
    
    Parameters:
    - parameters: dict, keys are parameter names and values are [min, max] lists
    - numSamples: int, number of samples to generate
    """
    # create one dictionary containing all parameters that have ranges
    parameters = bounds if bounds is not None else config["dsParameterBounds"]

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

def generate_geometry_valid_samples(config, nSamples, max_attempts=10000, bounds=None):
    valid_samples = []
    attempts = 0
    batch_size = max(50, nSamples * 5)

    while len(valid_samples) < nSamples and attempts < max_attempts:
        candidates = generateSamples(
            config,
            min(batch_size, max_attempts - attempts),
            bounds=bounds,
        )
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
    normalized_bounds = {}
    for name in OBJECTIVE_LOG_PARAMETERS:
        lower, upper = bounds[name]
        lower = float(lower)
        upper = float(upper)
        if upper <= lower:
            raise ValueError(f"Invalid bounds for '{name}': upper must exceed lower")
        normalized_bounds[name] = (lower, upper)
    return normalized_bounds

def validate_fraction_config(value, name):
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"Configuration value '{name}' must be > 0 and <= 1")
    return value

def validate_shrink_factor(value, name):
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"Configuration value '{name}' must be > 0 and < 1")
    return value

def make_centered_subspace_bounds(center, full_bounds, width_fraction):
    active_bounds = {}
    for name in OBJECTIVE_LOG_PARAMETERS:
        full_lower, full_upper = full_bounds[name]
        full_width = full_upper - full_lower
        active_width = min(full_width, full_width * width_fraction)
        center_value = min(max(_as_float(center, name), full_lower), full_upper)

        active_lower = center_value - active_width / 2.0
        active_upper = center_value + active_width / 2.0
        if active_lower < full_lower:
            active_upper += full_lower - active_lower
            active_lower = full_lower
        if active_upper > full_upper:
            active_lower -= active_upper - full_upper
            active_upper = full_upper

        active_bounds[name] = (
            max(full_lower, active_lower),
            min(full_upper, active_upper),
        )
    return active_bounds

def sample_within_bounds(sample, bounds):
    for name in OBJECTIVE_LOG_PARAMETERS:
        lower, upper = bounds[name]
        value = _as_float(sample, name)
        if value < lower or value > upper:
            return False
    return True

def select_response_surface_training_results(
    feasible_results,
    active_bounds,
    full_bounds,
    min_training_points,
):
    local_results = [
        result for result in feasible_results
        if sample_within_bounds(result, active_bounds)
    ]
    if len(local_results) >= min_training_points:
        return local_results, active_bounds, "active sub-design space"
    return feasible_results, full_bounds, "full design history"

def has_meaningful_improvement(previous_best, current_best, relative_tolerance):
    if current_best is None:
        return False
    if previous_best is None:
        return True
    threshold = max(1.0, abs(previous_best)) * relative_tolerance
    return current_best > previous_best + threshold

def format_bounds_for_log(bounds):
    return ", ".join(
        f"{name}=[{bounds[name][0]:.3g}, {bounds[name][1]:.3g}]"
        for name in OBJECTIVE_LOG_PARAMETERS
    )

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
    bounds=None,
    minimum_required=None,
):
    valid_samples = []
    seen_keys = set()
    attempts = 0
    batch_size = max(50, nSamples * 5)

    while len(valid_samples) < nSamples and attempts < max_attempts:
        candidates = generateSamples(
            config,
            min(batch_size, max_attempts - attempts),
            bounds=bounds,
        )
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

    required_count = nSamples if minimum_required is None else minimum_required
    if len(valid_samples) < nSamples and len(valid_samples) >= required_count:
        region_name = "active sub-design space" if bounds is not None else "full design space"
        print(
            f"Generated {len(valid_samples)} of {nSamples} requested "
            f"unique geometry-valid samples from {region_name} after {attempts} "
            "candidates; continuing with the partial pool."
        )
        return valid_samples

    if len(valid_samples) < nSamples:
        region_name = "active sub-design space" if bounds is not None else "full design space"
        raise RuntimeError(
            f"Could only generate {len(valid_samples)} unique geometry-valid samples "
            f"from {region_name} after {attempts} candidates. Tighten config.json "
            "bounds, reduce candidate pool sizes, widen the SRSM subspace, or raise "
            "max_candidate_attempts."
        )

    region_name = "active sub-design space" if bounds is not None else "full design space"
    print(
        f"Generated {len(valid_samples)} unique geometry-valid samples from "
        f"{region_name} after {attempts} candidates."
    )
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
    srsm_iteration=None,
    predicted_objective=None,
    srsm_subspace_fraction=None,
):
    row = {
        "eval_id": eval_id,
        "objective_value": objective_value if fem_valid else "",
        "max_stress": max_stress if fem_valid else "",
        "geometry_valid": geometry_valid,
        "stress_valid": stress_valid,
        "fem_valid": fem_valid,
        "failure_reason": failure_reason,
        "srsm_iteration": srsm_iteration if srsm_iteration is not None else "",
        "predicted_objective": (
            predicted_objective if predicted_objective is not None else ""
        ),
        "srsm_subspace_fraction": (
            srsm_subspace_fraction if srsm_subspace_fraction is not None else ""
        ),
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

def csv_float_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def csv_int_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None

def read_prediction_history(csv_path):
    best_prediction_by_iteration = {}
    fem_objective_for_best_prediction_by_iteration = {}
    subspace_fraction_by_iteration = {}
    predicted_row_count = 0
    row_count = 0

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_count += 1
            srsm_iteration = csv_int_or_none(row.get("srsm_iteration"))
            subspace_fraction = csv_float_or_none(row.get("srsm_subspace_fraction"))
            if srsm_iteration is not None and subspace_fraction is not None:
                subspace_fraction_by_iteration.setdefault(
                    srsm_iteration,
                    subspace_fraction,
                )

            predicted_objective = csv_float_or_none(row.get("predicted_objective"))
            if srsm_iteration is None or predicted_objective is None:
                continue

            predicted_row_count += 1
            previous_best = best_prediction_by_iteration.get(srsm_iteration)
            if previous_best is None or predicted_objective > previous_best:
                best_prediction_by_iteration[srsm_iteration] = predicted_objective
                fem_objective_for_best_prediction_by_iteration[srsm_iteration] = (
                    csv_float_or_none(row.get("objective_value"))
                )

    iterations = sorted(best_prediction_by_iteration)
    subspace_iterations = sorted(subspace_fraction_by_iteration)
    fem_iterations = [
        iteration
        for iteration in iterations
        if fem_objective_for_best_prediction_by_iteration.get(iteration) is not None
    ]

    return {
        "iterations": iterations,
        "predictions": [
            best_prediction_by_iteration[iteration] for iteration in iterations
        ],
        "plot_iterations": iterations,
        "fem_iterations": fem_iterations,
        "fem_objectives": [
            fem_objective_for_best_prediction_by_iteration[iteration]
            for iteration in fem_iterations
        ],
        "subspace_iterations": subspace_iterations,
        "subspace_plot_iterations": subspace_iterations,
        "subspace_fractions": [
            subspace_fraction_by_iteration[iteration]
            for iteration in subspace_iterations
        ],
        "predicted_row_count": predicted_row_count,
        "row_count": row_count,
    }

def save_convergence_plot(csv_path, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print(
            "matplotlib is not available; skipping convergence plot. "
            f"The objective CSV remains complete at {csv_path}."
        )
        return False

    try:
        history = read_prediction_history(csv_path)
    except FileNotFoundError:
        print(f"Cannot create convergence plot because CSV was not found: {csv_path}")
        return False
    except OSError as exc:
        print(f"Cannot create convergence plot from {csv_path}: {exc}")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if history["iterations"]:
        ax.plot(
            history["plot_iterations"],
            history["predictions"],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="Surrogate predicted energy",
        )
        if history["fem_iterations"]:
            ax.plot(
                history["fem_iterations"],
                history["fem_objectives"],
                marker="^",
                linewidth=2.0,
                color="#7db8e8",
                label="FEM evaluated energy",
            )
    else:
        message = "No SRSM predictions found"
        if history["row_count"] == 0:
            message = "No evaluation rows found"
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax_subspace = ax.twinx()
    if history["subspace_iterations"]:
        ax_subspace.plot(
            history["subspace_plot_iterations"],
            history["subspace_fractions"],
            marker="s",
            linewidth=2.0,
            color="tab:red",
            label="ROI side-length fraction",
        )

    ax.set_xlabel("SRSM iteration step")
    ax.set_ylabel("Absorbed energy")
    ax_subspace.set_ylabel("ROI side-length fraction")
    ax.set_title("Successive RSM convergence")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    all_plot_iterations = (
        history["plot_iterations"]
        + history["fem_iterations"]
        + history["subspace_plot_iterations"]
    )
    if all_plot_iterations:
        min_iteration = min(all_plot_iterations)
        max_iteration = max(all_plot_iterations)
        if min_iteration == max_iteration:
            ax.set_xlim(left=min_iteration - 0.5, right=max_iteration + 0.5)
        else:
            ax.set_xlim(left=min_iteration, right=max_iteration)
        ax.set_xticks(range(min_iteration, max_iteration + 1))

    line_handles = []
    line_labels = []
    for axis in (ax, ax_subspace):
        handles, labels = axis.get_legend_handles_labels()
        line_handles.extend(handles)
        line_labels.extend(labels)
    if line_handles:
        ax.legend(line_handles, line_labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if history["iterations"]:
        print(
            f"Saved convergence plot to {output_path} "
            f"({len(history['iterations'])} SRSM iterations from "
            f"{history['predicted_row_count']} predicted candidates)."
        )
    else:
        print(
            f"Saved convergence plot to {output_path}; "
            "no SRSM predictions were found in the CSV."
        )
    return True

def evaluate_and_log_candidate(
    csv_path,
    eval_id,
    sample,
    freeCAD_journal,
    freeCADExecPath,
    best_feasible_objective_so_far,
    best_feasible_design_so_far,
    srsm_iteration=None,
    predicted_objective=None,
    srsm_subspace_fraction=None,
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
        srsm_iteration=srsm_iteration,
        predicted_objective=predicted_objective,
        srsm_subspace_fraction=srsm_subspace_fraction,
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
        "srsm_iteration": srsm_iteration,
        "predicted_objective": predicted_objective,
        "srsm_subspace_fraction": srsm_subspace_fraction,
    }
    return evaluation, best_feasible_objective_so_far, best_feasible_design_so_far

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the FreeCAD-backed successive RSM optimizer or plot convergence from an existing CSV."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only read the objective CSV and save the SRSM convergence plot.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=os.path.join(os.getcwd(), "objective_evaluations.csv"),
        help="Path to the objective evaluations CSV.",
    )
    parser.add_argument(
        "--plot-output",
        default=os.path.join(os.getcwd(), "plots", "convergence.png"),
        help="Path for the convergence plot image.",
    )
    parser.add_argument(
        "--output-log",
        default=os.path.join(os.getcwd(), "output.txt"),
        help="Path for the command-line output log.",
    )
    return parser.parse_args(argv)

def run_with_output_log(args, run_function):
    output_dir = os.path.dirname(args.output_log)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_log, "w", encoding="utf-8") as output_log:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, output_log)
        sys.stderr = TeeStream(original_stderr, output_log)
        try:
            print(f"Writing command-line output to {args.output_log}")
            return run_function()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

def run_optimizer(args):
    """
    Run a budgeted successive response surface optimization loop.

    Each SRSM iteration builds a response surface for the current region of
    interest, samples only inside that active sub-design space, evaluates the
    most promising predictions with FreeCAD, then moves and shrinks the next
    active region around the best feasible FEM result.
    """
    if args.plot_only:
        save_convergence_plot(args.csv_path, args.plot_output)
        return

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
    srsm_initial_subspace_fraction = validate_fraction_config(
        get_config_float(config, "srsm_initial_subspace_fraction", 0.6, minimum=0.0),
        "srsm_initial_subspace_fraction",
    )
    srsm_min_subspace_fraction = validate_fraction_config(
        get_config_float(config, "srsm_min_subspace_fraction", 0.1, minimum=0.0),
        "srsm_min_subspace_fraction",
    )
    if srsm_min_subspace_fraction > srsm_initial_subspace_fraction:
        raise ValueError(
            "Configuration value 'srsm_min_subspace_fraction' must be <= "
            "'srsm_initial_subspace_fraction'"
        )
    srsm_subspace_shrink_factor = validate_shrink_factor(
        get_config_float(config, "srsm_subspace_shrink_factor", 0.7, minimum=0.0),
        "srsm_subspace_shrink_factor",
    )
    srsm_failure_shrink_factor = validate_shrink_factor(
        get_config_float(config, "srsm_failure_shrink_factor", 0.5, minimum=0.0),
        "srsm_failure_shrink_factor",
    )
    srsm_improvement_tolerance = get_config_float(
        config,
        "srsm_improvement_tolerance",
        1.0e-6,
        minimum=0.0,
    )
    if min_surrogate_training_points > max_fem_evaluations:
        print(
            "Warning: min_surrogate_training_points is greater than "
            "max_fem_evaluations, so the adaptive SRSM phase cannot start "
            "with the current config."
        )

    objective_log_path = args.csv_path
    initialize_objective_log(objective_log_path)

    feasible_results = []
    evaluated_design_keys = set()
    best_feasible_objective_so_far = None
    best_feasible_design_so_far = None
    eval_id = 0
    fem_evaluations = 0
    srsm_iteration = 0
    active_subspace_fraction = srsm_initial_subspace_fraction

    def evaluate_candidates(
        samples,
        phase_name,
        candidate_srsm_iteration=None,
        predicted_values=None,
        candidate_srsm_subspace_fraction=None,
    ):
        nonlocal eval_id
        nonlocal fem_evaluations
        nonlocal best_feasible_objective_so_far
        nonlocal best_feasible_design_so_far

        if predicted_values is None:
            predicted_values = [None] * len(samples)
        if len(predicted_values) != len(samples):
            raise ValueError("predicted_values must match the number of samples")

        evaluations = []
        for sample, predicted_objective in zip(samples, predicted_values):
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
                    srsm_iteration=candidate_srsm_iteration,
                    predicted_objective=predicted_objective,
                    srsm_subspace_fraction=candidate_srsm_subspace_fraction,
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

            evaluations.append(evaluation)

        return evaluations

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
        selected_srsm_iteration = None
        selected_predictions = None
        previous_best_before_srsm = None

        if len(feasible_results) >= min_surrogate_training_points:
            previous_best_before_srsm = best_feasible_objective_so_far
            srsm_iteration += 1
            selected_srsm_iteration = srsm_iteration
            active_bounds = make_centered_subspace_bounds(
                best_feasible_design_so_far,
                bounds,
                active_subspace_fraction,
            )
            training_results, model_bounds, training_source = (
                select_response_surface_training_results(
                    feasible_results,
                    active_bounds,
                    bounds,
                    min_surrogate_training_points,
                )
            )
            model = fit_response_surface(training_results, model_bounds, rsm_ridge_lambda)
            candidate_pool_count = max(surrogate_candidate_pool_size, batch_count)
            candidate_pool = generate_unique_geometry_valid_samples(
                config,
                candidate_pool_count,
                evaluated_design_keys,
                cache_round_decimals,
                max_attempts=max_candidate_attempts,
                bounds=active_bounds,
                minimum_required=batch_count,
            )
            predictions = predict_response_surface(model, candidate_pool)
            ranked_indices = np.argsort(predictions)[::-1]
            selected_samples = [
                candidate_pool[index] for index in ranked_indices[:batch_count]
            ]
            selected_predictions = [
                float(predictions[index]) for index in ranked_indices[:batch_count]
            ]
            best_iteration_prediction = selected_predictions[0]
            print(
                f"SRSM iteration {srsm_iteration}: fitted quadratic response surface with "
                f"{model['training_count']} feasible FEM points from {training_source} "
                f"({QUADRATIC_FEATURE_COUNT} features)."
            )
            print(
                f"SRSM iteration {srsm_iteration}: active sub-design fraction "
                f"{active_subspace_fraction:.3g}; best predicted absorbed energy "
                f"is {best_iteration_prediction:.6g}; evaluating top "
                f"{len(selected_samples)} candidates."
            )
            print(
                f"SRSM iteration {srsm_iteration}: active bounds "
                f"{format_bounds_for_log(active_bounds)}"
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
        phase_name = "Exploration"
        if selected_srsm_iteration is not None:
            phase_name = f"SRSM iteration {selected_srsm_iteration}"
        batch_evaluations = evaluate_candidates(
            selected_samples,
            phase_name,
            candidate_srsm_iteration=selected_srsm_iteration,
            predicted_values=selected_predictions,
            candidate_srsm_subspace_fraction=(
                active_subspace_fraction
                if selected_srsm_iteration is not None
                else None
            ),
        )
        if selected_srsm_iteration is not None:
            improved = has_meaningful_improvement(
                previous_best_before_srsm,
                best_feasible_objective_so_far,
                srsm_improvement_tolerance,
            )
            shrink_factor = (
                srsm_subspace_shrink_factor
                if improved
                else srsm_failure_shrink_factor
            )
            active_subspace_fraction = max(
                srsm_min_subspace_fraction,
                active_subspace_fraction * shrink_factor,
            )
            computed_objectives = [
                evaluation["energy_objective"]
                for evaluation in batch_evaluations
                if evaluation["feasible"]
            ]
            computed_text = "no feasible computed objective"
            if computed_objectives:
                computed_text = f"best computed absorbed energy {max(computed_objectives):.6g}"
            adaptation_reason = "improved" if improved else "did not improve"
            print(
                f"SRSM iteration {selected_srsm_iteration}: {computed_text}; "
                f"{adaptation_reason}; next sub-design fraction "
                f"{active_subspace_fraction:.3g}."
            )
            save_convergence_plot(objective_log_path, args.plot_output)
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

    save_convergence_plot(objective_log_path, args.plot_output)

def main(argv=None):
    args = parse_args(argv)
    return run_with_output_log(args, lambda: run_optimizer(args))



if __name__ == "__main__":
    main()

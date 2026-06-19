import json
import sys
import os
import math
import traceback
import numpy as np
import csv

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
PROJECT_FILE = r"./optml_master.FCStd"
# -----------------------------------------------------------------------------
# END OF USER SETTINGS
# -----------------------------------------------------------------------------

# Do not touch the rest of this script unless you really know what you are doing

# name of spreadsheet object in the FCStd file
SPREADSHEET_NAME = "Spreadsheet"  

# --- Update parameters here (name: value) ---
PARAM_UPDATES = json.loads(sys.argv[2])  # safely parse Python literal string

# --- Load FreeCAD ---
import FreeCAD
import Part

# FEM imports
from femtools.ccxtools import FemToolsCcx
from femresult import resulttools

# Optional: only needed if you want explicit Gmsh remeshing
try:
    from femmesh.gmshtools import GmshTools
    HAS_GMSH_TOOLS = True
except Exception:
    HAS_GMSH_TOOLS = False


# -----------------------------------------------------------------------------
# CAD FILE SETTINGS
# -----------------------------------------------------------------------------
SPREADSHEET_NAME = "Spreadsheet"
ANALYSIS_NAME = "Analysis"
CSV_OUT = "triangle_peeq_area_products.csv"

# If True, tries explicit Gmsh remesh when a Gmsh mesh object is found.
# Otherwise relies on doc.recompute() and the existing model setup.
FORCE_GMSH_REMESH = False


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def log(msg):
    FreeCAD.Console.PrintMessage(str(msg) + "\n")
    print(msg)


def fail(msg, code=1):
    FreeCAD.Console.PrintError(str(msg) + "\n")
    print("ERROR:", msg)
    sys.exit(code)


def open_doc(path):
    if not os.path.isfile(path):
        fail(f"FCStd file not found: {path}")
    doc = FreeCAD.openDocument(path)
    if doc is None:
        fail(f"Could not open document: {path}")
    return doc


def get_analysis(doc, analysis_name=None):
    if analysis_name:
        obj = doc.getObject(analysis_name)
        if obj is None:
            fail(f"Analysis object '{analysis_name}' not found")
        return obj

    analyses = [o for o in doc.Objects if o.isDerivedFrom("Fem::FemAnalysis")]
    if len(analyses) != 1:
        fail(f"Expected exactly one FEM analysis, found {len(analyses)}")
    return analyses[0]


def get_solver_from_analysis(analysis):
    solvers = [o for o in analysis.Group if o.isDerivedFrom("Fem::FemSolverObject")]
    if len(solvers) != 1:
        fail(f"Expected exactly one solver in analysis, found {len(solvers)}")
    return solvers[0]


def get_mesh_from_analysis(analysis):
    meshes = [o for o in analysis.Group if o.isDerivedFrom("Fem::FemMeshObject")]
    if len(meshes) != 1:
        fail(f"Expected exactly one mesh object in analysis, found {len(meshes)}")
    return meshes[0]


def update_spreadsheet(sheet, updates):
    aliases = {}
    try:
        aliases = sheet.getAllAliases()
    except Exception:
        pass

    missing = []
    for alias, value in updates.items():
        if alias in aliases:
            cell = aliases[alias]
            sheet.set(cell, str(value))
            log(f"Updated alias '{alias}' in cell {cell} -> {value}")
        else:
            # fallback: try alias as direct cell reference
            try:
                sheet.set(alias, str(value))
                log(f"Updated direct cell '{alias}' -> {value}")
            except Exception:
                missing.append(alias)

    if missing:
        fail(f"These spreadsheet aliases/cells were not found: {missing}")


def assert_no_sketch_solver_failures(doc):
    bad = []
    for obj in doc.Objects:
        if obj.TypeId == "Sketcher::SketchObject":
            msgs = list(getattr(obj, "SolverMessages", []) or [])
            if msgs:
                bad.append((obj.Name, obj.Label, msgs))

    if bad:
        lines = ["Sketch solver failure(s) detected after spreadsheet update:"]
        for name, label, msgs in bad:
            lines.append(f"  - {name} ({label})")
            for m in msgs:
                lines.append(f"      {m}")
        fail("\n".join(lines), code=2)


def assert_no_recompute_failures(doc):
    bad = []
    for obj in doc.Objects:
        state = list(getattr(obj, "State", []) or [])
        if state and any(s in state for s in ("Recompute failed", "InvalidGeometry")):
            bad.append((obj.Name, obj.Label, state))

    if bad:
        lines = ["Document/object recompute failures detected:"]
        for name, label, state in bad:
            lines.append(f"  - {name} ({label}): {state}")
        fail("\n".join(lines), code=3)


def remesh_if_needed(analysis, mesh_obj):
    if not FORCE_GMSH_REMESH:
        return

    if not HAS_GMSH_TOOLS:
        fail("FORCE_GMSH_REMESH=True but femmesh.gmshtools.GmshTools is unavailable")

    # Heuristic for Gmsh mesh objects
    if "Gmsh" in mesh_obj.TypeId or "Gmsh" in getattr(mesh_obj, "Label", ""):
        log(f"Explicitly remeshing with Gmsh for mesh object '{mesh_obj.Name}'")
        gmsh = GmshTools(analysis, mesh_obj)
        error = gmsh.create_mesh()
        if error:
            fail(f"Gmsh remeshing failed: {error}")
    else:
        log("FORCE_GMSH_REMESH=True, but mesh object does not look like a Gmsh mesh. Skipping explicit remesh.")


def run_calculix_analysis(analysis, solver):
    fea = FemToolsCcx(analysis=analysis, solver=solver)
    fea.purge_results()
    fea.run()

    # FemToolsCcx.run() normally writes input, runs ccx, and loads results.
    # Still verify that a result object exists afterward.
    results = [o for o in analysis.Group if o.isDerivedFrom("Fem::FemResultObject")]
    if not results:
        fail("No FEM result object found after solver run")
    return results


def newest_mechanical_result(analysis):
    mech = [o for o in analysis.Group if o.isDerivedFrom("Fem::FemResultMechanical")]
    if not mech:
        # fallback, depending on version/class registration
        mech = [o for o in analysis.Group if o.isDerivedFrom("Fem::FemResultObject")]
    if not mech:
        fail("No mechanical result object found in analysis")

    # choose the last one added to the analysis group
    return mech[-1]


def get_node_peeq_map(result_obj):
    node_numbers = list(getattr(result_obj, "NodeNumbers", []) or [])
    peeq = list(getattr(result_obj, "Peeq", []) or [])
    max_peeq = np.max(np.array(peeq))

    if not node_numbers:
        fail("Result object has no NodeNumbers")
    if not peeq:
        fail("Result object has no Peeq data. Make sure your nonlinear/plastic analysis writes equivalent plastic strain.")

    if len(node_numbers) != len(peeq):
        fail(f"Length mismatch: NodeNumbers={len(node_numbers)} vs Peeq={len(peeq)}")

    return dict(zip(node_numbers, peeq)), max_peeq


def vector_from_fc(v):
    return (float(v.x), float(v.y), float(v.z))


def triangle_area_3d(p1, p2, p3):
    ax, ay, az = p1
    bx, by, bz = p2
    cx, cy, cz = p3

    ux, uy, uz = (bx - ax, by - ay, bz - az)
    vx, vy, vz = (cx - ax, cy - ay, cz - az)

    cxp = uy * vz - uz * vy
    cyp = uz * vx - ux * vz
    czp = ux * vy - uy * vx

    return 0.5 * math.sqrt(cxp * cxp + cyp * cyp + czp * czp)


def get_node_xyz(femmesh, node_id):
    # getNodeById is the standard Python wrapper for the mesh node lookup
    node = femmesh.getNodeById(node_id)
    return vector_from_fc(node)

def get_triangular_face_element_ids(femmesh):
    """
    Return IDs of 3-node triangular face elements in a version-tolerant way.
    """
    tri_ids = []

    # Many FreeCAD Python wrappers expose getFaces()
    if hasattr(femmesh, "getFaces"):
        for elem_id in femmesh.Faces:
            node_ids = list(femmesh.getElementNodes(elem_id))[:3]
            #if elem_id == 500:
                #print(node_ids)
            if len(node_ids) == 3:
                tri_ids.append(elem_id)
        return tri_ids

    # Fallback: use getElementTypes / getIdByElementType if available
    elif hasattr(femmesh, "getElementTypes") and hasattr(femmesh, "getIdByElementType"):
        for etype in femmesh.getElementTypes():
            etype_str = str(etype)
            if "Tria" in etype_str or "Triangle" in etype_str:
                for elem_id in femmesh.getIdByElementType(etype):
                    node_ids = list(femmesh.getElementNodes(elem_id))
                    if len(node_ids) == 3:
                        tri_ids.append(elem_id)
        return tri_ids

    raise AttributeError(
        "Could not find a usable face-element access method on FemMesh. "
        "Available attributes: {}".format([a for a in dir(femmesh) if not a.startswith("_")])
    )


def extract_tri_face_metrics(mesh_obj, result_obj, csv_path):
    femmesh = mesh_obj.FemMesh
    #print(femmesh.Faces)
    node_to_peeq, max_peeq = get_node_peeq_map(result_obj)

    face_ids = get_triangular_face_element_ids(femmesh)
    if not face_ids:
        fail("Mesh has no face-only elements. For a plane stress triangular mesh, face elements are expected.")

    total_weighted_sum = 0.0
    missing_peeq_nodes = set()

    for elem_id in sorted(face_ids):
        node_ids = list(femmesh.getElementNodes(elem_id))[:3]

        # Only triangular elements
        if len(node_ids) != 3:
            continue

        try:
            vals = [float(node_to_peeq[nid]) for nid in node_ids]
        except KeyError as e:
            missing_peeq_nodes.add(int(e.args[0]))
            continue

        p1 = get_node_xyz(femmesh, node_ids[0])
        p2 = get_node_xyz(femmesh, node_ids[1])
        p3 = get_node_xyz(femmesh, node_ids[2])
        #if elem_id == 500:
            #print(p1,p2,p3)

        area = triangle_area_3d(p1, p2, p3)
        avg_peeq = sum(vals) / 3.0
        m_lin = (250 - 100) / 0.2
        avg_energy = 100 * avg_peeq + m_lin * avg_peeq * avg_peeq / 2
        weighted = avg_energy * area
        total_weighted_sum += weighted

    max_stress = 100 + m_lin * max_peeq
    if missing_peeq_nodes:
        fail(f"Some mesh nodes were missing in the result mapping: {sorted(missing_peeq_nodes)[:20]}")

    return total_weighted_sum, max_stress


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # Open project
    doc = open_doc(PROJECT_FILE)

    # Get spreadsheet
    sheet = doc.getObject(SPREADSHEET_NAME)
    log(PARAM_UPDATES)
    if not sheet:
        sys.exit(1)
    try:
        log("Updating spreadsheet parameters ...")
        # Update parameters
        for name, value in PARAM_UPDATES.items():

            sheet.set(name, str(value))

        log("Recomputing document ...")
        sheet.recompute()
        doc.recompute()

        analysis = get_analysis(doc, ANALYSIS_NAME)
        solver = get_solver_from_analysis(analysis)
        mesh_obj = get_mesh_from_analysis(analysis)

        assert_no_sketch_solver_failures(doc)
        assert_no_recompute_failures(doc)

        log("Running CalculiX analysis ...")
        run_calculix_analysis(analysis, solver)

        result_obj = newest_mechanical_result(analysis)
        log(f"Using result object: {result_obj.Name}")

        total_weighted_sum, max_stress = extract_tri_face_metrics(mesh_obj, result_obj, CSV_OUT)

        #doc.save()

        log(f"Total sum of avg(Peeq) * area over triangles = {total_weighted_sum:.12g}")
        mesh_obj = [
        obj for obj in doc.Objects
        if obj.isDerivedFrom("Fem::FemMeshObject")
        ]

    except SystemExit:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        fail(f"{e}\n{tb}", code=99)
        FreeCAD.closeDocument(doc.Name)
    finally:
        try:
            FreeCAD.closeDocument(doc.Name)
            # this has to be the last thing you log!!!
            log(total_weighted_sum)
            log(max_stress)
            sys.exit(0)
        except Exception:
            sys.exit(1)


main()

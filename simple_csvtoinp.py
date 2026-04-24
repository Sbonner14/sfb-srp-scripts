#!/usr/bin/env python3
"""
CalculiX .inp file generator for FDM 3D printing simulation.

Reads a hexahedral mesh (.inp) and an event series (.csv) to produce
a CalculiX input file that simulates the FDM process using:
- Element birth/death via *MODEL CHANGE
- Discrete element activation with nozzle temperature assignment
- Uncoupled thermo-mechanical analysis with convection cooling (*FILM)
- Configurable coarsening strategies to reduce step count

Design decisions:
- C3D8 elements (thermal DOF required for uncoupled analysis)
- *SOLID SECTION placed ONLY in model definition (before *STEP)
- *MODEL CHANGE ADD restores elements with pre-assigned material
- Cooling by *FILM (convection) only — no *RADIATE
- Build plate identified by element centroid Z < 0
- Build plate is NEVER removed and maintains bed temperature
- Event matching never assigns toolpath events to build plate elements
"""

import csv
import sys
import os
import argparse
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree


# ================================================================
# MESH PARSING
# ================================================================

def parse_inp_mesh(filepath):
    """
    Parse an Abaqus/CalculiX .inp file to extract nodes and elements.

    Returns:
        nodes: dict {node_id: (x, y, z)}
        elements: dict {elem_id: [n1, n2, ..., n8]}
        elsets: dict {name: [elem_ids]}
        nsets: dict {name: [node_ids]}
    """
    nodes = {}
    elements = {}
    elsets = defaultdict(list)
    nsets = defaultdict(list)

    current_section = None
    current_set_name = None
    generate = False

    pending_element_id = None
    pending_element_nodes = []
    expected_node_count = 8

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('**'):
                continue

            if line.startswith('*'):
                if pending_element_id is not None:
                    elements[pending_element_id] = \
                        pending_element_nodes[:expected_node_count]
                    if current_set_name:
                        elsets[current_set_name].append(pending_element_id)
                    pending_element_id = None
                    pending_element_nodes = []

                upper = line.upper()

                skip_kw = ('*NODE OUTPUT', '*NODE PRINT', '*NODE FILE',
                           '*ELEMENT OUTPUT', '*ELEMENT PRINT',
                           '*ELEMENT FILE')

                if upper.startswith('*NODE') and not any(
                        upper.startswith(k) for k in skip_kw):
                    current_section = 'NODE'
                    current_set_name = None
                    for p in line.split(',')[1:]:
                        kv = p.strip().split('=')
                        if (len(kv) == 2
                                and kv[0].strip().upper() == 'NSET'):
                            current_set_name = kv[1].strip()
                    continue

                elif upper.startswith('*ELEMENT') and not any(
                        upper.startswith(k) for k in skip_kw):
                    current_section = 'ELEMENT'
                    current_set_name = None
                    for p in line.split(',')[1:]:
                        kv = p.strip().split('=')
                        if (len(kv) == 2
                                and kv[0].strip().upper() == 'ELSET'):
                            current_set_name = kv[1].strip()
                        elif (len(kv) == 2
                              and kv[0].strip().upper() == 'TYPE'):
                            etype = kv[1].strip().upper()
                            if 'C3D20' in etype:
                                expected_node_count = 20
                            elif 'C3D10' in etype:
                                expected_node_count = 10
                            else:
                                expected_node_count = 8
                    continue

                elif upper.startswith('*ELSET'):
                    current_section = 'ELSET'
                    generate = 'GENERATE' in upper
                    for p in line.split(',')[1:]:
                        kv = p.strip().split('=')
                        if (len(kv) == 2
                                and kv[0].strip().upper() == 'ELSET'):
                            current_set_name = kv[1].strip()
                    continue

                elif upper.startswith('*NSET'):
                    current_section = 'NSET'
                    generate = 'GENERATE' in upper
                    for p in line.split(',')[1:]:
                        kv = p.strip().split('=')
                        if (len(kv) == 2
                                and kv[0].strip().upper() == 'NSET'):
                            current_set_name = kv[1].strip()
                    continue

                else:
                    current_section = None
                    current_set_name = None
                    continue

            # Data lines
            if current_section == 'NODE':
                parts = line.replace(',', ' ').split()
                if len(parts) >= 4:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    nodes[nid] = (x, y, z)
                    if current_set_name:
                        nsets[current_set_name].append(nid)

            elif current_section == 'ELEMENT':
                parts = line.rstrip(',').replace(',', ' ').split()
                int_parts = [int(p) for p in parts if p]

                if pending_element_id is not None:
                    pending_element_nodes.extend(int_parts)
                    if len(pending_element_nodes) >= expected_node_count:
                        elements[pending_element_id] = \
                            pending_element_nodes[:expected_node_count]
                        if current_set_name:
                            elsets[current_set_name].append(
                                pending_element_id)
                        pending_element_id = None
                        pending_element_nodes = []
                else:
                    eid = int_parts[0]
                    conn = int_parts[1:]
                    if len(conn) >= expected_node_count:
                        elements[eid] = conn[:expected_node_count]
                        if current_set_name:
                            elsets[current_set_name].append(eid)
                    else:
                        pending_element_id = eid
                        pending_element_nodes = conn

            elif current_section == 'ELSET' and current_set_name:
                if generate:
                    parts = line.replace(',', ' ').split()
                    int_parts = [int(p) for p in parts if p]
                    start, end = int_parts[0], int_parts[1]
                    step = int_parts[2] if len(int_parts) > 2 else 1
                    elsets[current_set_name].extend(
                        range(start, end + 1, step))
                else:
                    for p in line.replace(',', ' ').split():
                        p = p.strip()
                        if p:
                            try:
                                elsets[current_set_name].append(int(p))
                            except ValueError:
                                pass

            elif current_section == 'NSET' and current_set_name:
                if generate:
                    parts = line.replace(',', ' ').split()
                    int_parts = [int(p) for p in parts if p]
                    start, end = int_parts[0], int_parts[1]
                    step = int_parts[2] if len(int_parts) > 2 else 1
                    nsets[current_set_name].extend(
                        range(start, end + 1, step))
                else:
                    for p in line.replace(',', ' ').split():
                        p = p.strip()
                        if p:
                            try:
                                nsets[current_set_name].append(int(p))
                            except ValueError:
                                pass

    if pending_element_id is not None:
        elements[pending_element_id] = \
            pending_element_nodes[:expected_node_count]
        if current_set_name:
            elsets[current_set_name].append(pending_element_id)

    return nodes, elements, elsets, nsets


# ================================================================
# EVENT SERIES PARSING
# ================================================================

def parse_event_series(filepath):
    """Parse event series CSV: time, x, y, z, extruding."""
    events = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if header:
            try:
                float(header[0].strip())
                events.append({
                    'time': float(header[0]),
                    'x': float(header[1]),
                    'y': float(header[2]),
                    'z': float(header[3]),
                    'extruding': int(float(header[4]))
                })
            except ValueError:
                pass

        for row in reader:
            if len(row) < 5:
                continue
            row = [r.strip() for r in row]
            if not row[0]:
                continue
            try:
                events.append({
                    'time': float(row[0]),
                    'x': float(row[1]),
                    'y': float(row[2]),
                    'z': float(row[3]),
                    'extruding': int(float(row[4]))
                })
            except ValueError:
                continue

    events.sort(key=lambda e: e['time'])
    return events


# ================================================================
# GEOMETRY UTILITIES
# ================================================================

def compute_element_centroids(nodes, elements):
    """Compute the centroid of each element."""
    centroids = {}
    for eid, conn in elements.items():
        coords = [nodes[nid] for nid in conn if nid in nodes]
        if coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cz = sum(c[2] for c in coords) / len(coords)
            centroids[eid] = (cx, cy, cz)
    return centroids


def compute_element_characteristic_size(nodes, elements):
    """Median element half-diagonal."""
    sizes = []
    for eid, conn in elements.items():
        coords = [nodes[nid] for nid in conn if nid in nodes]
        if not coords:
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)
        sizes.append(np.sqrt(dx**2 + dy**2 + dz**2) / 2.0)
    return float(np.median(sizes)) if sizes else 1.0


def get_element_nodes(elements, elem_ids):
    """Get all unique node IDs belonging to a set of elements."""
    node_set = set()
    for eid in elem_ids:
        if eid in elements:
            for nid in elements[eid]:
                node_set.add(nid)
    return sorted(node_set)


def get_external_faces(elements, elem_ids_set):
    """
    Identify external faces for convection.
    A face is external if not shared with another element in the set.
    """
    hex_faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    face_count = defaultdict(int)
    face_to_elem = defaultdict(list)

    for eid in elem_ids_set:
        if eid not in elements:
            continue
        conn = elements[eid]
        for fi, face_local in enumerate(hex_faces):
            face_key = tuple(sorted(conn[i] for i in face_local))
            face_count[face_key] += 1
            face_to_elem[face_key].append((eid, fi + 1))

    external_faces = []
    for face_key, count in face_count.items():
        if count == 1:
            for eid, face_num in face_to_elem[face_key]:
                external_faces.append((eid, face_num))

    return external_faces


# ================================================================
# BUILD PLATE IDENTIFICATION
# ================================================================

def identify_build_plate(elements, centroids):
    """
    Identify build plate elements by centroid Z < 0.

    The user's mesh has build plate elements below Z=0 and
    printed model elements at Z >= 0. This is the sole criterion
    for distinguishing them.

    Returns:
        buildplate_eids: set of element IDs with centroid Z < 0
        model_eids: set of element IDs with centroid Z >= 0
    """
    buildplate_eids = set()
    model_eids = set()

    for eid in elements:
        if eid not in centroids:
            continue
        cz = centroids[eid][2]
        if cz < 0.0:
            buildplate_eids.add(eid)
        else:
            model_eids.add(eid)

    return buildplate_eids, model_eids


# ================================================================
# EVENT-TO-ELEMENT ASSIGNMENT
# ================================================================

def assign_events_to_elements(events, centroids, elements, nodes,
                              model_eids, search_radius_factor=1.5):
    """
    Assign extrusion events to mesh elements. Only considers elements
    in model_eids (NOT build plate elements).

    Multi-pass strategy:
      Pass 1: Each event -> nearest model element (KD-tree)
      Pass 2: Each unmatched model element -> nearby events (reverse)
      Pass 3: Neighbor propagation with relaxing threshold
      Pass 4: Convex hull envelope per Z-layer

    Returns:
        activation_sequence: list of (elem_id, time) sorted by time
        activated_elements: set of activated element IDs
    """
    extrusion_events = [e for e in events if e['extruding'] == 1]
    if not extrusion_events:
        return [], set()

    # Only consider model elements, never build plate
    model_elem_ids = sorted(model_eids)
    model_centroids = {eid: centroids[eid] for eid in model_elem_ids
                       if eid in centroids}
    model_ids_list = list(model_centroids.keys())
    model_centroid_array = np.array([model_centroids[eid]
                                     for eid in model_ids_list])

    # KD-tree of model element centroids only
    tree = cKDTree(model_centroid_array)

    event_coords = np.array([[e['x'], e['y'], e['z']] for e in extrusion_events])
    event_times = np.array([e['time'] for e in extrusion_events])
    event_tree = cKDTree(event_coords)

    activation_map = {}

    # ---- Pass 1: Each event -> nearest model element ----
    distances, indices = tree.query(event_coords, k=1)
    for ei in range(len(extrusion_events)):
        eid = model_ids_list[indices[ei]]
        t = event_times[ei]
        if eid not in activation_map or t < activation_map[eid]:
            activation_map[eid] = t

    pass1_count = len(activation_map)
    print(f"    Pass 1 (event->nearest model element): "
          f"{pass1_count} elements")

    # ---- Pass 2: Each unmatched model element -> nearby events ----
    elem_radii = {}
    for eid in model_ids_list:
        conn = elements[eid]
        coords = [nodes[nid] for nid in conn if nid in nodes]
        if not coords:
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)
        elem_radii[eid] = (np.sqrt(dx**2 + dy**2 + dz**2) / 2.0
                           * search_radius_factor)

    for eid in model_ids_list:
        if eid in activation_map:
            continue
        if eid not in elem_radii:
            continue
        cx, cy, cz = model_centroids[eid]
        radius = elem_radii[eid]

        nearby = event_tree.query_ball_point([cx, cy, cz], radius)
        if nearby:
            earliest_idx = min(nearby, key=lambda i: event_times[i])
            activation_map[eid] = event_times[earliest_idx]

    pass2_count = len(activation_map) - pass1_count
    print(f"    Pass 2 (element->nearby events): "
          f"{pass2_count} additional")

    # ---- Pass 3: Neighbor propagation with relaxing threshold ----
    # Build face-adjacency among model elements only
    face_to_elem = defaultdict(list)
    hex_faces = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    for eid in model_ids_list:
        conn = elements[eid]
        for face_local in hex_faces:
            face_key = tuple(sorted(conn[i] for i in face_local))
            face_to_elem[face_key].append(eid)

    adjacency = defaultdict(set)
    for face_key, eids_sharing in face_to_elem.items():
        if len(eids_sharing) == 2:
            e1, e2 = eids_sharing
            adjacency[e1].add(e2)
            adjacency[e2].add(e1)

    thresholds = [0.5, 0.4, 0.3, 0.2]
    max_iters = 30
    total_propagated = 0
    total_iterations = 0

    for threshold in thresholds:
        for _ in range(max_iters):
            newly_activated = {}
            for eid in model_ids_list:
                if eid in activation_map:
                    continue
                neighbors = adjacency.get(eid, set())
                if not neighbors:
                    continue
                active_times = [activation_map[n] for n in neighbors
                                if n in activation_map]
                if not active_times:
                    continue
                frac = len(active_times) / len(neighbors)
                if frac >= threshold:
                    newly_activated[eid] = float(np.median(active_times))

            if not newly_activated:
                break
            activation_map.update(newly_activated)
            total_propagated += len(newly_activated)
            total_iterations += 1

    # Final: activate if ANY neighbor is active
    for _ in range(max_iters):
        newly_activated = {}
        for eid in model_ids_list:
            if eid in activation_map:
                continue
            neighbors = adjacency.get(eid, set())
            active_times = [activation_map[n] for n in neighbors
                            if n in activation_map]
            if active_times:
                newly_activated[eid] = float(np.median(active_times))

        if not newly_activated:
            break
        activation_map.update(newly_activated)
        total_propagated += len(newly_activated)
        total_iterations += 1

    print(f"    Pass 3 (neighbor propagation, "
          f"{total_iterations} iters): {total_propagated} additional")

    # ---- Pass 4: Z-envelope convex hull ----
    remaining = [eid for eid in model_ids_list
                 if eid not in activation_map]

    pass4_count = 0
    if remaining:
        event_z_min = event_coords[:, 2].min()
        event_z_max = event_coords[:, 2].max()

        all_model_z = sorted(set(
            round(model_centroids[eid][2], 6)
            for eid in model_ids_list if eid in model_centroids))

        if len(all_model_z) > 1:
            spacings = [all_model_z[i+1] - all_model_z[i]
                        for i in range(len(all_model_z) - 1)]
            layer_height = float(np.median(spacings))
        else:
            layer_height = 0.2

        half_layer = layer_height * 0.6

        try:
            from scipy.spatial import Delaunay

            layer_hulls = {}
            layer_times = {}

            for z_level in all_model_z:
                if z_level < event_z_min - half_layer:
                    continue
                if z_level > event_z_max + half_layer:
                    continue

                z_mask = (np.abs(event_coords[:, 2] - z_level)
                          <= half_layer)
                layer_xy = event_coords[z_mask][:, :2]
                layer_t = event_times[z_mask]

                if len(layer_xy) < 3:
                    continue

                try:
                    hull = Delaunay(layer_xy)
                    layer_hulls[z_level] = hull
                    layer_times[z_level] = float(np.median(layer_t))
                except Exception:
                    continue

            for eid in remaining:
                if eid in activation_map:
                    continue
                if eid not in model_centroids:
                    continue

                cx, cy, cz = model_centroids[eid]

                if cz < event_z_min - half_layer:
                    continue
                if cz > event_z_max + half_layer:
                    continue

                closest_z = min(layer_hulls.keys(),
                                key=lambda z: abs(z - cz),
                                default=None)
                if closest_z is None:
                    continue
                if abs(closest_z - cz) > half_layer * 2:
                    continue

                hull = layer_hulls[closest_z]
                if hull.find_simplex(np.array([cx, cy])) >= 0:
                    activation_map[eid] = layer_times[closest_z]
                    pass4_count += 1

        except ImportError:
            print("    Pass 4 skipped (scipy.spatial.Delaunay "
                  "not available)")

    print(f"    Pass 4 (Z-envelope hull): {pass4_count} additional")

    activation_sequence = sorted(activation_map.items(),
                                 key=lambda x: x[1])
    activated_elements = set(activation_map.keys())

    return activation_sequence, activated_elements


def print_activation_diagnostics(events, centroids, elements,
                                 model_eids, buildplate_eids,
                                 activated_elements):
    """Print diagnostic information about activation coverage."""
    extrusion_events = [e for e in events if e['extruding'] == 1]
    if not extrusion_events:
        print("    No extrusion events found!")
        return

    event_coords = np.array([[e['x'], e['y'], e['z']]
                             for e in extrusion_events])

    model_centroids = np.array([centroids[eid] for eid in model_eids
                                if eid in centroids])

    print(f"\n    --- Activation Diagnostics ---")
    print(f"    Build plate elements (Z < 0): {len(buildplate_eids)}")
    print(f"    Model elements (Z >= 0): {len(model_eids)}")
    print(f"    Event coordinate ranges:")
    print(f"      X: [{event_coords[:, 0].min():.4f}, "
          f"{event_coords[:, 0].max():.4f}]")
    print(f"      Y: [{event_coords[:, 1].min():.4f}, "
          f"{event_coords[:, 1].max():.4f}]")
    print(f"      Z: [{event_coords[:, 2].min():.4f}, "
          f"{event_coords[:, 2].max():.4f}]")

    if len(model_centroids) > 0:
        print(f"    Model centroid ranges (Z >= 0 only):")
        print(f"      X: [{model_centroids[:, 0].min():.4f}, "
              f"{model_centroids[:, 0].max():.4f}]")
        print(f"      Y: [{model_centroids[:, 1].min():.4f}, "
              f"{model_centroids[:, 1].max():.4f}]")
        print(f"      Z: [{model_centroids[:, 2].min():.4f}, "
              f"{model_centroids[:, 2].max():.4f}]")

        for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            e_min = event_coords[:, axis].min()
            e_max = event_coords[:, axis].max()
            m_min = model_centroids[:, axis].min()
            m_max = model_centroids[:, axis].max()
            overlap_min = max(e_min, m_min)
            overlap_max = min(e_max, m_max)
            if overlap_min > overlap_max:
                print(f"    WARNING: NO OVERLAP in {name}! "
                      f"Events [{e_min:.4f},{e_max:.4f}] vs "
                      f"Model [{m_min:.4f},{m_max:.4f}]")
            else:
                e_range = e_max - e_min
                pct = ((overlap_max - overlap_min)
                       / max(e_range, 1e-30)) * 100
                print(f"    {name} overlap: "
                      f"[{overlap_min:.4f}, {overlap_max:.4f}] "
                      f"({pct:.1f}% of event range)")

    activated_model = activated_elements & model_eids
    activated_bp = activated_elements & buildplate_eids
    print(f"    Model elements activated: {len(activated_model)} / "
          f"{len(model_eids)} "
          f"({100.0 * len(activated_model) / max(len(model_eids), 1):.1f}%)")
    if activated_bp:
        print(f"    WARNING: {len(activated_bp)} build plate elements "
              f"were activated (should be 0)")
    print(f"    --- End Diagnostics ---\n")


# ================================================================
# COARSENING / GROUPING STRATEGIES
# ================================================================

def group_by_exact_time(activation_sequence, time_tolerance=1e-8):
    """Group elements that activate at the same time."""
    if not activation_sequence:
        return []
    groups = []
    current_time = activation_sequence[0][1]
    current_group = [activation_sequence[0][0]]
    for eid, t in activation_sequence[1:]:
        if abs(t - current_time) < time_tolerance:
            current_group.append(eid)
        else:
            groups.append((current_time, current_group))
            current_time = t
            current_group = [eid]
    groups.append((current_time, current_group))
    return groups


def group_by_layer(activation_sequence, elements, nodes,
                   layer_tolerance=None):
    """Group all activations at the same Z-height."""
    if not activation_sequence:
        return []
    elem_z = {}
    for eid, t in activation_sequence:
        conn = elements[eid]
        z_coords = [nodes[nid][2] for nid in conn if nid in nodes]
        if z_coords:
            elem_z[eid] = np.mean(z_coords)

    if layer_tolerance is None:
        z_vals = sorted(set(round(z, 6) for z in elem_z.values()))
        if len(z_vals) > 1:
            spacings = [z_vals[i+1] - z_vals[i]
                        for i in range(len(z_vals) - 1)]
            layer_tolerance = min(spacings) * 0.4
        else:
            layer_tolerance = 0.01

    layer_groups = defaultdict(list)
    for eid, t in activation_sequence:
        if eid in elem_z:
            z_layer = round(elem_z[eid] / layer_tolerance) * layer_tolerance
            layer_groups[z_layer].append((eid, t))

    result = []
    for z in sorted(layer_groups.keys()):
        elems_and_times = layer_groups[z]
        earliest_time = min(t for _, t in elems_and_times)
        elem_ids = [eid for eid, _ in elems_and_times]
        result.append((earliest_time, elem_ids))
    return result


def group_by_time_window(activation_sequence, window_size=5.0):
    """Group all activations within a fixed time window."""
    if not activation_sequence:
        return []
    groups = []
    current_start = activation_sequence[0][1]
    current_group = []
    for eid, t in activation_sequence:
        if t - current_start <= window_size:
            current_group.append(eid)
        else:
            if current_group:
                groups.append((current_start, current_group))
            current_start = t
            current_group = [eid]
    if current_group:
        groups.append((current_start, current_group))
    return groups


def group_by_n_layers(activation_sequence, elements, nodes,
                      n_layers_per_step=5, layer_tolerance=None):
    """Combine every N layers into one step."""
    layer_groups = group_by_layer(activation_sequence, elements, nodes,
                                  layer_tolerance)
    lumped = []
    for i in range(0, len(layer_groups), n_layers_per_step):
        batch = layer_groups[i:i + n_layers_per_step]
        earliest = min(t for t, _ in batch)
        all_elems = []
        for _, elems in batch:
            all_elems.extend(elems)
        lumped.append((earliest, all_elems))
    return lumped


def group_by_layer_sectors(activation_sequence, elements, nodes,
                           sectors_per_layer=4, layer_tolerance=None):
    """Layer grouping + N intra-layer sectors by time order."""
    layer_groups = group_by_layer(activation_sequence, elements, nodes,
                                  layer_tolerance)
    time_lookup = dict(activation_sequence)
    refined = []
    for t_layer, elem_ids in layer_groups:
        elem_times = [(eid, time_lookup.get(eid, t_layer))
                      for eid in elem_ids]
        elem_times.sort(key=lambda x: x[1])
        n = max(1, len(elem_times))
        chunk_size = max(1, n // sectors_per_layer)
        for i in range(0, n, chunk_size):
            chunk = elem_times[i:i + chunk_size]
            refined.append((chunk[0][1], [e for e, _ in chunk]))
    return refined


def apply_coarsening(activation_sequence, elements, nodes, method, param):
    """Dispatch to the selected coarsening strategy."""
    if method == 'none':
        return group_by_exact_time(activation_sequence)
    elif method == 'layer':
        tol = param if param is not None else None
        return group_by_layer(activation_sequence, elements, nodes,
                              layer_tolerance=tol)
    elif method == 'time':
        window = param if param is not None else 5.0
        return group_by_time_window(activation_sequence,
                                    window_size=window)
    elif method == 'multilayer':
        n = int(param) if param is not None else 5
        return group_by_n_layers(activation_sequence, elements, nodes,
                                 n_layers_per_step=n)
    elif method == 'sectors':
        n = int(param) if param is not None else 4
        return group_by_layer_sectors(activation_sequence, elements, nodes,
                                      sectors_per_layer=n)
    else:
        raise ValueError(f"Unknown coarsening method: '{method}'")


# ================================================================
# .INP WRITING UTILITIES
# ================================================================

def write_elset(f, name, elem_ids, items_per_line=10):
    """Write an element set."""
    f.write(f"*ELSET, ELSET={name}\n")
    ids = sorted(elem_ids)
    for i, eid in enumerate(ids):
        if i > 0 and i % items_per_line == 0:
            f.write("\n")
        if i % items_per_line != 0:
            f.write(", ")
        f.write(f"{eid}")
    f.write("\n")


def write_nset(f, name, node_ids, items_per_line=10):
    """Write a node set."""
    f.write(f"*NSET, NSET={name}\n")
    ids = sorted(node_ids)
    for i, nid in enumerate(ids):
        if i > 0 and i % items_per_line == 0:
            f.write("\n")
        if i % items_per_line != 0:
            f.write(", ")
        f.write(f"{nid}")
    f.write("\n")


def validate_activation_groups(activation_groups, elements,
                               buildplate_eids):
    """
    Ensure every element ID in every activation group:
    1. Exists in the elements dict (will be written to *ELEMENT)
    2. Is NOT a build plate element

    Returns cleaned activation_groups and reports any removals.
    """
    valid_eids = set(elements.keys())
    total_removed = 0
    cleaned_groups = []

    for t, eids in activation_groups:
        clean = []
        for eid in eids:
            if eid not in valid_eids:
                total_removed += 1
            elif eid in buildplate_eids:
                total_removed += 1
            else:
                clean.append(eid)
        if clean:
            cleaned_groups.append((t, clean))

    if total_removed > 0:
        print(f"    Validation: removed {total_removed} invalid element "
              f"references from activation groups")

    return cleaned_groups


def validate_activated_set(all_activated_elems, elements, buildplate_eids):
    """
    Ensure the activated element set only contains valid model elements.
    """
    valid_eids = set(elements.keys())
    cleaned = all_activated_elems & valid_eids - buildplate_eids
    removed = len(all_activated_elems) - len(cleaned)
    if removed > 0:
        print(f"    Validation: removed {removed} invalid elements "
              f"from activated set")
    return cleaned

# ================================================================
# MAIN .INP FILE WRITER
# ================================================================

def write_inp_file(output_path, nodes, elements, activation_groups,
                   all_activated_elems, all_elem_ids,
                   buildplate_eids, model_eids,
                   nozzle_temp=200.0, bed_temp=25.0, ambient_temp=25.0,
                   convection_coeff=10.0, material_props=None, output_frequency=10):
    # output_frequency: write results every N activation groups.
    #   0 = only final step
    #   1 = every step (huge files)
    #   10 = every 10th group (reasonable)
    if material_props is None:
        material_props = {
            'density': 1.25e-9,
            'conductivity': 0.15,
            'specific_heat': 1.4e9,
            'youngs_modulus': 2634.0,
            'poissons_ratio': 0.35,
            'thermal_expansion': 7.25e-5,
        }

    # THE definitive valid ID sets — nothing else is trusted
    valid_elem_ids = set(elements.keys())
    valid_node_ids = set(nodes.keys())

    def safe_elems(eids):
        """Return only element IDs that actually exist."""
        return sorted(eid for eid in eids if eid in valid_elem_ids)

    def safe_nodes(nids):
        """Return only node IDs that actually exist."""
        return sorted(nid for nid in nids if nid in valid_node_ids)

    def safe_faces(face_list):
        """Return only faces whose element actually exists."""
        return [(eid, fnum) for eid, fnum in face_list
                if eid in valid_elem_ids]

    # Validated sets
    bp_elems = safe_elems(buildplate_eids)
    bp_elem_set = set(bp_elems)
    printed_elems = safe_elems(all_activated_elems)
    printed_elem_set = set(printed_elems)
    unactivated_model = safe_elems(model_eids - all_activated_elems)

    all_node_ids = sorted(nodes.keys())
    buildplate_nodes = safe_nodes(
        get_element_nodes(elements, bp_elems))

    # BC nodes
    if len(buildplate_nodes) >= 3:
        bp_arr = np.array([nodes[n] for n in buildplate_nodes])
        min_corner = bp_arr.min(axis=0)
        idx0 = np.argmin(np.linalg.norm(bp_arr - min_corner, axis=1))
        bc_node1 = buildplate_nodes[idx0]
        max_x = np.array([bp_arr[:, 0].max(),
                          min_corner[1], min_corner[2]])
        idx1 = np.argmin(np.linalg.norm(bp_arr - max_x, axis=1))
        bc_node2 = buildplate_nodes[idx1]
        max_y = np.array([min_corner[0],
                          bp_arr[:, 1].max(), min_corner[2]])
        idx2 = np.argmin(np.linalg.norm(bp_arr - max_y, axis=1))
        bc_node3 = buildplate_nodes[idx2]
        bc_nodes = [bc_node1, bc_node2, bc_node3]
    else:
        bc_nodes = buildplate_nodes[:1]

    # Build plate bottom nodes
    bp_z_values = [nodes[n][2] for n in buildplate_nodes]
    bp_z_min = min(bp_z_values)
    bp_z_tol = (max(bp_z_values) - bp_z_min) * 0.1 + 1e-6
    bp_bottom_nodes = safe_nodes(
        [n for n in buildplate_nodes
         if nodes[n][2] <= bp_z_min + bp_z_tol])

    # Pre-validate ALL activation groups
    clean_groups = []
    for t, eids in activation_groups:
        clean = safe_elems(eids)
        if clean:
            clean_groups.append((t, clean))
    activation_groups = clean_groups

    # Diagnostic: verify no invalid IDs exist anywhere
    all_group_eids = set()
    for _, eids in activation_groups:
        all_group_eids.update(eids)
    invalid_in_groups = all_group_eids - valid_elem_ids
    if invalid_in_groups:
        print(f"  BUG: {len(invalid_in_groups)} invalid element IDs "
              f"in activation groups after validation!")
        print(f"  Examples: {sorted(invalid_in_groups)[:10]}")

    invalid_in_printed = printed_elem_set - valid_elem_ids
    if invalid_in_printed:
        print(f"  BUG: {len(invalid_in_printed)} invalid element IDs "
              f"in PRINTED_ALL after validation!")

    with open(output_path, 'w') as f:
        # ---- HEADER ----
        f.write("**\n")
        f.write("** CalculiX FDM 3D Printing Simulation\n")
        f.write(f"** Build plate elements (Z<0): {len(bp_elems)}\n")
        f.write(f"** Printed elements: {len(printed_elems)}\n")
        f.write(f"** Unactivated model elements: "
                f"{len(unactivated_model)}\n")
        f.write(f"** Activation groups: {len(activation_groups)}\n")
        f.write(f"** Total steps: {1 + 2 * len(activation_groups)}\n")
        f.write(f"** Nozzle: {nozzle_temp} C, Bed: {bed_temp} C, "
                f"Ambient: {ambient_temp} C\n")
        f.write("** Element type: C3D8\n")
        f.write("**\n\n")

        # ---- NODES ----
        f.write("*NODE\n")
        for nid in all_node_ids:
            x, y, z = nodes[nid]
            f.write(f"{nid}, {x:.10e}, {y:.10e}, {z:.10e}\n")

        # ---- ELEMENTS ----
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for eid in sorted(elements.keys()):
            conn = elements[eid]
            f.write(f"{eid}, {', '.join(str(n) for n in conn)}\n")

        # ---- SETS ----
        # Write sets with trailing blank line to ensure separation
        write_nset(f, "NALL", all_node_ids)
        f.write("\n")
        write_elset(f, "EALL", sorted(valid_elem_ids))
        f.write("\n")

        write_elset(f, "BUILDPLATE", bp_elems)
        f.write("\n")
        write_nset(f, "NBUILDPLATE", buildplate_nodes)
        f.write("\n")
        write_nset(f, "NBUILDPLATE_BOT", bp_bottom_nodes)
        f.write("\n")

        if printed_elems:
            write_elset(f, "PRINTED_ALL", printed_elems)
            f.write("\n")

        if unactivated_model:
            write_elset(f, "MODEL_INACTIVE", unactivated_model)
            f.write("\n")

        for gi, (t, eids) in enumerate(activation_groups):
            write_elset(f, f"ACTGRP_{gi + 1}", eids)
            f.write("\n")
            grp_nodes = safe_nodes(get_element_nodes(elements, eids))
            write_nset(f, f"NACTGRP_{gi + 1}", grp_nodes)
            f.write("\n")

        # ---- MATERIALS ----
        f.write("*MATERIAL, NAME=MAT_PRINT\n")
        f.write("*DENSITY\n")
        f.write(f"{material_props['density']:.6e}\n")
        f.write("*ELASTIC\n")
        f.write(f"100.0, {material_props['poissons_ratio']:.4f}, {nozzle_temp:.1f}\n")
        f.write(f"{material_props['youngs_modulus']:.1f}, {material_props['poissons_ratio']:.4f}, {ambient_temp:.1f}\n")
        f.write("*EXPANSION\n")
        f.write(f"{material_props['thermal_expansion']:.6e}\n")
        f.write("*CONDUCTIVITY\n")
        f.write(f"{material_props['conductivity']:.6e}\n")
        f.write("*SPECIFIC HEAT\n")
        f.write(f"{material_props['specific_heat']:.6e}\n")
        f.write("\n")

        f.write("*MATERIAL, NAME=MAT_BUILDPLATE\n")
        f.write("*DENSITY\n")
        f.write(f"{material_props['density']:.6e}\n")
        f.write("*ELASTIC\n")
        f.write(f"{material_props['youngs_modulus']:.1f}, "
                f"{material_props['poissons_ratio']:.4f}\n")
        f.write("*EXPANSION\n")
        f.write(f"{material_props['thermal_expansion']:.6e}\n")
        f.write("*CONDUCTIVITY\n")
        f.write(f"{material_props['conductivity']:.6e}\n")
        f.write("*SPECIFIC HEAT\n")
        f.write(f"{material_props['specific_heat']:.6e}\n")
        f.write("\n")

        # ---- SECTIONS ----
        f.write("*SOLID SECTION, ELSET=BUILDPLATE, "
                "MATERIAL=MAT_BUILDPLATE\n\n")
        if printed_elems:
            f.write("*SOLID SECTION, ELSET=PRINTED_ALL, "
                    "MATERIAL=MAT_PRINT\n\n")
        if unactivated_model:
            f.write("*SOLID SECTION, ELSET=MODEL_INACTIVE, "
                    "MATERIAL=MAT_PRINT\n\n")

        # ---- INITIAL CONDITIONS ----
        f.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        f.write(f"NALL, {ambient_temp:.1f}\n\n")

        # ============================================================
        # HELPER: write boundary conditions block
        # ============================================================
        def write_mechanical_bcs():
            f.write("*BOUNDARY\n")
            if len(bc_nodes) >= 3:
                f.write(f"{bc_nodes[0]}, 1, 3, 0.0\n")
                f.write(f"{bc_nodes[1]}, 2, 3, 0.0\n")
                f.write(f"{bc_nodes[2]}, 3, 3, 0.0\n")
            else:
                f.write(f"{bc_nodes[0]}, 1, 3, 0.0\n")

        def write_bed_temp_bc():
            f.write("*BOUNDARY\n")
            f.write(f"NBUILDPLATE_BOT, 11, 11, {bed_temp:.1f}\n")

        def write_all_persistent_bcs_new():
            """Write with OP=NEW — clears previous, re-applies."""
            f.write("*BOUNDARY, OP=NEW\n")
            if len(bc_nodes) >= 3:
                f.write(f"{bc_nodes[0]}, 1, 3, 0.0\n")
                f.write(f"{bc_nodes[1]}, 2, 3, 0.0\n")
                f.write(f"{bc_nodes[2]}, 3, 3, 0.0\n")
            else:
                f.write(f"{bc_nodes[0]}, 1, 3, 0.0\n")
            f.write(f"NBUILDPLATE_BOT, 11, 11, {bed_temp:.1f}\n")

        def write_film(active_elem_set):
            """Write *FILM only for verified active elements."""
            ext_faces = get_external_faces(elements, active_elem_set)
            verified = safe_faces(ext_faces)
            # Extra check: only elements in active set
            verified = [(eid, fn) for eid, fn in verified
                        if eid in active_elem_set]
            if verified:
                f.write("*FILM\n")
                for eid, fnum in verified:
                    f.write(f"{eid}, F{fnum}, "
                            f"{ambient_temp:.1f}, "
                            f"{convection_coeff:.4f}\n")

        # ============================================================
        # STEP 1: Dummy step with ALL elements active
        # Forces CalculiX to write complete geometry to .frd file
        # so that elements added later via *MODEL CHANGE are visible
        # in post-processing.
        # ============================================================
        f.write("**\n** STEP 1: Dummy — write full geometry to .frd\n")
        f.write("** All elements active so .frd contains full mesh\n**\n")
        f.write("*STEP\n")
        f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
        f.write("1.0, 1.0, 1e-8, 1.0\n")
        f.write("**\n")

        # Mechanical BCs needed even for dummy step
        write_mechanical_bcs()
        write_bed_temp_bc()

        # Set everything to ambient temperature
        f.write("*BOUNDARY\n")
        f.write(f"NALL, 11, 11, {ambient_temp:.1f}\n")

        # Write full output so geometry is registered
        f.write("*NODE FILE\n")
        f.write("NT, U\n")
        f.write("*EL FILE\n")
        f.write("S\n")
        f.write("*END STEP\n\n")

        # ============================================================
        # STEP 2: Remove printed + inactive model elements
        # ONLY build plate remains active
        # ============================================================
        f.write("**\n** STEP 2: Remove printed elements\n**\n")
        f.write("*STEP\n")
        f.write("*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
        f.write("1.0, 1.0, 1e-8, 1.0\n")

        if printed_elems:
            f.write("*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n")
            f.write("PRINTED_ALL\n")
        if unactivated_model:
            f.write("*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n")
            f.write("MODEL_INACTIVE\n")

        write_mechanical_bcs()
        write_bed_temp_bc()

        # FILM only on build plate — the ONLY active elements now
        write_film(bp_elem_set)

        f.write("*NODE FILE, FREQUENCY=0\n")
        f.write("*EL FILE, FREQUENCY=0\n")
        f.write("*END STEP\n\n")

        # ============================================================
        # ACTIVATION + COOLING STEPS
        # ============================================================
        step_num = 3
        cumulative_activated = set()
        total_groups = len(activation_groups)

        for gi, (act_time, elem_ids) in enumerate(activation_groups):
            is_last = (gi == total_groups - 1)
            is_output_step = (is_last or (output_frequency > 0 and gi % output_frequency == 0))
            
            group_name = f"ACTGRP_{gi + 1}"
            nset_name = f"NACTGRP_{gi + 1}"

            if gi == 0:
                dt = max(act_time, 0.1)
            else:
                dt = max(act_time - activation_groups[gi - 1][0], 0.01)
            act_dt = min(dt * 0.1, 0.1)
            if act_dt < 1e-6:
                act_dt = 0.01

            # ---- ACTIVATION STEP ----
            f.write(f"** STEP {step_num}: Activate {group_name} "
                    f"(t={act_time:.4f}s, {len(elem_ids)} elems)\n")
                        # ---- ACTIVATION STEP ----
            f.write(f"*STEP, INC=10000, INCF=50\n")
            f.write(f"*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
            f.write(f"{act_dt:.6e}, {act_dt:.6e}, "
                    f"{act_dt * 1e-6:.6e}, {act_dt:.6e}\n")
            f.write("*CONTROLS, PARAMETERS=TIME INCREMENTATION\n")
            f.write("8,10,9,16,10,4,,5\n")
            f.write("0.25,0.5,0.75,0.85,,,1.5,\n")
            f.write("*CONTROLS, PARAMETERS=FIELD\n")
            f.write("0.05,0.01\n")

            f.write(f"*MODEL CHANGE, TYPE=ELEMENT, ADD\n")
            f.write(f"{group_name}\n")

            f.write(f"*BOUNDARY\n")
            f.write(f"{nset_name}, 11, 11, {nozzle_temp:.1f}\n")

            write_mechanical_bcs()
            write_bed_temp_bc()

            if is_output_step:
                f.write("*NODE FILE\n")
                f.write("NT, U\n")
                f.write("*EL FILE\n")
                f.write("S\n")
            else:
                f.write("*NODE FILE, FREQUENCY=0\n")
                f.write("*EL FILE, FREQUENCY=0\n")
            f.write("*END STEP\n\n")
            step_num += 1

            # ---- COOLING STEP ----
            if gi < len(activation_groups) - 1:
                next_time = activation_groups[gi + 1][0]
                cool_dt = max(next_time - act_time - act_dt, 0.1)
            else:
                cool_dt = 60.0

            init_inc = min(cool_dt * 0.001, 0.1)
            min_inc = max(init_inc * 1e-6, 1e-12)
            max_inc = cool_dt * 0.5

            f.write(f"** STEP {step_num}: Cooling after {group_name} "
                    f"({cool_dt:.4f}s)\n")
            f.write(f"*STEP, INC=1000000, INCF=50\n")
            f.write(f"*UNCOUPLED TEMPERATURE-DISPLACEMENT\n")
            f.write(f"{init_inc:.6e}, {cool_dt:.6e}, "
                    f"{min_inc:.6e}, {max_inc:.6e}\n")
            f.write("*CONTROLS, PARAMETERS=TIME INCREMENTATION\n")
            f.write("8,10,9,16,10,4,,5\n")
            f.write("0.25,0.5,0.75,0.85,,,1.5,\n")
            f.write("*CONTROLS, PARAMETERS=FIELD\n")
            f.write("0.05,0.01\n")

            write_all_persistent_bcs_new()

            # FILM on all currently active elements
            cumulative_activated.update(elem_ids)
            currently_active = (bp_elem_set | cumulative_activated) \
                               & valid_elem_ids
            write_film(currently_active)

            if is_output_step:
                f.write("*NODE FILE\n")
                f.write("NT, U\n")
                f.write("*EL FILE\n")
                f.write("S\n")
            else:
                f.write("*NODE FILE, FREQUENCY=0\n")
                f.write("*EL FILE, FREQUENCY=0\n")
            f.write("*END STEP\n\n")
            step_num += 1

    print(f"  Written {step_num - 1} steps to {output_path}")


# ================================================================
# MAIN DRIVER
# ================================================================

def renumber_elements_contiguous(elements, activation_groups,
                                 all_activated_elems,
                                 buildplate_eids, model_eids):
    """
    Renumber element IDs so that:
    - Build plate elements come FIRST: IDs 1 .. N_bp
    - Model elements follow: IDs (N_bp+1) .. (N_bp + N_model)

    This ensures CalculiX reads build plate elements before any
    sets or load cards reference them.
    """
    bp_sorted = sorted(buildplate_eids)
    model_sorted = sorted(model_eids)

    all_known = buildplate_eids | model_eids
    other_sorted = sorted(eid for eid in elements if eid not in all_known)

    ordered = bp_sorted + model_sorted + other_sorted

    old_to_new = {}
    for new_id, old_id in enumerate(ordered, start=1):
        old_to_new[old_id] = new_id

    new_elements = {}
    for old_id, conn in elements.items():
        if old_id in old_to_new:
            new_elements[old_to_new[old_id]] = conn

    new_groups = []
    for t, eids in activation_groups:
        new_eids = [old_to_new[e] for e in eids if e in old_to_new]
        if new_eids:
            new_groups.append((t, new_eids))

    new_activated = {old_to_new[e] for e in all_activated_elems
                     if e in old_to_new}
    new_bp = {old_to_new[e] for e in buildplate_eids
              if e in old_to_new}
    new_model = {old_to_new[e] for e in model_eids
                 if e in old_to_new}

    print(f"    Renumbered {len(old_to_new)} elements")
    print(f"    Build plate: 1-{len(bp_sorted)}")
    print(f"    Model: {len(bp_sorted) + 1}-"
          f"{len(bp_sorted) + len(model_sorted)}")

    return (new_elements, new_groups, new_activated,
            new_bp, new_model, old_to_new)


def renumber_nodes_contiguous(nodes, elements):
    """
    Renumber all node IDs to be contiguous starting from 1.
    Updates both the nodes dict and all element connectivity.

    Returns:
        new_nodes: dict with contiguous IDs
        new_elements: dict with updated connectivity
        old_to_new_nodes: mapping dict
    """
    old_ids_sorted = sorted(nodes.keys())
    old_to_new = {old_id: (i + 1)
                  for i, old_id in enumerate(old_ids_sorted)}

    new_nodes = {}
    for old_id, coords in nodes.items():
        new_nodes[old_to_new[old_id]] = coords

    new_elements = {}
    for eid, conn in elements.items():
        new_conn = [old_to_new[nid] for nid in conn]
        new_elements[eid] = new_conn

    print(f"    Renumbered {len(old_to_new)} nodes: "
          f"ID range {min(old_ids_sorted)}-{max(old_ids_sorted)} "
          f"-> 1-{len(old_ids_sorted)}")

    return new_nodes, new_elements, old_to_new


def generate_fdm_simulation(mesh_file, event_file, output_file,
                            nozzle_temp=200.0, bed_temp=25.0,
                            ambient_temp=25.0, convection_coeff=10.0,
                            coarsen_method='layer', coarsen_param=None,
                            search_radius_factor=1.5,
                            material_props=None, output_frequency=10):
    """Main pipeline."""

    print("=" * 60)
    print("CalculiX FDM 3D Printing Simulation Generator")
    print("=" * 60)

    # ---- [1] Parse mesh ----
    print(f"\n[1] Reading mesh: {mesh_file}")
    nodes, elements, elsets, nsets = parse_inp_mesh(mesh_file)
    print(f"    Nodes: {len(nodes)}")
    print(f"    Elements: {len(elements)}")
    print(f"    Element ID range: {min(elements.keys())} - "
          f"{max(elements.keys())}")
    print(f"    Node ID range: {min(nodes.keys())} - "
          f"{max(nodes.keys())}")

    # ---- [2] Compute centroids ----
    print(f"\n[2] Computing element geometry...")
    centroids = compute_element_centroids(nodes, elements)
    elem_size = compute_element_characteristic_size(nodes, elements)
    print(f"    Median element half-diagonal: {elem_size:.4f}")

    # ---- [3] Identify build plate ----
    print(f"\n[3] Identifying build plate (centroid Z < 0)...")
    buildplate_eids, model_eids = identify_build_plate(
        elements, centroids)
    print(f"    Build plate elements (Z < 0): {len(buildplate_eids)}")
    print(f"    Model elements (Z >= 0): {len(model_eids)}")

    if not buildplate_eids:
        print("    ERROR: No elements with centroid Z < 0!")
        sys.exit(1)

    # ---- [4] Parse events ----
    print(f"\n[4] Reading events: {event_file}")
    events = parse_event_series(event_file)
    extrusion_events = [e for e in events if e['extruding'] == 1]
    print(f"    Total events: {len(events)}")
    print(f"    Extrusion events: {len(extrusion_events)}")
    if events:
        print(f"    Time range: {events[0]['time']:.4f} - "
              f"{events[-1]['time']:.4f} s")

    # ---- [5] Assign events ----
    print(f"\n[5] Assigning events to model elements "
          f"(search_radius={search_radius_factor})...")
    activation_sequence, all_activated = assign_events_to_elements(
        events, centroids, elements, nodes,
        model_eids, search_radius_factor=search_radius_factor)

    bp_in_activated = all_activated & buildplate_eids
    if bp_in_activated:
        print(f"    Removing {len(bp_in_activated)} build plate elements")
        all_activated -= bp_in_activated
        activation_sequence = [(eid, t) for eid, t in activation_sequence
                               if eid not in bp_in_activated]

    print_activation_diagnostics(events, centroids, elements,
                                 model_eids, buildplate_eids,
                                 all_activated)

    # ---- [6] Coarsening ----
    print(f"[6] Coarsening: method='{coarsen_method}', "
          f"param={coarsen_param}")
    activation_groups = apply_coarsening(
        activation_sequence, elements, nodes,
        coarsen_method, coarsen_param)

    # ---- [7] Validate ----
    print(f"\n[7] Validating activation data...")
    all_activated = validate_activated_set(
        all_activated, elements, buildplate_eids)
    activation_groups = validate_activation_groups(
        activation_groups, elements, buildplate_eids)

    total_steps = 2 + 2 * len(activation_groups)
    print(f"    Final groups: {len(activation_groups)}")
    print(f"    Final activated elements: {len(all_activated)}")
    print(f"    Total CalculiX steps: {total_steps}")

    # ---- [8] Renumber for correct ordering ----
    print(f"\n[8] Renumbering IDs...")

    # Check node contiguity
    node_ids_sorted = sorted(nodes.keys())
    needs_node_renumber = (node_ids_sorted != list(
        range(1, len(node_ids_sorted) + 1)))

    if needs_node_renumber:
        nodes, elements, node_map = renumber_nodes_contiguous(
            nodes, elements)
        centroids = compute_element_centroids(nodes, elements)
    else:
        print(f"    Nodes already contiguous (1-{len(nodes)})")

    # Check element ordering: build plate must have LOWEST IDs
    # so CalculiX reads them before any sets reference high IDs
    bp_min = min(buildplate_eids) if buildplate_eids else 0
    bp_max = max(buildplate_eids) if buildplate_eids else 0
    model_min = min(model_eids) if model_eids else 0

    needs_elem_renumber = (bp_min > len(buildplate_eids) or
                           bp_max > model_min)

    if needs_elem_renumber:
        print(f"    Build plate IDs ({bp_min}-{bp_max}) are higher "
              f"than model IDs — renumbering to put build plate first")
        (elements, activation_groups, all_activated,
         buildplate_eids, model_eids, elem_map) = \
            renumber_elements_contiguous(
                elements, activation_groups, all_activated,
                buildplate_eids, model_eids)
    else:
        elem_ids_sorted = sorted(elements.keys())
        if elem_ids_sorted != list(range(1, len(elem_ids_sorted) + 1)):
            print(f"    Element IDs not contiguous — renumbering")
            (elements, activation_groups, all_activated,
             buildplate_eids, model_eids, elem_map) = \
                renumber_elements_contiguous(
                    elements, activation_groups, all_activated,
                    buildplate_eids, model_eids)
        else:
            print(f"    Elements already contiguous with build plate "
                  f"first (1-{len(elements)})")

    # ---- [9] Write ----
    print(f"\n[9] Writing: {output_file}")
    write_inp_file(
        output_path=output_file,
        nodes=nodes,
        elements=elements,
        activation_groups=activation_groups,
        all_activated_elems=all_activated,
        all_elem_ids=sorted(elements.keys()),
        buildplate_eids=buildplate_eids,
        model_eids=model_eids,
        nozzle_temp=nozzle_temp,
        bed_temp=bed_temp,
        ambient_temp=ambient_temp,
        convection_coeff=convection_coeff,
        material_props=material_props,
        output_frequency=output_frequency
    )

    print(f"\n{'=' * 60}")
    print(f"Run with: ccx -i {os.path.splitext(output_file)[0]}")
    print(f"{'=' * 60}")


# ================================================================
# EXAMPLE GENERATORS
# ================================================================

def create_example_mesh(filepath, nx=5, ny=5, nz=3,
                        dx=2.0, dy=2.0, dz=0.4, bp_layers=2):
    """Create a test mesh with build plate below Z=0."""
    with open(filepath, 'w') as f:
        f.write("** Example mesh with build plate below Z=0\n")
        f.write("*NODE\n")
        nid = 1
        node_map = {}
        total_z_layers = bp_layers + nz
        for k in range(total_z_layers + 1):
            z = (k - bp_layers) * dz  # BP layers are at Z < 0
            for j in range(ny + 1):
                for i in range(nx + 1):
                    f.write(f"{nid}, {i * dx:.6f}, "
                            f"{j * dy:.6f}, {z:.6f}\n")
                    node_map[(i, j, k)] = nid
                    nid += 1
        f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        eid = 1
        for k in range(total_z_layers):
            for j in range(ny):
                for i in range(nx):
                    n = [node_map[(i+di, j+dj, k+dk)]
                         for di, dj, dk in
                         [(0,0,0), (1,0,0), (1,1,0), (0,1,0),
                          (0,0,1), (1,0,1), (1,1,1), (0,1,1)]]
                    f.write(f"{eid}, {', '.join(map(str, n))}\n")
                    eid += 1
    bp_count = bp_layers * nx * ny
    model_count = nz * nx * ny
    print(f"Created: {filepath}")
    print(f"  Nodes: {nid-1}, Elements: {eid-1}")
    print(f"  Build plate (Z<0): {bp_count}, Model (Z>=0): {model_count}")


def create_example_events(filepath, nx=5, ny=5, nz=3,
                          dx=2.0, dy=2.0, dz=0.4, speed=5.0):
    """Create a raster-pattern event series (Z >= 0 only)."""
    events = []
    t = 0.0
    dt_move = dx / speed
    for k in range(nz):
        z = (k + 0.5) * dz  # Centroids at Z >= 0
        for j in range(ny):
            y = (j + 0.5) * dy
            x_range = (range(nx) if j % 2 == 0
                       else range(nx - 1, -1, -1))
            for i in x_range:
                x = (i + 0.5) * dx
                events.append((t, x, y, z, 1))
                t += dt_move
        t += 1.0
    with open(filepath, 'w') as f:
        f.write("time,x,y,z,extruding\n")
        for e in events:
            f.write(f"{e[0]:.6f},{e[1]:.6f},{e[2]:.6f},"
                    f"{e[3]:.6f},{e[4]}\n")
    print(f"Created: {filepath} ({len(events)} events, {t:.1f}s)")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate CalculiX .inp for FDM simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Build plate requirement:
  Your mesh MUST contain elements with centroids at Z < 0 to
  represent the build plate. These elements are never deactivated
  and are held at the bed temperature throughout the simulation.
  Model (printed) elements must have centroids at Z >= 0.

Coarsening methods:
  none        One step per unique activation time
  layer       One step per Z-layer [default]
  time        Fixed time window (--coarsen-param = seconds)
  multilayer  N layers per step (--coarsen-param = N)
  sectors     Layer + N intra-layer sectors (--coarsen-param = N)

Examples:
  %(prog)s --mesh mesh.inp --events events.csv
  %(prog)s --mesh mesh.inp --events events.csv --coarsen multilayer --coarsen-param 5
  %(prog)s --example --coarsen layer
        """)
    parser.add_argument("--mesh", type=str)
    parser.add_argument("--events", type=str)
    parser.add_argument("--output", type=str, default="fdm_simulation.inp")
    parser.add_argument("--nozzle-temp", type=float, default=200.0) # Adjusted for PHBV
    parser.add_argument("--bed-temp", type=float, default=25.0) # Adjusted for PHBV: cold bed
    parser.add_argument("--ambient-temp", type=float, default=25.0)
    parser.add_argument("--convection", type=float, default=10.0)
    parser.add_argument("--coarsen", type=str, default="layer",
                        choices=["none", "layer", "time",
                                 "multilayer", "sectors"])
    parser.add_argument("--coarsen-param", type=float, default=None)
    parser.add_argument("--search-radius", type=float, default=1.5)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--output-frequency", type=int, default=10,
                        help="Write results every N groups. "
                            "0=final only, 1=every step (default: 10)")

    args = parser.parse_args()

    if args.example:
        print("Generating example with build plate...\n")
        create_example_mesh("example_mesh.inp")
        create_example_events("example_events.csv")
        generate_fdm_simulation(
            mesh_file="example_mesh.inp",
            event_file="example_events.csv",
            output_file=args.output,
            nozzle_temp=args.nozzle_temp,
            bed_temp=args.bed_temp,
            ambient_temp=args.ambient_temp,
            convection_coeff=args.convection,
            coarsen_method=args.coarsen,
            coarsen_param=args.coarsen_param,
            search_radius_factor=args.search_radius,
            output_frequency=args.output_frequency
        )
    elif args.mesh and args.events:
        generate_fdm_simulation(
            mesh_file=args.mesh,
            event_file=args.events,
            output_file=args.output,
            nozzle_temp=args.nozzle_temp,
            bed_temp=args.bed_temp,
            ambient_temp=args.ambient_temp,
            convection_coeff=args.convection,
            coarsen_method=args.coarsen,
            coarsen_param=args.coarsen_param,
            search_radius_factor=args.search_radius,
            output_frequency=args.output_frequency
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
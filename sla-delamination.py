import zipfile
import csv
import os
import glob
import cv2
import numpy as np
from datetime import datetime

def analyze_sla_layers(sl1s_path, pixel_pitch_mm=0.049):
    """
    Extracts area and perimeter data from an SL1S file.
    
    :param sl1s_path: Path to the .sl1s file
    :param pixel_pitch_mm: The physical size of a single pixel on the LCD screen.
                           0.0498mm is a common value for 4K/8K monochrome screens.
                           Check your specific printer's XY resolution.
    """
    layer_data = []

    print(f"Opening {sl1s_path}...")
    
    # SL1S files are standard ZIP archives
    with zipfile.ZipFile(sl1s_path, 'r') as archive:
        # Filter for the layer PNG images (usually inside a folder or named sequentially)
        # We sort them to ensure they are processed from layer 0 upwards
        png_files = [f for f in archive.namelist() if f.endswith('.png')]
        png_files.sort()
        
        if not png_files:
            print("No PNG images found in the archive.")
            return layer_data

        print(f"Found {len(png_files)} layers. Processing...")

        for index, img_name in enumerate(png_files):
            # Extract the raw image bytes directly from memory without unzipping to disk
            with archive.open(img_name) as file:
                img_bytes = file.read()
                
            # Convert bytes to a numpy array, then decode into an OpenCV image (Grayscale)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Threshold the image to ensure it is strictly binary (0 or 255)
            # White pixels (255) represent the cured resin
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # --- 1. CALCULATE AREA ---
            # Count the number of white pixels and convert to mm^2
            pixel_area = np.count_nonzero(thresh)
            physical_area = pixel_area * (pixel_pitch_mm ** 2)
            
            # --- 2. CALCULATE PERIMETER ---
            # cv2.RETR_LIST retrieves ALL contours (outer boundaries AND inner holes)
            # This is critical for hollow prints, as inner holes add to the perimeter
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sum the arc length of all detected contours. 
            # True indicates that the contours are closed loops.
            pixel_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            physical_perimeter = pixel_perimeter * pixel_pitch_mm
            
            layer_data.append({
                'layer': index,
                'filename': img_name,
                'area_mm2': physical_area,
                'perimeter_mm': physical_perimeter,
                'shape_factor': (4 * np.pi * physical_area / (physical_perimeter ** 2)) if physical_perimeter > 0 else 0
            })

    print("Processing complete.")
    return layer_data


def evaluate_print_forces(layer_data, resin_params, printer_params):
    """
    Calculates peel force, tensile limit, and adhesion limits for each layer
    to identify high-risk failure points in an SLA print.
    """
    # Unpack printer parameters (Convert all to standard SI units: meters, seconds)
    h_dot = printer_params['lift_speed_mm_s'] * 1e-3     # Lift speed (m/s)
    h_eff = printer_params['effective_gap_mm'] * 1e-3    # Micro-gap before FEP yields (m)
    
    # Unpack resin parameters (SI units: Pascals)
    eta = resin_params['viscosity_cps'] * 1e-3           # Viscosity (Pa*s)
    uts = resin_params['uts_mpa'] * 1e6                  # Ultimate Tensile Strength (Pa)
    adh = resin_params['adhesion_mpa'] * 1e6             # Build plate adhesion strength (Pa)
    
    # Calculate the base area (Layer 0) to determine total build plate grip
    if not layer_data:
        return []
        
    base_area_m2 = layer_data[0]['area_mm2'] * 1e-6
    # F = Stress * Area
    f_adhesion_total = adh * base_area_m2
    
    # Pre-calculate the Stefan constant: K = (3 * eta * h_dot) / (2 * h^3)
    stefan_constant = (3 * eta * h_dot) / (2 * (h_eff ** 3))
    
    risk_report = []

    for data in layer_data:
        layer_idx = data['layer']
        A_m2 = data['area_mm2'] * 1e-6
        P_m = data['perimeter_mm'] * 1e-3
        shape_factor = data['shape_factor']
        
        # 1. CALCULATE PEEL FORCE (F_peel)
        if A_m2 == 0:
            f_peel = 0
        elif shape_factor > 0.8:
            # Mostly solid layer (approximated as a disk)
            f_peel = stefan_constant * (A_m2 ** 2) / np.pi
        else:
            # Hollow or highly complex layer
            # Effective wall thickness: w = A / P
            if P_m > 0:
                w_m = A_m2 / P_m
                f_peel = stefan_constant * A_m2 * (w_m ** 2)
            else:
                f_peel = 0
                
        # 2. CALCULATE STRUCTURAL INTEGRITY (F_tensile)
        # How much force can THIS specific layer withstand before snapping?
        f_tensile = uts * A_m2
        
        # 3. EVALUATE RISKS
        # Risk 1: The peel force is stronger than the resin's tensile strength (Support/Layer Snapping)
        layer_tear_risk = f_peel > f_tensile
        
        # Risk 2: The peel force is stronger than the build plate adhesion (Pancaking)
        pancake_risk = f_peel > f_adhesion_total
        
        # Calculate safety factors (Higher is better, < 1.0 means failure)
        sf_tear = f_tensile / f_peel if f_peel > 0 else float('inf')
        sf_plate = f_adhesion_total / f_peel if f_peel > 0 else float('inf')

        risk_report.append({
            'layer': layer_idx,
            'f_peel_N': f_peel,
            'f_tensile_N': f_tensile,
            'sf_tear': sf_tear,
            'sf_plate': sf_plate,
            'risk_tear': layer_tear_risk,
            'risk_pancake': pancake_risk
        })

    return risk_report, f_adhesion_total


def calculate_max_safe_lift_speed(layer_data, resin_params, printer_params,
                                   target_sf=2.0, speed_ceiling_mm_s=22.0):
    """
    Determines the maximum lift speed (mm/s) that keeps every layer of a model
    at or below the target safety factor, i.e. entirely within "LOW" risk.
 
    The Stefan squeezing-flow equation is linear in lift speed (h_dot):
 
        F_peel = K(h_dot) * f(geometry)
 
    where K = (3 * eta * h_dot) / (2 * h_eff³).
 
    Because F_peel ∝ h_dot, we can solve directly for the speed that makes
    the worst layer's safety factor exactly equal to `target_sf`, then cap it
    at `speed_ceiling_mm_s` to respect realistic printer limits.
 
    For each layer the binding constraint is the tighter of:
        • SF_tear  = F_tensile / F_peel  ≥ target_sf  → F_peel ≤ F_tensile / target_sf
        • SF_plate = F_adhesion / F_peel ≥ target_sf  → F_peel ≤ F_adhesion / target_sf
 
    Since F_peel = F_peel_at_unit_speed × h_dot, the per-layer speed limit is:
 
        h_dot_max = F_limit / F_peel_at_unit_speed
 
    The model's safe speed is the minimum across all layers.
 
    Parameters
    ----------
    layer_data : list[dict]
        Output of analyze_sla_layers().
    resin_params : dict
        Keys: viscosity_cps, uts_mpa, adhesion_mpa.
    printer_params : dict
        Keys: lift_speed_mm_s (used only for h_eff), effective_gap_mm.
    target_sf : float
        Desired minimum safety factor. 2.0 is recommended for reliable prints.
        Must be > 0.
    speed_ceiling_mm_s : float
        Hard upper bound on the returned speed (printer mechanical limit).
 
    Returns
    -------
    dict with keys:
        max_safe_speed_mm_s   – the recommended lift speed for this model
        limiting_layer        – layer index that sets the constraint
        limiting_constraint   – "tear" or "pancake"
        per_layer_limits      – list of dicts (layer, speed_limit_tear,
                                speed_limit_pancake, binding_speed_mm_s)
    """
    if not layer_data:
        return None
 
    if target_sf <= 0:
        raise ValueError("target_sf must be positive.")
 
    # ── SI conversions (h_dot factored out — set to 1 m/s as unit reference) ─
    h_eff_m = printer_params['effective_gap_mm'] * 1e-3
    eta     = resin_params['viscosity_cps']  * 1e-3   # Pa·s
    uts     = resin_params['uts_mpa']        * 1e6    # Pa
    adh     = resin_params['adhesion_mpa']   * 1e6    # Pa
 
    base_area_m2     = layer_data[0]['area_mm2'] * 1e-6
    f_adhesion_total = adh * base_area_m2
 
    # Stefan constant with h_dot = 1 m/s  →  scale F_peel linearly with speed
    stefan_unit = (3 * eta * 1.0) / (2 * (h_eff_m ** 3))
 
    per_layer_limits = []
    overall_min_speed = speed_ceiling_mm_s   # start at ceiling, pull down
    limiting_layer      = None
    limiting_constraint = None
 
    for data in layer_data:
        A_m2         = data['area_mm2']     * 1e-6
        P_m          = data['perimeter_mm'] * 1e-3
        shape_factor = data['shape_factor']
 
        if A_m2 == 0:
            per_layer_limits.append({
                'layer':                data['layer'],
                'speed_limit_tear_mm_s':    speed_ceiling_mm_s,
                'speed_limit_pancake_mm_s': speed_ceiling_mm_s,
                'binding_speed_mm_s':       speed_ceiling_mm_s,
            })
            continue
 
        # F_peel per unit h_dot (m/s) — same branching logic as evaluate_print_forces
        if shape_factor > 0.8:
            f_peel_unit = stefan_unit * (A_m2 ** 2) / np.pi
        else:
            w_m         = (A_m2 / P_m) if P_m > 0 else 0.0
            f_peel_unit = stefan_unit * A_m2 * (w_m ** 2)
 
        f_tensile = uts * A_m2
 
        # Maximum allowable F_peel to achieve target_sf for each constraint
        f_limit_tear    = f_tensile      / target_sf
        f_limit_pancake = f_adhesion_total / target_sf
 
        # h_dot (m/s) at which F_peel would exactly hit the limit
        # Convert to mm/s and clamp to [0, ceiling]
        def to_mm_s(f_limit):
            if f_peel_unit <= 0:
                return speed_ceiling_mm_s
            return min((f_limit / f_peel_unit) * 1e3, speed_ceiling_mm_s)
 
        spd_tear    = to_mm_s(f_limit_tear)
        spd_pancake = to_mm_s(f_limit_pancake)
        binding_spd = min(spd_tear, spd_pancake)
 
        per_layer_limits.append({
            'layer':                    data['layer'],
            'speed_limit_tear_mm_s':    round(spd_tear,    4),
            'speed_limit_pancake_mm_s': round(spd_pancake, 4),
            'binding_speed_mm_s':       round(binding_spd, 4),
        })
 
        if binding_spd < overall_min_speed:
            overall_min_speed   = binding_spd
            limiting_layer      = data['layer']
            limiting_constraint = 'tear' if spd_tear <= spd_pancake else 'pancake'
 
    return {
        'max_safe_speed_mm_s':   round(overall_min_speed, 4),
        'limiting_layer':        limiting_layer,
        'limiting_constraint':   limiting_constraint,
        'per_layer_limits':      per_layer_limits,
    }


def compile_delamination_report(
    sl1s_paths,
    output_csv,
    resin_params,
    printer_params,
    pixel_pitch_mm=0.049,
    target_sf=2.0,
    speed_ceiling_mm_s=22.0,
):
    """
    Processes one or more .sl1s files and writes a unified delamination-risk
    CSV that can be imported directly into Excel, Pandas, or any BI tool.
 
    Each row represents a single layer from a single file.
 
    Parameters
    ----------
    sl1s_paths : list[str] | str
        A single file path, a list of file paths, or a glob pattern such as
        ``"prints/*.sl1s"``.  All matched files are processed in alphabetical
        order.
    output_csv : str
        Destination path for the output CSV file.
    resin_params : dict
        Keys: ``viscosity_cps``, ``uts_mpa``, ``adhesion_mpa``.
    printer_params : dict
        Keys: ``lift_speed_mm_s``, ``effective_gap_mm``.
    pixel_pitch_mm : float
        Physical size of one screen pixel in millimetres.
        Defaults to 0.0498 mm (common 4K/8K mono screen).
    target_sf : float
        Safety factor threshold used by the lift speed calculator.
        Layers below this SF are considered MEDIUM/HIGH risk. Default: 2.0.
    speed_ceiling_mm_s : float
        Hard upper bound passed to the lift speed calculator (printer limit).
 
    Returns
    -------
    summary : list[dict]
        One summary dict per file with keys:
        ``file``, ``total_layers``, ``high_risk_layers``,
        ``max_peel_N``, ``min_sf_tear``, ``f_adhesion_N``,
        ``max_safe_speed_mm_s``, ``limiting_layer``, ``limiting_constraint``.
 
    CSV columns
    -----------
    file, layer, area_mm2, perimeter_mm, shape_factor,
    f_peel_N, f_tensile_N, sf_tear, sf_plate,
    risk_tear, risk_pancake, risk_level,
    speed_limit_tear_mm_s, speed_limit_pancake_mm_s, binding_speed_mm_s
    """
    # ── Resolve file list ────────────────────────────────────────────────────
    if isinstance(sl1s_paths, str):
        resolved = sorted(glob.glob(sl1s_paths)) or [sl1s_paths]
    else:
        resolved = sorted(sl1s_paths)
 
    if not resolved:
        raise FileNotFoundError(f"No files matched: {sl1s_paths}")
 
    # ── CSV field names ──────────────────────────────────────────────────────
    fieldnames = [
        'file',
        'layer',
        'area_mm2',
        'perimeter_mm',
        'shape_factor',
        'f_peel_N',
        'f_tensile_N',
        'sf_tear',
        'sf_plate',
        'risk_tear',
        'risk_pancake',
        'risk_level',               # "HIGH" | "MEDIUM" | "LOW"
        'speed_limit_tear_mm_s',    # max speed before tear failure on this layer
        'speed_limit_pancake_mm_s', # max speed before pancake failure on this layer
        'binding_speed_mm_s',       # tighter of the two — the per-layer safe speed
    ]
 
    summary    = []
    total_rows = 0
 
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
 
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
 
        for sl1s_path in resolved:
            filename = os.path.basename(sl1s_path)
            print(f"\n[{datetime.now():%H:%M:%S}] Processing: {filename}")
 
            if not os.path.isfile(sl1s_path):
                print(f"  [SKIP] File not found: {sl1s_path}")
                continue
 
            # ── Geometry extraction ──────────────────────────────────────────
            try:
                layer_data = analyze_sla_layers(sl1s_path, pixel_pitch_mm)
            except Exception as exc:
                print(f"  [ERROR] Could not read layers: {exc}")
                continue
 
            if not layer_data:
                print(f"  [SKIP] No usable layer data.")
                continue
 
            # ── Force / risk evaluation ──────────────────────────────────────
            try:
                risk_report, f_adhesion = evaluate_print_forces(
                    layer_data, resin_params, printer_params
                )
            except Exception as exc:
                print(f"  [ERROR] Force evaluation failed: {exc}")
                continue
 
            # ── Lift speed calculation ───────────────────────────────────────
            try:
                speed_result = calculate_max_safe_lift_speed(
                    layer_data, resin_params, printer_params,
                    target_sf=target_sf,
                    speed_ceiling_mm_s=speed_ceiling_mm_s,
                )
            except Exception as exc:
                print(f"  [WARN] Speed calculation failed: {exc}")
                speed_result = None
 
            # Index per-layer speed limits for fast lookup while writing rows
            speed_by_layer = {}
            if speed_result:
                speed_by_layer = {
                    entry['layer']: entry
                    for entry in speed_result['per_layer_limits']
                }
 
            # ── Write rows & accumulate per-file stats ───────────────────────
            high_risk_count = 0
            max_peel        = 0.0
            min_sf_tear     = float('inf')
 
            geo_by_layer = {d['layer']: d for d in layer_data}
 
            for r in risk_report:
                geo   = geo_by_layer.get(r['layer'], {})
                spds  = speed_by_layer.get(r['layer'], {})
 
                if r['risk_tear'] or r['risk_pancake']:
                    risk_level = 'HIGH'
                    high_risk_count += 1
                elif r['sf_tear'] < target_sf or r['sf_plate'] < target_sf:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
 
                max_peel = max(max_peel, r['f_peel_N'])
                if r['f_peel_N'] > 0:
                    min_sf_tear = min(min_sf_tear, r['sf_tear'])
 
                row = {
                    'file':                     filename,
                    'layer':                    r['layer'],
                    'area_mm2':                 round(geo.get('area_mm2',     0.0), 4),
                    'perimeter_mm':             round(geo.get('perimeter_mm', 0.0), 4),
                    'shape_factor':             round(geo.get('shape_factor', 0.0), 6),
                    'f_peel_N':                 round(r['f_peel_N'],    4),
                    'f_tensile_N':              round(r['f_tensile_N'], 4),
                    'sf_tear':                  round(r['sf_tear'],   4) if r['sf_tear']   != float('inf') else 'inf',
                    'sf_plate':                 round(r['sf_plate'],  4) if r['sf_plate']  != float('inf') else 'inf',
                    'risk_tear':                r['risk_tear'],
                    'risk_pancake':             r['risk_pancake'],
                    'risk_level':               risk_level,
                    'speed_limit_tear_mm_s':    spds.get('speed_limit_tear_mm_s',    ''),
                    'speed_limit_pancake_mm_s': spds.get('speed_limit_pancake_mm_s', ''),
                    'binding_speed_mm_s':       spds.get('binding_speed_mm_s',       ''),
                }
                writer.writerow(row)
                total_rows += 1
 
            max_safe_spd       = speed_result['max_safe_speed_mm_s'] if speed_result else None
            limiting_layer_idx = speed_result['limiting_layer']      if speed_result else None
            limiting_con       = speed_result['limiting_constraint']  if speed_result else None
 
            file_summary = {
                'file':                 filename,
                'total_layers':         len(risk_report),
                'high_risk_layers':     high_risk_count,
                'max_peel_N':           round(max_peel, 4),
                'min_sf_tear':          round(min_sf_tear, 4) if min_sf_tear != float('inf') else 'inf',
                'f_adhesion_N':         round(f_adhesion, 4),
                'max_safe_speed_mm_s':  max_safe_spd,
                'limiting_layer':       limiting_layer_idx,
                'limiting_constraint':  limiting_con,
            }
            summary.append(file_summary)
 
            print(f"  Layers processed        : {len(risk_report)}")
            print(f"  High-risk layers        : {high_risk_count}")
            print(f"  Max peel force          : {max_peel:.2f} N")
            print(f"  Min SF (tear)           : {min_sf_tear:.3f}")
            if max_safe_spd is not None:
                print(f"  Max safe lift speed     : {max_safe_spd:.4f} mm/s  "
                      f"(limited by layer {limiting_layer_idx}, {limiting_con})")
 
    print(f"\n✓ Done. {total_rows} rows written to: {output_csv}")
    return summary


# ── Resin properties ─────────────────────────────────────────────────────
resin_params = {
    'viscosity_cps': 711,   # centipoise  (standard resin at room temp)
    'uts_mpa':        57.3, # MPa         (Ultimate Tensile Strength)
    'adhesion_mpa':    5.0, # MPa         (estimated build-plate grip)
}

# ── Printer settings ─────────────────────────────────────────────────────
printer_params = {
    'lift_speed_mm_s': 1.0,   # mm/s  (= 60 mm/min)
    'effective_gap_mm': 0.05, # mm    (micro-gap before FEP film yields)
}

# ── Supply files in any of these three ways ───────────────────────────────
#   Option A: explicit list
# files = [
#     "prints/model_a.sl1s",
#     "prints/model_b.sl1s",
# ]

#   Option B: glob pattern (uncomment to use)
files = r"C:\Users\Samem\Downloads\3D-models\Research\UVTools-files\*.sl1s"

#   Option C: single file (uncomment to use)
# files = "my_print.sl1s"

summary = compile_delamination_report(
    sl1s_paths         = files,
    output_csv         = "output/delamination_report.csv",
    resin_params       = resin_params,
    printer_params     = printer_params,
    pixel_pitch_mm     = 0.049,  # Anycubic Mono X — adjust for your printer
    target_sf          = 2.0,    # safety factor that defines "LOW" risk
    speed_ceiling_mm_s = 22.0,   # your printer's maximum lift speed
)

# ── Print per-file summary table ─────────────────────────────────────────
col = f"{'File':<30} {'Layers':>7} {'High-Risk':>10} {'Max Peel (N)':>13} {'Min SF':>7} {'Safe Speed (mm/s)':>18} {'Limiting Layer':>15}"
print(f"\n{col}")
print("─" * len(col))
for s in summary:
    spd = f"{s['max_safe_speed_mm_s']:.4f}" if s['max_safe_speed_mm_s'] is not None else "n/a"
    lim = str(s['limiting_layer']) if s['limiting_layer'] is not None else "n/a"
    print(f"{s['file']:<30} {s['total_layers']:>7} {s['high_risk_layers']:>10} "
            f"{s['max_peel_N']:>13.2f} {str(s['min_sf_tear']):>7} {spd:>18} {lim:>15}")
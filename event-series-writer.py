from pyGCodeDecode import gcode_interpreter
# from pyGCodeDecode.plotter import plot_3d
from pyGCodeDecode.abaqus_file_generator import generate_abaqus_event_series
from pathlib import Path
import csv

setup = gcode_interpreter.setup(
    presets_file=r"C:\Users\Samem\Downloads\FDM-Research-Apps\pyGCodeDecode-main\pyGCodeDecode\data\default_printer_presets.yaml",
    printer="prusa_mini"
    )

gcode_folder = Path(r"C:\Users\Samem\Downloads\3D-models\Research\gcode-files")
gcode_paths = list(gcode_folder.glob("*.gcode"))

simulation_list = []
for path in gcode_paths:
    simulation = gcode_interpreter.simulation(
        gcode_path=path,
        initial_machine_setup=setup
        )
    simulation_list.append(simulation)

# for sim in simulation_list:
#     plot_3d(sim)

# Fill model_list with names of models from gcode_folder
model_list = []
for file in gcode_folder.iterdir():
    if file.is_file() and "-prusa" in file.name:
        prefix = file.name.split('-prusa', 1)[0]
        model_list.append(prefix)

# The main loop to generate event series files
i = 0
while i < len(simulation_list):
    generate_abaqus_event_series(
            simulation=simulation_list[i],
            filepath=rf"C:\Users\Samem\Downloads\3D-models\Research\Model-Data\FDM\event-series\event-series-{model_list[i]}.csv"
    )
    filepath = rf"C:\Users\Samem\Downloads\3D-models\Research\Model-Data\FDM\event-series\event-series-{model_list[i]}.csv"
    with open(filepath, "r") as infile:
        existing = infile.readlines()
    with open(filepath, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["time", "x", "y", "z", "extruding"])
        outfile.writelines(existing)
    i += 1

print("Done Completely.")
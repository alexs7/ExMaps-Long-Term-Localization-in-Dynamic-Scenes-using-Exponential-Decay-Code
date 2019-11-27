import os
import re
import subprocess
import tempfile

colmap_bin = "colmap/COLMAP.app/Contents/MacOS/colmap"

def override_ini_parameters(ini, params):
    if not params:
        return ini  # nothing to do

    for (key, value) in params.items():
        if '.' in key:  # Update a parameter in a section.
            section, setting = key.split('.')
            section_text_old = re.findall(r'(\[%s\][^\[]+)' % section, ini, re.M)[0]
            section_text_new = re.sub(setting + '=.*', setting + '=' + str(value), section_text_old)
            ini = ini.replace(section_text_old, section_text_new)
        elif key + '=' in ini:  # Update an existing parameter
            ini = re.sub(key + '=.*', key + '=' + str(value), ini)
        else:  # Add a parameter.
            ini = key + '=' + str(value) + '\n' + ini

    return ini

def save_ini(contents, ini_save_path):
    # If there is no path given, generate a temporary one.
    if ini_save_path is None:
        with tempfile.NamedTemporaryFile() as f:
            ini_save_path = f.name

    # Write contents to ini file.
    with open(ini_save_path, 'w') as f:
        f.write(contents)

    return ini_save_path

def run(database_path, image_path, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_feature_extraction.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    colmap_ini = override_ini_parameters(colmap_ini, {
        'database_path': database_path.replace('\\', '/'),
        'image_path': image_path.replace('\\', '/')
    })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "feature_extractor", "--project_path", ini_save_path]

    subprocess.check_call(colmap_command)
import os
import re
import subprocess
import tempfile

colmap_bin = "colmap"

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

def feature_extractor_arcore(database_path, image_path, image_list_path=None, ini_save_path=None, params=None, query=False):
    # Find and read template INI.
    if(not query):
        input_ini_file = os.path.join('template_inis', 'ar_core_feature_extraction', 'colmap_feature_extraction.ini')
    else:
        input_ini_file = os.path.join('template_inis', 'ar_core_feature_extraction', 'colmap_feature_extraction_query.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    if(image_list_path == None):
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'image_path': image_path.replace('\\', '/')
        })
    else:
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'image_path': image_path.replace('\\', '/'),
            'image_list_path': image_list_path.replace('\\', '/')
        })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "feature_extractor", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def feature_extractor_cmu(database_path, image_path, image_list_path=None, ini_save_path=None, params=None, query=False):
    # Find and read template INI.
    if(not query):
        input_ini_file = os.path.join('template_inis', 'cmu_feature_extraction', 'colmap_feature_extraction.ini')
    else:
        input_ini_file = os.path.join('template_inis', 'cmu_feature_extraction', 'colmap_feature_extraction_query.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    if(image_list_path == None):
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'image_path': image_path.replace('\\', '/')
        })
    else:
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'image_path': image_path.replace('\\', '/'),
            'image_list_path': image_list_path.replace('\\', '/')
        })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "feature_extractor", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def feature_extractor_lamar(database_path, image_path, cam_id, feature_ini_dir, image_list_path=None, ini_save_path=None, params=None):
    # Find and read template INI.
    input_ini_file = os.path.join(feature_ini_dir, f'colmap_feature_extraction_{cam_id}.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    if(image_list_path == None):
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'image_path': image_path.replace('\\', '/')
        })
    else:
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'image_path': image_path.replace('\\', '/'),
            'image_list_path': image_list_path.replace('\\', '/')
        })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "feature_extractor", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def vocab_tree_matcher(database_path, match_list_path=None, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_vocab_tree_matcher.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    if(match_list_path == None):
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/')
        })
    else:
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
            'VocabTreeMatching.match_list_path': match_list_path.replace('\\', '/')
        })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "vocab_tree_matcher", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def exhaustive_matcher(database_path, match_list_path=None, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_exhaustive_matcher.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    if(match_list_path == None):
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/')
        })
    else:
        colmap_ini = override_ini_parameters(colmap_ini, {
            'database_path': database_path.replace('\\', '/'),
        })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "exhaustive_matcher", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def mapper(database_path, image_path, output_path, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_mapper.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    colmap_ini = override_ini_parameters(colmap_ini, {
        'database_path': database_path.replace('\\', '/'),
        'image_path': image_path.replace('\\', '/'),
        'output_path': output_path.replace('\\', '/'),
    })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "mapper", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def image_registrator(database_path, input_path, output_path, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_image_registrator.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    colmap_ini = override_ini_parameters(colmap_ini, {
        'database_path': database_path.replace('\\', '/'),
        'input_path': input_path.replace('\\', '/'),
        'output_path': output_path.replace('\\', '/')
    })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "image_registrator", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def model_converter(database_path, input_path, output_path, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_model_converter.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    colmap_ini = override_ini_parameters(colmap_ini, {
        'input_path': input_path.replace('\\', '/'),
        'output_path': output_path.replace('\\', '/'),
        'output_type': "TXT"
    })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "model_converter", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)

def model_aligner(path_to_model, path_to_geo_registered_model, path_to_text_file = None):
    assert(path_to_text_file != None)
    colmap_command = [colmap_bin, "model_aligner", "--input_path", path_to_model,
                                                    "--output_path", path_to_geo_registered_model,
                                                    "--ref_images_path", path_to_text_file,
                                                    "--robust_alignment", "1",
                                                    "--robust_alignment_max_error", "0.2"]
    # Call COLMAP.
    subprocess.check_call(colmap_command)

def point_triangulator(database_path, image_path, input_path, output_path, ini_save_path=None, params=None):

    # Find and read template INI.
    input_ini_file = os.path.join('template_inis', 'colmap_point_triangulator.ini')
    with open(input_ini_file, 'r') as f:
        colmap_ini = f.read()

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Update some parameters, if requested.
    colmap_ini = override_ini_parameters(colmap_ini, params)

    # Add database and image directory to the INI.
    # NB. Forward slashes in paths are expected by the GUI.
    colmap_ini = override_ini_parameters(colmap_ini, {
        'database_path': database_path.replace('\\', '/'),
        'image_path': image_path.replace('\\', '/'),
        'input_path': input_path.replace('\\', '/'),
        'output_path': output_path.replace('\\', '/'),
    })

    # Save INI file.
    ini_save_path = save_ini(colmap_ini, ini_save_path)

    # Call COLMAP.
    colmap_command = [colmap_bin, "point_triangulator", "--project_path", ini_save_path]
    print(f"Running: {colmap_command}")
    subprocess.check_call(colmap_command)


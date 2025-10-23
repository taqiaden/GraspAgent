import os
from collections import OrderedDict
from configparser import ConfigParser

# from filelock import FileLock

# lock = FileLock("file.lock")

config_file_path='Configurations/config.ini'
counters_file_path='Configurations/counters.ini'

config_folder='Configurations/'

def creat_section(section,config_file=config_file_path):
    config = ConfigParser()

    config.read(config_file)
    config.add_section(section)

    with open(config_file, 'w') as f:
        config.write(f)

def check(section,key,config_file,default='0'):
    config = ConfigParser()

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            config.write(f)  # writes an empty file
    else: config.read(config_file)

    if not config.has_section(section):
        config.add_section(section)
        with open(config_file, 'w') as f:
            config.write(f)
    if not config.has_option(section,key):
        save_key(key,default,section,config_file)

def save_key(key,value,section='main',config_file=config_file_path):
    try:
        config = ConfigParser()

        if not isinstance(value,str): value=str(value)
        config.read(config_file)
        config.set(section, key, value)
        with open(config_file, 'w') as f:
            config.write(f)
    except Exception as e:
        print(str(e))
        remove_duplicates_config(config_file,config_file)

def get_value(key,section='main',config_file=config_file_path):
    config = ConfigParser()

    config.read(config_file)
    check(section,key,config_file)
    return config.get(section, key)

def get_float(key,section='main',config_file=config_file_path,default=0.):
    # try:
        check(section,key,config_file,default=str(default))
        config = ConfigParser()
        config.read(config_file)
        return config.getfloat(section, key)
    # except Exception as e:
    #     print('dynamic_config get_float ,',str(e))
    #     remove_duplicates_config(config_file,config_file)
    #     return float(default)

def get_int(key,section='main',config_file=config_file_path):
    config = ConfigParser()

    config.read(config_file)
    return config.getint(section, key)

def add_to_value_(key,delta,section='main'):
    old_value=get_float(key,section)
    new_value=str(delta+old_value)
    save_key(key,new_value,section)

def add_to_value(key,delta,section='main',lock_other_process=True):
    if lock_other_process:
        with lock:
            add_to_value_(key,delta,section)
    else: add_to_value_(key,delta,section)

def remove_duplicates_config(input_file, output_file):
    config = ConfigParser(strict=False)
    config._sections = OrderedDict()  # Preserve order

    # Read raw lines to remove duplicates manually
    seen = {}
    output_lines = []
    current_section = None

    with open(input_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                current_section = stripped
                seen[current_section] = set()
                output_lines.append(line)
            elif '=' in line and current_section:
                key = line.split('=')[0].strip()
                if key not in seen[current_section]:
                    seen[current_section].add(key)
                    output_lines.append(line)
                else:
                    # Duplicate detected — skip this line
                    pass
            else:
                output_lines.append(line)

    # Write cleaned config
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    check('section', 'key', 'test_config', default=str(0.))
    check('section', 'key', 'test_config2', default=str(0.))

    # get_float('test_config', config_file='test_config')
    # save_key("collision_times", 0.0, section='Grasp_GAN')
    # save_key("out_of_scope_times", 0.0, section='Grasp_GAN')
    # save_key("good_firmness_times", 0.0, section='Grasp_GAN')
    # save_key("C_running_loss", 0.0, section='Grasp_GAN')
    # save_key("G_running_loss", 0.0, section='Grasp_GAN')

    # add_to_value("performance_indicator",0.1, section='Grasp_GAN')
    # creat_section('Grasp_GAN')
    # v=get_value("performance_indicator", section='Grasp_GAN')
    # print(v)
    # save_key("performance_indicator", 0.1, section='Grasp_GAN')

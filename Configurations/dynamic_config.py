from configparser import ConfigParser
from filelock import FileLock
lock = FileLock("file.lock")
config = ConfigParser()

config_file_path='Configurations/config.ini'
counters_file_path='Configurations/counters.ini'

def creat_section(section,config_file=config_file_path):
    config.read(config_file)
    config.add_section(section)

    with open(config_file, 'w') as f:
        config.write(f)

def check(section,key,config_file,default='0'):
    if not config.has_section(section):
        config.add_section(section)
    if not config.has_option(section,key):
        save_key(key,default,section,config_file)

def save_key(key,value,section='main',config_file=config_file_path):
    if not isinstance(value,str): value=str(value)
    config.read(config_file)
    config.set(section, key, value)

    with open(config_file, 'w') as f:
        config.write(f)
def get_value(key,section='main',config_file=config_file_path):
    config.read(config_file)
    check(section,key,config_file)
    return config.get(section, key)

def get_float(key,section='main',config_file=config_file_path):
    config.read(config_file)
    check(section,key,config_file)
    return config.getfloat(section, key)

def get_int(key,section='main',config_file=config_file_path):
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

if __name__ == "__main__":
    save_key("collision_times", 0.0, section='Grasp_GAN')
    save_key("out_of_scope_times", 0.0, section='Grasp_GAN')
    save_key("good_firmness_times", 0.0, section='Grasp_GAN')
    save_key("C_running_loss", 0.0, section='Grasp_GAN')
    save_key("G_running_loss", 0.0, section='Grasp_GAN')

    # add_to_value("performance_indicator",0.1, section='Grasp_GAN')
    # creat_section('Grasp_GAN')
    # v=get_value("performance_indicator", section='Grasp_GAN')
    # print(v)
    # save_key("performance_indicator", 0.1, section='Grasp_GAN')

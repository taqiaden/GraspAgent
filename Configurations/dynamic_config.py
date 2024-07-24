from configparser import ConfigParser
config = ConfigParser()

config_file_path='config.ini'

def creat_section(section):
    config.read(config_file_path)
    config.add_section(section)

    with open('config.ini', 'w') as f:
        config.write(f)
def save_key(key,value,section='main'):
    if not isinstance(value,str): value=str(value)
    config.read(config_file_path)
    config.set(section, key, value)

    with open(config_file_path, 'w') as f:
        config.write(f)
def get_value(key,section='main'):
    config.read(config_file_path)
    return config.get(section, key)

def get_float(key,section='main'):
    config.read(config_file_path)
    return config.getfloat(section, key)


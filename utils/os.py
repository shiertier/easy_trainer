from os import environ


__all__ = ['get_env', 'set_env', 'get_var']

def get_env(var_name, default_value=None):
    value = environ.get(var_name, default_value)
    if value is None:
        return None
    if ',' in value:
        return convert_str_to_list(value)
    else:
        return environ.get(var_name, default_value)

def set_env(var_name, value):
    if isinstance(value, list):
        value = convert_list_to_str(value)
    environ[var_name] = value

def get_var(var, config_value):
    env_var = get_env(var)
    if not env_var:
        return config_value

def convert_str_to_list(s):
    return [x.strip() for x in s.split(',')]

def convert_list_to_str(l):
    return ','.join(l)
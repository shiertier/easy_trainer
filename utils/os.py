from os import environ


__all__ = ['get_env', 'set_env', 'get_var']

def get_env(var_name, default_value=None):
    return environ.get(var_name, default_value)

def set_env(var_name, value):
    environ[var_name] = value

def get_var(var, config_value):
    env_var = get_env(var)
    if not env_var:
        return config_value

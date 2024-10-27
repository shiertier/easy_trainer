
__all__ = ["none_or_type"]

def none_or_type(value, desired_type):
    if value == "None":
        return None
    return desired_type(value)
__all__ = ["none_or_type", 
           "trim_off_prompt"
           ]

def none_or_type(value, desired_type):
    if value == "None":
        return None
    return desired_type(value)

def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
    # Trim off the prompt
    while True:
        try:
            i = input_ids.index(eoh_id)
        except ValueError:
            break
        
        input_ids = input_ids[i + 1:]
    
    # Trim off the end
    try:
        i = input_ids.index(eot_id)
    except ValueError:
        return input_ids
    
    return input_ids[:i]
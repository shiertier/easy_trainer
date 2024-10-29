import gettext
import os
from ..config import *

__all__ = ['i18n']

class Translater:
    def __init__(self, 
                 language_str: str, 
                 locales_dir: str):
        self.translation = gettext.translation('messages', locales_dir, languages=[language_str], fallback=True)
        self.translation.install()

    def translate(self, 
                  input: str,
                  replace_dict: dict = None):
        if replace_dict is None:
            return self.translation.gettext(input)
        else:
            result = self.translation.gettext(input)
            for key, value in replace_dict.items():
                result = result.replace(key, str(value))
            return result

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
locales_dir = os.path.join(project_root, 'locales')

if LANGUAGE_STR not in LANGUAGE_AVILIABLE:
    raise ValueError(f"BASE_LANGUAGE_STR {LANGUAGE_STR} is not in LANGUAGE_AVILIABLE {LANGUAGE_AVILIABLE}")

translater = Translater(LANGUAGE_STR, locales_dir)

i18n = translater.translate
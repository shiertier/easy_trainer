from os import O_WRONLY, O_CREAT, O_EXCL
from os import write as os_write
from os import close as os_close
from os import open as os_open
from pathlib import Path

def write_caption(image_path: Path, caption: str):
    caption_path = image_path.with_suffix(".txt")

    try:
        f = os_open(caption_path, O_WRONLY | O_CREAT | O_EXCL)  # Write-only, create if not exist, fail if exists
    except FileExistsError:
        # logging.warning(f"Caption file '{caption_path}' already exists")
        return
    except Exception as e:
        # logging.error(f"Failed to open caption file '{caption_path}': {e}")
        return
    
    try:
        os_write(f, caption.encode("utf-8"))
        os_close(f)
    except Exception as e:
        # logging.error(f"Failed to write caption to '{caption_path}': {e}")
        return
from ..utils import *
from huggingface_hub import snapshot_download

__all__ = ['download_huggingface_model', 'convert_huggingface_url']

def download_huggingface_model(url_or_repo, repo_type='repo', local_dir='./huggingface_models'):
    repo_name,file_path = convert_huggingface_url(url_or_repo)

    if repo_type == "repo":
        snapshot_download(repo_name, local_dir=local_dir, force_download=True)

    elif repo_type == "file":
        snapshot_download(repo_name, local_dir=local_dir, filename=file_path, force_download=True)
    else:
        raise ValueError(i18n("repo_type must be 'repo' or 'file'"))

def convert_huggingface_url(url):
    url.replace('hf-mirror.com', 'huggingface.co')
    if url.count('/') == 1:
        return url, None
    elif url.startswith('https://huggingface.co') and url.count('/') == 4:
        return url.splite('.co/')[1] , None
    elif url.startswith('https://huggingface.co') and url.count('/') > 4:
        url = url.replace('?download=true', '')
        url_body = url.split('.co/')[1]
        if '/resolve/main' in url_body:
            repo, file_path = url_body.split('/resolve/main')
        elif '/blob/main' in url_body:
            repo, file_path = url_body.split('/blob/main')
        else:
            raise ValueError(i18n("url is not valid"))
        return repo, file_path
    else:
        raise ValueError(i18n("url is not valid"))

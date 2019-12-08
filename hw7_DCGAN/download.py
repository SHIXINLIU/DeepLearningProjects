from __future__ import print_function
import os
import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={ 'id': id }, stream=True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)


def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None


def save_response_content(response, destination, chunk_size=32*1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
              unit='B', unit_scale=True, desc=destination):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)


def download_celeb_a(dirpath):
  filename, drive_id  = "celebA.tar.gz", "1gWqvG6E7VXGi257Y-K2tGKBsKaIAuBR2"
  save_path = os.path.join(dirpath, filename)

  if os.path.exists(save_path):
    print('[*] {} already exists'.format(save_path))
  else:
    download_file_from_google_drive(drive_id, save_path)


def prepare_data_dir(path = './data'):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    prepare_data_dir()
    download_celeb_a('./data')

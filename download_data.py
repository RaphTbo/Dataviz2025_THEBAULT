
import requests, os

# NOTE: Replace the placeholder URLs with the real "Télécharger" links from the data.gouv resource page.
URLS = {
    "reg_ara": "https://www.data.gouv.fr/fr/datasets/r/5b3c2cee-44b7-48bd-b4e8-439a03ff6cd2",
    "reg_pac": "https://www.data.gouv.fr/fr/datasets/r/83a1f131-9e23-4c3b-b1c6-e58f33fe7b80",
    "reg_occ": "https://www.data.gouv.fr/fr/datasets/r/0c463ef6-c00a-48e2-b50a-d17cfe998b84",
    "place": "https://www.data.gouv.fr/fr/datasets/r/cf247ad9-5bcd-4c8a-8f4d-f49f0803bca1",
    "product": "https://www.data.gouv.fr/fr/datasets/r/0254ad40-3d26-472e-a628-bd9233f22ee0"
}


OUT = "data"
os.makedirs(OUT, exist_ok=True)

def download(url, outpath):
    print(f'Downloading {url} -> {outpath}')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(outpath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

if __name__ == '__main__':
    for k, url in URLS.items():
        out = os.path.join(OUT, f'{k}.csv')
        try:
            download(url, out)
        except Exception as e:
            print('Failed to download', url, e)
            print('Edit download_data.py to set the correct data.gouv.fr resource URLs for each CSV (open the dataset page and use the "Télécharger" button -> copy link).')

"""A script to download the KiTS23 dataset into this repository"""
import os.path
import sys
from tqdm import tqdm
from pathlib import Path
import urllib.request
import shutil
from time import sleep

from pangteen import config, utils
from pangteen.kits import TRAINING_CASE_NUMBERS

def get_destination(dst_folder, case_id: str, create: bool = False) -> Path:
    destination = os.path.join(dst_folder, case_id, "imaging.nii.gz")
    if create:
        utils.maybe_mkdir(os.path.join(dst_folder, case_id))
    return Path(destination)


def cleanup(tmp_pth: Path, e: Exception):
    if tmp_pth.exists():
        tmp_pth.unlink()

    if e is None:
        print("\nInterrupted.\n")
        sys.exit()
    raise(e)


def download_case(dst_folder, case_num: int, pbar: tqdm, retry=True):
    remote_name = f"master_{case_num:05d}.nii.gz"
    url = f"https://kits19.sfo2.digitaloceanspaces.com/{remote_name}"
    destination = get_destination(dst_folder, f"case_{case_num:05d}", True)
    tmp_pth = destination.parent / f".partial.{destination.name}"
    try:
        urllib.request.urlretrieve(url, str(tmp_pth))
        shutil.move(str(tmp_pth), str(destination))
    except KeyboardInterrupt as e:
        pbar.close()
        while True:
            try:
                sleep(0.1)
                cleanup(tmp_pth, None)
            except KeyboardInterrupt:
                pass
    except Exception as e:
        if retry:
            print(f"\nFailed to download case_{case_num:05d}. Retrying...")
            sleep(5)
            download_case(dst_folder, case_num, pbar, retry=False)
        pbar.close()
        while True:
            try:
                cleanup(tmp_pth, e)
            except KeyboardInterrupt:
                pass


def download_dataset(dst_folder):
    # Make output directory if it doesn't exist already
    utils.maybe_mkdir(dst_folder)

    # Determine which cases still need to be downloaded
    left_to_download = []
    for case_num in TRAINING_CASE_NUMBERS:
        case_id = f"case_{case_num:05d}"
        dst = get_destination(dst_folder, case_id)
        if not dst.exists():
            left_to_download = left_to_download + [case_num]

    # Show progressbar as cases are downloaded
    print(f"\nFound {len(left_to_download)} cases to download\n")
    for case_num in (pbar := tqdm(left_to_download)):
        pbar.set_description(f"Dowloading case_{case_num:05d}...")
        download_case(dst_folder, case_num, pbar)


if __name__ == "__main__":
    '''
    python -u pangteen/kits/download_kits2023.py
    '''
    download_dataset(config.kits_dataset_folder)
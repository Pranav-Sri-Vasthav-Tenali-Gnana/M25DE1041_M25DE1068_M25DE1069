import requests
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from config import DATA_DIR, SPEAKERS, DIGITS, REPS, BASE_URL, N_PER_DIGIT


def download_fsdd(n_per_digit=N_PER_DIGIT, speakers=None):
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    if speakers is None:
        speakers = SPEAKERS

    dataset = []
    total = len(DIGITS) * len(speakers) * n_per_digit

    with tqdm(total=total, desc="Downloading FSDD") as pbar:
        for digit in DIGITS:
            for speaker in speakers:
                count = 0
                for rep in REPS:
                    if count >= n_per_digit:
                        break
                    fname = f"{digit}_{speaker}_{rep}.wav"
                    fpath = data_dir / fname
                    if not fpath.exists():
                        url = BASE_URL + fname
                        try:
                            r = requests.get(url, timeout=10)
                            if r.status_code == 200:
                                fpath.write_bytes(r.content)
                            else:
                                pbar.update(1)
                                continue
                        except Exception:
                            pbar.update(1)
                            continue
                    dataset.append((str(fpath), digit))
                    count += 1
                    pbar.update(1)

    return dataset


def get_label_counts(dataset):
    return Counter(label for _, label in dataset)


if __name__ == "__main__":
    dataset = download_fsdd()
    label_counts = get_label_counts(dataset)
    print(f"Dataset ready — {len(dataset)} audio files across {len(DIGITS)} classes.")
    print("Samples per digit:", dict(sorted(label_counts.items())))

import os
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_imdb_directory(
    data_dir: str,
    export_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load IMDB review data from directory structure (pos/ and neg/ folders).
    """

    pos_dir = os.path.join(data_dir, "pos")
    neg_dir = os.path.join(data_dir, "neg")

    if not os.path.isdir(pos_dir) or not os.path.isdir(neg_dir):
        logger.error(f"Missing 'pos' or 'neg' directories in {data_dir}")
        raise FileNotFoundError(
            f"Both 'pos' and 'neg' directories must exist in {data_dir}"
        )

    def load_reviews_from_dir(directory: str, label: int) -> list[dict]:
        reviews = []
        skipped_files = 0

        for fname in os.listdir(directory):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if not text:
                    skipped_files += 1
                    continue

                reviews.append({
                    "review": text,
                    "label": label
                })

            except Exception as e:
                skipped_files += 1
                logger.warning(f"Failed to read {fpath}: {e}")

        if skipped_files > 0:
            logger.info(
                f"Skipped {skipped_files} files in '{directory}' due to read/validation issues."
            )

        return reviews  # ✅ THIS WAS MISSING

    # Load reviews
    pos_reviews = load_reviews_from_dir(pos_dir, 1)
    neg_reviews = load_reviews_from_dir(neg_dir, 0)

    if not pos_reviews or not neg_reviews:
        logger.error("No reviews found in one or both class folders.")
        raise ValueError("No reviews found in 'pos' or 'neg' directories.")

    # Create DataFrame
    df = pd.DataFrame(pos_reviews + neg_reviews)

    if not {"review", "label"}.issubset(df.columns):
        raise ValueError("Schema invalid: DataFrame must have ['review', 'label'] columns.")

    logger.info(
        f"Loaded {len(df)} reviews "
        f"({len(pos_reviews)} positive, {len(neg_reviews)} negative)"
    )

    if export_csv_path:
        df.to_csv(export_csv_path, index=False, encoding="utf-8")
        logger.info(f"Exported DataFrame to {export_csv_path}")

    return df

"""One-time script to process initial batch of PDFs through the pipeline.

Expects PDFs in gs://{BUCKET_NAME}/uploads/. Runs the full pipeline:
Layout Parser -> page rendering -> embedding -> index update.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.processor import process_all_new


def main():
    print("Starting initial PDF batch processing...")
    start = time.time()

    results = process_all_new()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print(f"\nProcessing complete in {minutes}m {seconds:.1f}s")
    if isinstance(results, dict):
        print(f"  Documents processed: {results.get('documents_processed', 0)}")
        print(f"  Text chunks created: {results.get('total_chunks', 0)}")
        print(f"  Page images created: {results.get('total_pages', 0)}")
    else:
        print("  No new PDFs found in uploads/")


if __name__ == "__main__":
    main()

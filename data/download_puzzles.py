#!/usr/bin/env python3

import os
import requests
import argparse
from urllib.parse import urljoin


def download_sourcemap(js_url):
    """
    Downloads the JS file and extracts the sourcemap URL from it.
    Then downloads and returns the sourcemap content.
    """
    # Download the main JS file
    response = requests.get(js_url)
    response.raise_for_status()
    js_content = response.text

    # Find the sourceMappingURL comment at the end of the file
    sourcemap_marker = "//# sourceMappingURL="
    if sourcemap_marker not in js_content:
        raise ValueError("No sourcemap reference found in JS file")

    # Extract sourcemap URL
    sourcemap_line = [
        line for line in js_content.splitlines() if sourcemap_marker in line
    ][-1]
    sourcemap_path = sourcemap_line.replace(sourcemap_marker, "").strip()

    # Construct absolute URL for sourcemap
    sourcemap_url = urljoin(js_url, sourcemap_path)

    # Download the sourcemap
    response = requests.get(sourcemap_url)
    response.raise_for_status()
    return response.json()


def extract_puzzles_content(sourcemap_json):
    """
    Extracts the puzzles/gtg_puzzles.js content from the sourcemap.
    """
    target_file = "puzzles/gtg_puzzles.js"
    try:
        file_index = sourcemap_json["sources"].index(target_file)
        return sourcemap_json["sourcesContent"][file_index]
    except (ValueError, KeyError, IndexError) as e:
        raise ValueError(f"Could not find {target_file} in sourcemap: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract puzzle data from sourcemap"
    )
    parser.add_argument(
        "--url",
        default="https://guessthe.game/static/js/main.973b6669.js",
        help="URL of the main JS file",
    )
    parser.add_argument(
        "--output",
        default="raw/20250329/gtg_puzzles.ts",
        help="Output path for the extracted puzzle data",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        print(f"Downloading and processing sourcemap from {args.url}")
        sourcemap_json = download_sourcemap(args.url)

        print("Extracting puzzle data")
        puzzle_content = extract_puzzles_content(sourcemap_json)

        print(f"Writing output to {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(puzzle_content)

        print("Done!")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {str(e)}")
        exit(1)
    except ValueError as e:
        print(f"Error processing data: {str(e)}")
        exit(1)
    except IOError as e:
        print(f"Error writing output file: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()

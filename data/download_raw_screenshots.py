#!/usr/bin/env python3

import os
import requests
import argparse
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urljoin
import logging
import time  # <-- Import time for sleep
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
BASE_URL = "https://guessthe.game/games/"
DEFAULT_START_ID = 1
DEFAULT_END_ID = 1070  # Adjust as needed
DEFAULT_CONCURRENCY = 10
USER_AGENT = "GuesstheGameDownloader/1.2 (Atomic, Concurrent, Retry)"  # Identify script

# --- Retry Configuration ---
MAX_RETRIES = 3  # Number of retries on failure (5xx or connection errors)
RETRY_DELAY_SECONDS = (
    5  # Seconds to wait between retries (used for backoff calculation)
)
RETRY_STATUS_CODES = [500, 502, 503, 504]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

# --- Helper Functions ---


def get_final_path(base_dir, game_id, seq, ext):
    """Constructs the final save path for a file."""
    return os.path.join(base_dir, str(game_id), f"{seq}.{ext}")


def download_file(url, final_path, session, verbose_log):
    """
    Downloads a single file from a URL to a final path atomically,
    handling redirects as 404s. Retries for connection errors and specific
    server errors (5xx) are handled by the session adapter.

    Args:
        url (str): The URL to download from.
        final_path (str): The desired final path for the file.
        session (requests.Session): The configured requests session with retries.
        verbose_log (bool): Flag for detailed logging.

    Returns:
        tuple: (status, message) where status is one of:
               'skipped': File already exists.
               'downloaded': File downloaded successfully.
               'not_found': URL returned 404, redirect, or HTML.
               'failed': Download failed after session retries or other error.
               message contains details.
    """
    if os.path.exists(final_path):
        return "skipped", f"Skipped: {final_path} already exists."

    target_dir = os.path.dirname(final_path)
    os.makedirs(target_dir, exist_ok=True)

    tmp_file_handle = None
    tmp_name = None
    response = None

    try:
        headers = {"User-Agent": USER_AGENT}
        # Make request WITHOUT following redirects
        # Retries for specific errors are handled by the session adapter
        response = session.get(
            url,
            stream=True,
            headers=headers,
            timeout=60,  # Timeout for each individual attempt
            allow_redirects=False,
        )

        # 1. Check for Redirects (treat as Not Found)
        if response.is_redirect:
            response.close()  # Ensure connection is closed
            return (
                "not_found",
                f"Not Found (Redirect {response.status_code}): {url}",
            )

        # 2. Check for explicit 404
        if response.status_code == 404:
            response.close()
            return "not_found", f"Not Found (404): {url}"

        # 2.5 Check Content-Type for unexpected HTML (treat as Not Found)
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            response.close()
            return "not_found", f"Not Found (HTML Response): {url}"

        # 3. Check for other client/server errors *after* potential retries
        #    The session adapter handles retries for codes in RETRY_STATUS_CODES.
        #    If it still fails with one of those, or encounters another error
        #    (like 403 Forbidden), raise_for_status() will trigger an exception.
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses not handled above

        # 4. If successful (2xx), proceed with download
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=target_dir, suffix=".tmp_dl"
        ) as tmp_file_handle:
            tmp_name = tmp_file_handle.name
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file_handle.write(chunk)

        os.rename(tmp_name, final_path)
        tmp_name = None  # Prevent cleanup in finally block
        return "downloaded", f"Downloaded: {final_path}"  # Success!

    except requests.exceptions.RetryError as e:
        # This exception wraps the original error after retries are exhausted
        error_msg = f"RetryError: Max retries exceeded for {url} -> {e}"
        log.error(f"ERROR: {error_msg}")
        return "failed", error_msg

    except requests.exceptions.RequestException as e:
        # Handle other request exceptions (connection, timeout *not caught by retry*, HTTP errors from raise_for_status)
        # Log the final error after potential session retries failed
        error_msg = f"RequestException ({type(e).__name__}): {url} -> {e}"
        # Check if it's an HTTPError to include status code if available
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            error_msg = (
                f"RequestException (HTTP {e.response.status_code}): {url} -> {e}"
            )

        log.error(f"ERROR: {error_msg}")
        return "failed", error_msg

    except Exception as e:
        # Catch other unexpected errors (e.g., filesystem during rename)
        error_msg = f"Unexpected Error ({type(e).__name__}): {url} -> {e}"
        log.error(f"ERROR: {error_msg}")
        return "failed", error_msg

    finally:
        # Ensure temp file is cleaned up if download failed mid-way or rename failed
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError as cleanup_err:
                log.warning(
                    f"Warning: Failed to clean up temporary file {tmp_name}: {cleanup_err}"
                )
        # Close response connection if it exists and wasn't closed earlier
        if response:
            response.close()


def process_game_seq(base_dir, game_id, seq, session, verbose_log):
    """
    Handles downloading for a specific game ID and sequence number.
    For seq=6, attempts to download both image (.webp) and video (.webm or .mp4).

    Returns:
        list: A list of tuples, where each tuple represents a download attempt:
              [(file_type, status, message), ...]
              file_type is 'image' or 'video'.
              status is 'downloaded', 'skipped', 'not_found', 'failed'.
    """
    results = []

    # --- Image Download (applies to all sequences) ---
    image_ext = "webp"
    image_url_path = f"{game_id}/{seq}.{image_ext}"
    image_url = urljoin(BASE_URL, image_url_path)
    image_final_path = get_final_path(base_dir, game_id, seq, image_ext)
    # Pass verbose_log to download_file for potential retry logging
    image_status, image_message = download_file(
        image_url, image_final_path, session, verbose_log
    )
    results.append(("image", image_status, image_message))
    if verbose_log and image_status in ["downloaded", "skipped"]:
        # Use tqdm.write for thread safety with progress bar
        tqdm.write(f"Task {game_id}/{seq} Image: {image_message}")
    # Failures/Not Found are logged by download_file or handled in main summary

    # --- Video Download (ONLY for sequence 6) ---
    if seq == 6:
        video_status = "pending"  # Status for the video part specifically
        video_message = f"No video found for {game_id}/{seq}"  # Default message
        final_video_result = None  # Store the final tuple for video

        # Define paths
        mp4_path = get_final_path(base_dir, game_id, seq, "mp4")
        webm_path = get_final_path(base_dir, game_id, seq, "webm")

        # --- Attempt MP4 First ---
        if os.path.exists(mp4_path):
            video_status = "skipped"
            video_message = f"Skipped: Video {mp4_path} already exists."
            final_video_result = ("video", video_status, video_message)
            if verbose_log:
                tqdm.write(f"Task {game_id}/{seq} Video: {video_message}")
        else:
            # Try downloading MP4 using the new video path structure
            video_ext = "mp4"
            video_url_path = f"{game_id}/video/{seq}.{video_ext}"
            video_url = urljoin(BASE_URL, video_url_path)
            status, message = download_file(video_url, mp4_path, session, verbose_log)

            if status == "downloaded":
                video_status = status
                video_message = message
                final_video_result = ("video", video_status, video_message)
                if verbose_log:
                    tqdm.write(f"Task {game_id}/{seq} Video: {video_message}")
            elif status == "failed":
                video_status = status
                video_message = message
                final_video_result = ("video", video_status, video_message)
                # Error logging handled by download_file
            elif status == "not_found":
                # MP4 not found, now try WEBM
                if verbose_log:
                    tqdm.write(
                        f"Task {game_id}/{seq} Video: MP4 not found, trying WEBM..."
                    )

                if os.path.exists(webm_path):
                    video_status = "skipped"
                    video_message = f"Skipped: Video {webm_path} already exists."
                    final_video_result = ("video", video_status, video_message)
                    if verbose_log:
                        tqdm.write(f"Task {game_id}/{seq} Video: {video_message}")
                else:
                    # Try downloading WEBM using the new video path structure
                    video_ext = "webm"
                    video_url_path = f"{game_id}/video/{seq}.{video_ext}"
                    video_url = urljoin(BASE_URL, video_url_path)
                    status, message = download_file(
                        video_url, webm_path, session, verbose_log
                    )

                    # Update status/message based on WEBM attempt
                    video_status = status
                    video_message = message
                    if status == "not_found":  # If WEBM also not found
                        video_message = f"Not Found: No video (.mp4 or .webm) found for {game_id}/video/{seq}"  # Update message slightly

                    final_video_result = ("video", video_status, video_message)
                    # Log success/skip if verbose, failures logged by download_file
                    if verbose_log and video_status in ["downloaded", "skipped"]:
                        tqdm.write(f"Task {game_id}/{seq} Video: {video_message}")

        # Append the final video result
        if final_video_result:
            results.append(final_video_result)
        else:
            # This case should ideally not be reached with the logic above, but as a fallback:
            results.append(
                (
                    "video",
                    "failed",
                    f"Failed to determine video status for {game_id}/{seq}",
                )
            )

    return results, game_id, seq  # Return original identifiers too


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Download pictures/videos from guessthe.game."
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=DEFAULT_START_ID,
        help=f"Starting game ID (default: {DEFAULT_START_ID})",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=DEFAULT_END_ID,
        help=f"Ending game ID (inclusive, default: {DEFAULT_END_ID})",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent downloads (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=os.getcwd(),
        help="Base directory to save files (default: current working directory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed status for each successful/skipped file attempt",
    )

    args = parser.parse_args()

    # Setup logging level based on verbosity (optional enhancement)
    # if args.verbose:
    #    log.setLevel(logging.DEBUG) # Or use for more detailed internal logs if needed

    if args.start_id > args.end_id:
        log.error("Error: Start ID cannot be greater than End ID.")
        exit(1)

    os.makedirs(args.directory, exist_ok=True)

    tasks_to_submit = []
    for game_id in range(args.start_id, args.end_id + 1):
        for seq in range(1, 6 + 1):
            tasks_to_submit.append((game_id, seq))

    log.info(f"Starting download for games {args.start_id} to {args.end_id}...")
    log.info(f"Saving to: {os.path.abspath(args.directory)}")
    log.info(f"Concurrency: {args.concurrency}")
    log.info(f"Retries on failure: {MAX_RETRIES} (Delay: {RETRY_DELAY_SECONDS}s)")
    log.info(f"Total tasks (Game ID/Sequence pairs): {len(tasks_to_submit)}")
    if args.verbose:
        log.info("Verbose mode enabled (shows successful/skipped items).")
    log.info("Failures and Retry Warnings will always be shown.")

    downloaded_files = 0
    skipped_files = 0
    failed_files = 0
    not_found_files = 0

    # --- Configure Session with Retries ---
    retry_strategy = Retry(
        total=MAX_RETRIES,
        status_forcelist=RETRY_STATUS_CODES,
        backoff_factor=RETRY_DELAY_SECONDS
        / 2,  # Rough conversion, backoff increases: 0, delay*1, delay*2, delay*4...
        # Note: ConnectionErrors are retried by default
        # You can add method_whitelist=False to retry on POST etc. if needed
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # --- End Session Configuration ---

    # --- Use the configured session ---
    with session:  # <-- Use the pre-configured session
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    process_game_seq,
                    args.directory,
                    game_id,
                    seq,
                    session,
                    args.verbose,
                ): (game_id, seq)
                for game_id, seq in tasks_to_submit
            }

            progress_bar = tqdm(
                total=len(tasks_to_submit), unit="task", desc="Processing Games"
            )
            for future in as_completed(futures):
                original_game_id, original_seq = futures[future]
                try:
                    results_list, _, _ = future.result()

                    for file_type, status, message in results_list:
                        if status == "downloaded":
                            downloaded_files += 1
                            # Verbose logging handled in process_game_seq
                        elif status == "skipped":
                            skipped_files += 1
                            # Verbose logging handled in process_game_seq
                        elif status == "not_found":
                            not_found_files += 1
                            # Only log not found message if verbose (it's not an error)
                            if args.verbose:
                                tqdm.write(
                                    f"Task {original_game_id}/{original_seq} {file_type.capitalize()}: {message}"
                                )
                        elif status == "failed":
                            failed_files += 1
                            # Error logging (final failure) handled in download_file
                        else:
                            failed_files += 1
                            tqdm.write(
                                f"UNEXPECTED STATUS: {status} for {original_game_id}/{original_seq} {file_type} - {message}"
                            )

                except Exception as exc:
                    failed_files += 1
                    tqdm.write(
                        f"ERROR (Future Exception): Task for game {original_game_id} seq {original_seq} generated an exception: {exc}"
                    )
                finally:
                    progress_bar.update(1)

            progress_bar.close()

    print("\n--- Download Summary ---")
    print(f"Total tasks processed (Game ID/Seq pairs): {len(tasks_to_submit)}")
    print(f"Files successfully downloaded: {downloaded_files}")
    print(f"Files skipped (already exist): {skipped_files}")
    print(f"Files not found (404/Redirect): {not_found_files}")
    print(f"Files failed to download (after retries): {failed_files}")
    print("------------------------")


if __name__ == "__main__":
    main()

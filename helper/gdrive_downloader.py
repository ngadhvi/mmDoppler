import subprocess
import os
import re

def download_with_rclone(remote_path, local_path):
    """
    Download a public folder from Google Drive using rclone.
    Note: Requires 'rclone config' to be set up with a remote named 'gdrive'.
    """
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
    
    # Extract folder ID if it's a URL Example: https://drive.google.com/drive/folders/1fI7C13G-UNubbeyqzopRXs2d2cwGM0F5
    folder_id = remote_path
    if "drive.google.com" in remote_path:
        match = re.search(r'folders/([a-zA-Z0-9_-]+)', remote_path)
        if match:
            folder_id = match.group(1)
    
    print(f"Syncing from GDrive folder {folder_id} to {local_path}...")
    
    # Use 'gdrive' remote which the user should configure.
    # The root_folder_id to point directly to the target folder.
    cmd = [
        "rclone", "copy", 
        f"gdrive,root_folder_id={folder_id}:", 
        local_path,
        # "--drive-use-trash=false",
        # "--drive-shared-with-me=true",
        "-P"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during rclone download: {e}")
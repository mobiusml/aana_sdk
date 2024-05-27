import hashlib
from pathlib import Path


def get_sha256_hash_file(filename: Path) -> str:
    """Compute SHA-256 hash of a file without loading it entirely in memory.

    Args:
        filename (Path): Path to the file to be hashed.

    Returns:
        str: SHA-256 hash of the file in hexadecimal format.
    """
    # Create a sha256 hash object
    sha256 = hashlib.sha256()

    # Open the file in binary mode
    with Path.open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    # Return the hexadecimal representation of the digest
    return sha256.hexdigest()

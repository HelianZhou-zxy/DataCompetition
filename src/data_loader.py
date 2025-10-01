import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


class Au20DataLoader:
    """
    A robust data loader for parsing a directory of Au20 cluster .xyz files.

    This class handles finding all .xyz files, parsing each one according to the
    specified format, and compiling them into a single, clean Pandas DataFrame.
    """

    def __init__(self, raw_data_path: str):
        """
        Initializes the data loader with the path to the raw data.

        Args:
            raw_data_path (str): The path to the directory containing .xyz files.
        """
        if not os.path.isdir(raw_data_path):
            raise FileNotFoundError(f"The specified directory does not exist: {raw_data_path}")
        self.raw_data_path = raw_data_path
        self.xyz_files = self._get_xyz_files()
        print(f"Found {len(self.xyz_files)} .xyz files in '{raw_data_path}'.")

    def _get_xyz_files(self) -> list:
        """Finds all .xyz files in the specified directory."""
        return glob.glob(os.path.join(self.raw_data_path, '*.xyz'))

    def _parse_single_xyz(self, file_path: str) -> dict | None:
        """
        Parses a single .xyz file and extracts its properties.

        Returns a dictionary with energy and coordinates, or None if parsing fails.
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Basic validation based on file structure
            if len(lines) < 22:
                print(f"Warning: Skipping malformed file (too short): {os.path.basename(file_path)}")
                return None

            num_atoms = int(lines[0].strip())
            if num_atoms != 20:
                print(f"Warning: Skipping file with incorrect atom count ({num_atoms}): {os.path.basename(file_path)}")
                return None

            total_energy = float(lines[1].strip())

            coordinates = []
            for line in lines[2:22]:
                parts = line.split()
                # Expecting format like "Au   x   y   z"
                coords = [float(p) for p in parts[1:]]
                coordinates.append(coords)

            return {
                'energy': total_energy,
                'coordinates': np.array(coordinates)
            }
        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing file {os.path.basename(file_path)}. Error: {e}. Skipping.")
            return None

    def load_data(self) -> pd.DataFrame:
        """
        Loads and parses all .xyz files into a Pandas DataFrame.

        Each row in the DataFrame represents one Au20 cluster.
        """
        all_clusters_data = []

        # Using tqdm for a progress bar - great for long-running tasks
        for file_path in tqdm(self.xyz_files, desc="Parsing XYZ files"):
            file_id = os.path.basename(file_path)
            parsed_data = self._parse_single_xyz(file_path)

            if parsed_data:
                # Add the file ID for traceability
                parsed_data['id'] = file_id
                all_clusters_data.append(parsed_data)

        if not all_clusters_data:
            print("Warning: No data was loaded. Please check the files in the raw data directory.")
            return pd.DataFrame()

        df = pd.DataFrame(all_clusters_data)

        # Reordering columns for better readability
        return df[['id', 'energy', 'coordinates']]
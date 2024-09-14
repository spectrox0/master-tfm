import os

import requests

# URL of the CSV file
url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"


def main():
# Define the destination folder where the script is located (adjust this path if necessary)
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the destination folder for the CSV
    dest_folder = os.path.join(script_dir, 'src', 'datasets')

# Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

# Define the full path for the CSV file
    dest_file = os.path.join(dest_folder, 'time_series_60min_singleindex.csv')

# Check if the file already exists
    if not os.path.exists(dest_file):
        # Download the CSV file
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest_file, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded successfully and saved to: {dest_file}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"File already exists at: {dest_file}")

if __name__ == "__main__":
    main()
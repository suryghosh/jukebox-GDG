import os
import json
from natsort import natsorted  # Import natsorted for natural sorting

def generate_json(folder_path, output_file="metadata.json"):
    folder_path = os.path.abspath(folder_path)  # Ensure absolute path
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return
    
    data = []
    raaga_index_map = {}  # Dictionary to store unique index per Raaga
    raaga_counter = 0  # Counter for unique index

    for thaat in sorted(os.listdir(folder_path)):  # Sorting for consistency
        thaat_path = os.path.join(folder_path, thaat)
        if not os.path.isdir(thaat_path):
            continue
        
        for raaga in sorted(os.listdir(thaat_path)):  # Raaga instead of sub-thaat
            raaga_path = os.path.join(thaat_path, raaga)
            if not os.path.isdir(raaga_path):
                continue

            # Assign a unique index if not already assigned
            if raaga not in raaga_index_map:
                raaga_index_map[raaga] = raaga_counter
                raaga_counter += 1  # Increment for next unique Raaga

            for song in sorted(os.listdir(raaga_path)):
                song_path = os.path.join(raaga_path, song)
                if not os.path.isdir(song_path):
                    continue

                # Natural sorting of clips
                clips = natsorted(os.listdir(song_path))  

                for clip in clips:
                    relative_song_path = os.path.relpath(os.path.join(song_path, clip), folder_path)
                    
                    data.append({
                        "Thaat": thaat,
                        "Raaga": raaga,  # Replaced "Sub-Thaat" with "Raaga"
                        "Song": clip,  # Corrected Song name
                        "Relative Path": relative_song_path.replace("\\", "/"),  # Cross-platform
                        "Index": raaga_index_map[raaga]  # Unique Raaga index
                    })

    # Write JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✅ JSON file saved as '{output_file}'")
    except Exception as e:
        print(f"❌ Error writing JSON file: {e}")

# Example usage
generate_json("C:/Users/visha/Desktop/GDG/Thaat_and_Raga/Thaat_and_Raga", "metadata.json")

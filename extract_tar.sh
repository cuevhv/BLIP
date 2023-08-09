#!/bin/bash

# Replace 'source_folder' with the path to the folder containing the tar files
source_folder="$1"

# Loop through each tar file in the source folder
for tar_file in "$source_folder"/*.tar; do
    # Extract the filename (without extension) from the path
    filename=$(basename "$tar_file" .tar)

    # Create the 'untar' folder with the same name
    untar_folder="${source_folder}_untar/$filename"
    echo "untar folder: $untar_folder"
    mkdir -p "$untar_folder"

    # Untar the file inside the 'untar' folder
    tar -xvf "$tar_file" -C "$untar_folder"

    echo "Untarred '$tar_file' into '$untar_folder'"
done


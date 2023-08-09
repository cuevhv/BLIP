#!/bin/bash

# Check if the number of arguments is correct
if [ $# -ne 3 ]; then
  echo "Usage: $0 <folder_path> <searching_name> <extension>"
  exit 1
fi

# Get the folder path and output file from the command line arguments
folder_path="$1"
searching_name="$2"  # tonemap
extension="$3"  #.jpg

original_folder_path="$folder_path"
folder_path=$(realpath "$folder_path")

# Use 'find' command to search for files in the specified folder.
# Pipe the results to 'grep' to filter files containing 'color' in their names.
# The results will be stored in the 'color_files' array.
color_files=()

echo "starting search"
while IFS= read -r -d '' file; do
  echo $file
  if echo "$file" | grep -iq "${searching_name}"; then
    color_files+=("$original_folder_path/$file")
    echo $file
  fi
# done < <(find "$folder_path" -type f -print0)
done < <(find "$folder_path" -type f -iname "*${searching_name}*.${extension}" -printf '%P\0')

# Sort the list of files in alphabetical order
sorted_color_files=($(printf '%s\n' "${color_files[@]}" | sort))


# Output the list of files found and save them to the output text file.
save_pth="$original_folder_path/dataset_imgs_list.out"
if [ ${#sorted_color_files[@]} -gt 0 ]; then
  echo "Files with 'color' in the name:"
  printf '%s\n' "${sorted_color_files[@]}"
  printf '%s\n' "${sorted_color_files[@]}" > "$save_pth"
  echo "Paths saved to $save_pth"
else
  echo "No files with '${searching_name}' in the name found."
fi

export PYTHONBUFFERED=1
export PATH=$PATH

folder_name="logs"

# Check if the folder exists
if [ ! -d "$folder_name" ]; then
    # If the folder doesn't exist, create it
    mkdir "$folder_name"
    echo "Folder '$folder_name' created successfully."
else
    echo "Folder '$folder_name' already exists."
fi

/home/hcuevas/miniconda3/envs/control/bin/python img_caption.py --file_fn $1 --out_json_fn out.json --parallel

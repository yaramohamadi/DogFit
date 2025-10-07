find /export/datasets/public/diffusion_datasets/ -type d -name '*processed*' | while read -r dir; do
  zip_file=$(find "$dir" -maxdepth 1 -type f -name '*.zip')
  if [[ -n "$zip_file" ]]; then
    mv "$zip_file" "$(dirname "$dir")/"
    echo "Moved $zip_file to $(dirname "$dir")/"
  fi
done
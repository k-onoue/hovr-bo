for file in config/*_3.json; do cp "$file" "${file/_3.json/_4.json}"; done

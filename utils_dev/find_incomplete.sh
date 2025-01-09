for dir in results/*; do [ -d "$dir" ] && (cd "$dir" && for f in *.csv; do [ -f "${f%.csv}.png" ] || { echo "$dir"; break; }; done); done

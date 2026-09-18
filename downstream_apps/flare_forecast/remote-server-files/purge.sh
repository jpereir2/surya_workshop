#!/usr/bin/env bash
while true; do
    find ~/scratch_space/surya_cache -type f -amin +10 -delete 2>/dev/null
    find ~/scratch_space/surya_cache -type d -empty -delete 2>/dev/null
    echo "[$(date +%H:%M)] purged, $(df -h /home | awk 'NR==2{print $4}') free"
    sleep 300
done

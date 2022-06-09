#!/usr/bin/env bash

################################################################################
#            Video Search Engine Dataset Creator - frame extractor
# -----------------------------------------------------------------------------
# This script extracts frames from *.mp4 videos at a specified framerate. For
# every input video it creates a directory with the same name as the video and
# puts all the extracted frames in that directory. The script uses ffmpeg for 
# extracting the frames, so make sure it is on your path. 
# 
# This script supports the following MANDATORY options:
#   -d <DIR>        The directory with the .mp4 videos. Will also be used to 
#                   create the new directories in.
#   -r <RATE>       The rate to sample the video at.
#
# This script supports the following OPTIONAL options:
#   -n <STR>        a string indicating how the frames will be named/modified. 
#                   E.g. "%03d" will produce frames named "frame_000.jpg", 
#                   "frame_001.jpg", etc. By default "%03d" will be used.
#
# Author: Aron Hoogeveen <aron.hoogeveen@gmail.com>
#
################################################################################

while getopts ":d:r:m:h" opt; do
    case ${opt} in
        d ) 
            dir=$OPTARG
        ;;
        r )
            rate=$OPTARG
        ;;
        m )
            modifier=$OPTARG
        ;;
        h )
            echo "           Video Search Engine Dataset Creator - frame extractor"
            echo "-----------------------------------------------------------------------------"
            echo "This script extracts frames from *.mp4 videos at a specified framerate. For"
            echo "every input video it creates a directory with the same name as the video and"
            echo "puts all the extracted frames in that directory. The script uses ffmpeg for "
            echo "extracting the frames, so make sure it is on your path. "
            echo ""
            echo "This script supports the following MANDATORY options:"
            echo "  -d <DIR>        The directory with the .mp4 videos. Will also be used to "
            echo "                  create the new directories in."
            echo "  -r <RATE>       The rate to sample the video at."
            echo ""
            echo "This script supports the following OPTIONAL options:"
            echo "  -m <STR>        a string indicating how the frames will be named/modified."
            echo "                  E.g. \"%03d\" will produce frames named \"frame_000.jpg\", "
            echo "                  \"frame_001.jpg\", etc. By default \"%03d\" will be used."
            echo ""
            echo "Author: Aron Hoogeveen <aron.hoogeveen@gmail.com>"
            exit 0
        ;;
        \? )
            >&2 echo "Unrecognized option: ${OPTARG}"
            exit 1
        ;;
        : )
            >&2 echo "Invalid option: '${OPTARG}' requires an argument"
            exit 1
        ;;
    esac
done
shift $((OPTIND -1))

# Check mandatory arguments
if [[ -z "${dir}" ]] || [[ -z "${rate}" ]]; then
    echo ">>> Argument -d and -r cannot be empty. Aborting."
    exit 1
fi

# Check optional arguments
if [[ -z "${modifier}" ]]; then
    modifier='%03d'
fi

if [[ ! -d "${dir}" ]]; then
    >&2 echo ">>> Directory \"${dir}\" does not exist!"
    exit 1
fi
cd "${dir}" || { >&2 echo ">>> ERROR: could not change to directory \"${dir}\""; exit 1; }
echo ">>> Processing all .mp4 files in \"$(pwd)\""

# Check if FFMPEG is available
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is not on your path! Do you have it installed?"
    exit 1
fi

# For all files that conform to *.mp4 we create a directory and extract all the
# frames (for framerate ${rate}) into it.
for file in *.mp4; do
    [[ -f "${file}" ]] || continue
    echo ">>> Processing file \"${file}\""

    path="${file%.mp4}"  # remove the .mp4 extension
    
    if [[ -d "${path}" ]]; then
        >&2 echo "!!! The directory \"$(pwd)/${path}\" already exists. Skipping this video."
        continue
    fi

    mkdir "${path}"
    ffmpeg -i "${file}" -r "${rate}" -hide_banner -loglevel error -s hd720 -start_number 0 "${path}"/frame_"${modifier}".jpg
    
done

echo ">>> Done."
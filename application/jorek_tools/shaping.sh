#!/bin/sh

# Greps for LCFS shape. Shaping parameters follow in the 8 lines
# after the grep hit so recover these lines.
# Only get the last grep hit with tail
if [ -z "$1" ]; then
	echo "Usage: shaping.sh <jorek_logfile>"
	exit
fi

cat $1 | grep "LCFS shape" -A 8 | tail -8

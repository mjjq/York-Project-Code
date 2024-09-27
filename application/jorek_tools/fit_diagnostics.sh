#!/bin/sh

# Run this script in the root dir of a simulation run to print basic results

echo "input"
echo "-------------"
cat logfile | grep eqdsk_psi_fact

echo
echo "X-points"
echo "-------------"
cat logfile | grep 'Lower X-point :'
cat logfile | grep 'Upper X-point :'

echo
echo "Psi"
echo "-------------"
cat logfile | grep -A 10 -- 'Plasma_type' | tail -10 # The -- tells unix to stop evaluating dashes as command line options

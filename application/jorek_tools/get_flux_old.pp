namelist intear
for step 0 do

qprofile

expressions Psi_N r_minor
mark_coords 1
set surfaces 200
average

expressions r_minor currdens
mark_coords 1
set surfaces 200
average

done

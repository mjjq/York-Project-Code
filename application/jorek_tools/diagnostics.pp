namelist inmastu

si-units
set verbose true
set debug true

set surfaces 200
set rad_range_max 0.97

for step 6300 do

qprofile

expressions Psi_N r_minor ne T_i T_e pres FFprime_loc zj currdens Btor vpar
mark_coords 1
average

done

exit

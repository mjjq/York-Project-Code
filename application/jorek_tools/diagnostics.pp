namelist inmastu

si-units
set verbose true
set debug true

for step 0 to 99999 do

qprofile

expressions Psi_N r_minor ne T_i T_e pres FFprime_loc zj currdens Btor vpar
mark_coords 1
set surfaces 200
average

done

exit

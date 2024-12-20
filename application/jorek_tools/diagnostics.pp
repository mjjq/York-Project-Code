namelist inmastu

si-units
set verbose true
set debug true

for step 0 do

qprofile

expressions Psi_N ne T_i T_e pres FFprime_loc zj currdens Btor V_phi vpar
mark_coords 1
set surfaces 200
set nmaxsteps 6000
average

done

exit

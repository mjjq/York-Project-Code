namelist inmastu



for step 0 to 99999 do

set surfaces 1000
qprofile

expressions Psi_N r_minor Jtor zj currdens rho
mark_coords 1
average

done

exit

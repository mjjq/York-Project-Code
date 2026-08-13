namelist inmastu


si-units

for step 0 to 99999 do

set surfaces 100

expressions Psi_N r_minor Jtor zj currdens ne T pres
mark_coords 1
average

done

exit

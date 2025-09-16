namelist inmastu



for step 0 do

set surfaces 2000
qprofile

expressions Psi_N r_minor Jtor zj currdens eta_T R Btor rho zkpar_T zkprof
mark_coords 1
average

done

exit

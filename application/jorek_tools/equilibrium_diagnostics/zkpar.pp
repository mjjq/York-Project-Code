namelist inmastu

si-units

set only_itor 0
set rad_range_max 0.9495
for step 0 do
expressions Psi_N zkpar_T zkprof eta_T eta_num_T visco_T visco_num_T
mark_coords 1
set surfaces 200
average
done

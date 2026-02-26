namelist inmastu

si-units

set surfaces 1000

expressions Psi_N r_minor Psi
mark_coords 2

set rad_range_min 0.001
set rad_range_max 0.2

expressions_four absolute phase
for step 0 to 99999 do
four2d
done


namelist inmastu

si-units

set surfaces 5000

expressions Psi_N r_minor Psi
mark_coords 2

set rad_range_min 0.3
set rad_range_max 0.7

expressions_four absolute phase
for step 0 to 99999 do
four2d
done


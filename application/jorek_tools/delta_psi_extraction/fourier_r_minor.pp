namelist inmastu

si-units

set surfaces 100

expressions Psi_N r_minor Psi
mark_coords 2

set rad_range_max 0.945

expressions_four absolute phase
for step 0 to 99999 do
four2d
done


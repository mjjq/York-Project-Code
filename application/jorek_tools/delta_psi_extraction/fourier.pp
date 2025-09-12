namelist inmastu

si-units

set surfaces 100
set rad_range_max 0.945

expressions Psi_N Psi
mark_coords 1

expressions_four absolute phase
for step 0 to 99999 do
four2d
done


namelist inmastu

si-units

set surfaces 100
!set deltaphi 0.1

expressions Psi_N Psi
mark_coords 1

expressions_four absolute phase
for step 0 to 99999 do
four2d
done


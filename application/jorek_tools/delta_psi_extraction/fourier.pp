namelist inmastu

si-units

set surfaces 100
!set deltaphi 0.1

expressions Psi_N Psi
mark_coords 1

expressions_four absolute phase
for step 600 to 2000 by 50 do
four2d
done

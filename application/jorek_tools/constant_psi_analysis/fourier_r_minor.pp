namelist inmastu

si-units
set_postproc_dir ./postproc_fourier_inner/

set surfaces 100
set rad_range_min 0.0
set rad_range_max 0.999

expressions Psi_N r_minor Psi
mark_coords 2

expressions_four absolute phase
for step 8100 to 9500 do
four2d
done


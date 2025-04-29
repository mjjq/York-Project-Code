# Note: Must have the relevant environment variables ($PROJ_HOME etc) specified in ~/.bashrc

alias analysis-venv="source $PROJ_HOME/jorek_analysis/York-Project-Code/venv/bin/activate"

alias plg="$JOREK_UTIL/plot_grids.sh"
alias pld="$JOREK_UTIL/plot_live_data.sh"
alias j2vtk="$JOREK_UTIL/convert2vtk.sh -j 32 ./jorek2vtk ./inmastu"

alias plq="python3 $JOREK_TOOLS/macroscopic_vars_analysis/plot_quantities.py"

batchgrowth() {
        cat useful_runs.txt | parallel 'cd {}; $JOREK_UTIL/extract_live_data.sh -si magnetic_growth_rates > magnetic_growth_rates.dat'
}

getbetas() {
        cat useful_runs.txt | xargs find | grep log | xargs grep -m 1 betap | awk '{printf "%.3f\n", $5}'
}

batchplotgrowth() {
        batchgrowth
        plq -f $(cat useful_runs.txt | xargs find | grep magnetic_growth) -yi 2 -xl "Time (ms)" -yl "Magnetic growth rate (1/s)" -l $(getbetas | sed s'/^/$\\beta_p=$/')
}


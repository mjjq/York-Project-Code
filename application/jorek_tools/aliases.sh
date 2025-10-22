#!/bin/bash

# Note: Must have the relevant environment variables ($PROJ_HOME etc) specified in ~/.bashrc

get_script_dir() {
	cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd
}

alias analysis-venv="source $PROJ_HOME/jorek_analysis/York-Project-Code/venv/bin/activate"

alias plg="$JOREK_UTIL/plot_grids.sh"
alias pld="$JOREK_UTIL/plot_live_data.sh"
alias plq="python3 $JOREK_TOOLS/macroscopic_vars_analysis/plot_quantities.py"

alias grept="grep -riI"

c() {
	python3 $(get_script_dir)/calculator.py "$1"
}

load_jorek_mod_csd() {
	module purge
	module load rhel8/default-icl intel-oneapi-tbb intel-oneapi-mkl fftw
	export HDF5_HOME="/usr/local/Cluster-Apps/hdf5/impi/1.8.9"
	export SZIP_LIB="/usr/local/Cluster-Apps/szip/2.1/lib"
}

jtvtk() {
        $JOREK_UTIL/convert2vtk.sh -j 32 "$@" ./jorek2vtk ./inmastu
}

restart_dir() {
	# Get directory of simulation from which this simulation restarted
	grep jorek_restart.h5 pointers.txt | awk '{print $NF}' | xargs dirname
}

cdres(){
	dname=$(restart_dir)
	# Check if first character is alphanumeric. if it is then this path is
	# likely a relative path.
	if [[ ${dname:0:1} =~ [[:alnum:]] ]]; then
		cd ../$dname
	else
		cd $dname
	fi
}

batchgrowth() {
        cat useful_runs.txt | parallel 'cd {}; $JOREK_UTIL/extract_live_data.sh $1 magnetic_growth_rates > magnetic_growth_rates.dat'
}

batchenergies() {
        cat useful_runs.txt | parallel "cd {}; $JOREK_UTIL/extract_live_data.sh $1 magnetic_energies > magnetic_energies.dat"
}


getbetas() {
        cat useful_runs.txt | xargs find | grep log | xargs grep -m 1 betap | awk '{printf "%.3f\n", $5}'
}

batchqlinputs() {
	cat useful_runs.txt | parallel 'cd {}; ./jorek2_postproc < $JOREK_TOOLS/quasi_linear_model/get_flux.pp'
}

batchqlgrowth() {
	cat useful_runs.txt | parallel -k 'cd {}/postproc; python3 -m experiments.jorek_growth_comparison.curvature_stabilisation 0.0 -si'
}

getlabels() {
	labelsarg="$(cat useful_runs.txt)"
	test -n "$(cat useful_runs_labels.txt)" && labelsarg="$(cat useful_runs_labels.txt)"
	echo "$labelsarg"
}

batchplotgrowth() {
        batchgrowth $1
	mapfile -t label_arr <<< "$(getlabels)"
	if [[ $1 == '-si' ]]; then
	        timelabel="Time (ms)"
		growth_label="Magnetic growth rate (1/s)"
	else
	        timelabel="Time [JOREK units]"
		growth_label="Magnetic growth rate [JOREK units]"
	fi
        plq -f $(cat useful_runs.txt | xargs find | grep magnetic_growth) -yi 2 -xl "$timelabel" -yl "$growth_label" -l "${label_arr[@]}"
}

batchplotenergies() {
        batchenergies $1
	mapfile -t label_arr <<< "$(getlabels)"
	if [[ $1 == '-si' ]]; then
		timelabel="Time (ms)"
	else
		timelabel="Time [JOREK units]"
	fi
	plq -f $(cat useful_runs.txt | xargs find | grep magnetic_energies) -yi 2 -xl "$timelabel" -yl "Normalised magnetic energy (arb)" -l "${label_arr[@]}" -ys log
}


batchplotqprof() {
        plq -f $(cat useful_runs.txt | xargs find | grep qprofile.dat) -xl "$\psi_N$" -yl "Safety factor" -l $(getbetas | sed s'/^/$\\beta_p=$/')
}

get_tstep() {
	logfile=$1
	plot=$2
	grep "time step" $logfile | awk '{print $8}' > t_step.tmp
	grep t_now $logfile | awk '{print $NF}' | sed 's/)://' > t_now.tmp
	paste t_now.tmp t_step.tmp | head -n -1 > tnow_tstep.txt
	if [[ "$plot" == "-p" ]]
	then
		plq -f tnow_tstep.txt -ys log -xl "Simulation time [JU]" -yl "Timestep [JU]"
	fi
}

batchdiagnostic() {
	cat useful_runs.txt | parallel 'cd {}; ./jorek2_postproc < $JOREK_TOOLS/diagnostics.pp'
}

datarun() {
	runnumber=$1
	find $JOREK_DATA -name "*run_$1*" -type d | sort | head -n 1
}

cdrun() {
	runnumber=$1
	cd "$(datarun $1)"
}

cdl() {
	cd $(realpath $1 | xargs dirname)
}

get_latest_h5_in_folder() {
	find $1 | grep -P 'jorek[0-9].*\.h5' | sort -r | head -n 1
}

gather_restart_files() {
	id=0
	for run in $(cat $1)
	do
		ln -s $(get_latest_h5_in_folder $run) ./jorek_restart_$id.h5
		id=$[$id+1]
	done
}

cdlr() {
	cd $(find . -name "run_*" -printf "%T@ %Tc %p\n" | sort -n | tail -1 | awk '{print $NF}')
}


alias fzf='~/applications/fzf'
gg() {
  git grep --line-number --untracked . \
  | fzf --ansi --delimiter : \
        --preview '
          FILE=$(echo {} | cut -d: -f1);
          LINE=$(echo {} | cut -d: -f2);
          START=$((LINE - 10));
          [ $START -lt 1 ] && START=1;
          END=$((LINE + 10));
          sed -n "${START},${END}p" "$FILE" 2>/dev/null
        ' \
        --preview-window=up:60% \
        --bind 'enter:execute(${EDITOR:-vim} +{2} {1})'
}

source $(get_script_dir)/delta_psi_extraction/delta_psi_main.sh

source $(get_script_dir)/postproc_wrapper.sh

source $(get_script_dir)/quasi_linear_model/model_params.sh

source $(get_script_dir)/poincare/plot_multiple.sh

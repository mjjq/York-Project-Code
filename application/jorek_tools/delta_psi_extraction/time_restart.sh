#!/bin/bash

get_time_map() {
	ls jorek[0-9]*.h5 | sed 's/jorek//; s/\.h5//' > indices.tmp
	h5ls -d $(ls jorek[0-9]*.h5 | sed 's/$/\/t_now/') | grep "(0)" | awk '{ print $2 }' > times.tmp

	t_norm=$(h5ls -d $(ls jorek[0-9]*.h5 | head -n 1 | sed 's/$/\/t_norm/') | grep "(0)" | awk '{print $2}')

	paste indices.tmp times.tmp | awk -v t_n=$t_norm '{ print $1 " " $2*t_n }'

	rm indices.tmp times.tmp
}

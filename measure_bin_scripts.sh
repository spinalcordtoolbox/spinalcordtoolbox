> somefile

for x in bin/*
do 
	# time $(basename $x); 
	# time $(basename $x) > /dev/null 2>&1 &

	# start=`date +%s%N | cut -b1-13`

	start=`gdate +%s%N | cut -b1-13`
	# $(basename $x) > somefile 2>&1 &
	$(basename $x);
	# sleep 0.97
	end=`gdate +%s%N | cut -b1-13`
	
	# runtime=$((end-start))

	runtime=`echo "scale=3;($end - $start)/1000" | bc -l`
	
	if [ $(echo " $runtime > 1" | bc) -eq 1 ]
	then
		printee=`echo "$(basename $x), $runtime"`
		echo $printee >> somefile 2>&1 &
		echo $printee
	fi

	echo $(basename $x),  $start, $end, $runtime
done

echo "Passed All"
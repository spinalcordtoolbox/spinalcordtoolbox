> somefile

source python/bin/activate

for x in bin/*
do 
	# time $(basename $x); 
	# time $(basename $x) > /dev/null 2>&1 &
	# time PYTHONPATH=$PWD python scripts/isct_test_ants.py > /dev/null  - Sample

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
		
		# while IFS=', ' read -r line; do
		count=1
		inserted=0
		while IFS=', ' read -r line || [[ -n "$line" ]]; do
			IFS=', ' read -r -a array <<< "$line"
			if [ $(echo "$runtime > ${array[1]}" | bc) -eq 1 ]
			then
				replacee=`echo "$((count))i $printee"`
				# gsed -i "1i isct_minc2volume-viewer, 1.439" somefile
				gsed -i "$replacee" somefile
				inserted=1
				break
			fi
			count=$count+1
			echo "Text read from file: $line"
		done < "somefile"
		
		if [ $inserted -eq 0 ]
		then
			echo $printee >> somefile 2>&1 &
		fi
		echo $printee
	fi

	echo $(basename $x),  $start, $end, $runtime
done

echo "Passed All"
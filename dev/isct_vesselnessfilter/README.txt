### COMPILING THE VESSELNESS FILTER

Note : ITK "Review modules" should be compiled on neuropoly@ferguson. If they are not compiled, the compilation will fail to find itkMultiScaleHessianBasedMeasureImageFilter.h
1. Compile ITK with "review modules" ON (Only have to do it once)
	a. clone ITK in a directory (It does not matter which)
	b. Create build folder inside the ITK folder
	c. while inside the build folder do : ccmake ..
	d. press c to start cmake configuration
	e. press t to show advanced compilation options
	f. check for Module_ITKReview and set it to ON
	g. press c to configure
	h. press g to generate makefiles
	i. press q to quit
	j. enter the command : make (It's really long to compile everything)
	k. sudo make install

2. Compile VTK on your computer (if it is not already compiled)
	a. clone/extract VTK in a directory
	b. create a build folder
	c. while inside the build folder, do : ccmake ..
	d. press c, then c, then g, then q
	e. execute : make
	f. execute : make install

3. Compile isct_vesselnessFilter
	a. use make_binaries.sh in sct/install

	OR

	b.- make a build folder under dev/isct_vesselnessfilter
	  - while inside the build folder, do : ccmake ..
	  - press c, then c, then g, then q
	  - make
	  - copy the binary produced to bin/linux or bin/osx

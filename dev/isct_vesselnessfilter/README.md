### COMPILING THE VESSELNESS FILTER

Note : ITK "Review modules" should be compiled on neuropoly@ferguson. If they are not compiled, the compilation will fail to find itkMultiScaleHessianBasedMeasureImageFilter.h

1. Compile ITK with "review modules" ON (Only have to do it once)
2. test
3. test
  1. test
  2. test


  1. clone ITK in a directory (It does not matter which)
  2. test
  3. test
  4. 
  

  2. Create build folder inside the ITK folder
  3. while inside the build folder do : ccmake ..
  4. press c to start cmake configuration
2. Compile VTK on your computer (if it is not already compiled)

	e. press t to show advanced compilation options
	f. check for Module_ITKReview and set it to ON
	g. press c to configure
	h. press g to generate makefiles
	i. press q to quit
	j. enter the command : make (It's really long to compile everything)
	k. sudo make install


	a. clone/extract VTK 6.3.0 in a directory (find it here: http://www.vtk.org/download/)
	b. create a build folder: ``mkdir build && cd build``
	c. while inside the build folder, type: ``ccmake ..``
	d. press c, then c, then g, then q
	e. type: ``make``
	f. type: ``make install``

3. Compile isct_vesselnessFilter
	a. use make_binaries.sh in sct/install

	OR

	b.- make a build folder under dev/isct_vesselnessfilter
	  - while inside the build folder, type: ``ccmake ..``
	  - press c, then c, then g, then q
	  - type: ``make``
	  - copy the binary produced to bin/linux or bin/osx

### COMPILING THE VESSELNESS FILTER

Note : ITK "Review modules" should be compiled on neuropoly@ferguson. If they are not compiled, the compilation will fail to find itkMultiScaleHessianBasedMeasureImageFilter.h

1. Compile ITK with "review modules" ON (Only have to do it once)
  1. clone ITK in a directory (It does not matter which)
  2. Create build folder inside the ITK folder
  3. while inside the build folder do : ccmake ..
  4. press c to start cmake configuration
  5. press t to show advanced compilation options
  6. check for Module_ITKReview and set it to ON
  7. press c to configure
  8. press g to generate makefiles
  9. press q to quit
  10. enter the command : make (It's really long to compile everything)
  11. sudo make install
2. Compile VTK on your computer (if it is not already compiled)
  1. clone/extract VTK 6.3.0 in a directory (find it here: http://www.vtk.org/download/)
  2. create a build folder: ``mkdir build && cd build``
  3. while inside the build folder, type: ``ccmake ..``
  4. press c, then c, then g, then q
  5. type: ``make``
  6. type: ``make install``
3. Compile isct_vesselnessFilter
  - use make_binaries.sh in sct/install, OR:
  - make a build folder under dev/isct_vesselnessfilter
    1. while inside the build folder, type: ``ccmake ..``
    2. press c, then c, then g, then q
    3. type: ``make``
    4. copy the binary produced to bin/linux or bin/osx

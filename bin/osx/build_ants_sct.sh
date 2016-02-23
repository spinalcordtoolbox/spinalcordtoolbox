#!/bin/bash
make -j 4

cd bin

cp ANTSUseLandmarkImagesWithTextFileToGetBSplineDisplacementField ${SCT_DIR}/bin/osx

cp ANTSUseLandmarkImagesWithTextFileToGetThinPlateDisplacementField ${SCT_DIR}/bin/osx

cp antsApplyTransforms ${SCT_DIR}/bin/osx

cp antsRegistration ${SCT_DIR}/bin/osx

cp antsSliceRegularizedRegistration ${SCT_DIR}/bin/osx

cp ANTSUseLandmarkImagesToGetAffineTransform ${SCT_DIR}/bin/osx

cp ComposeMultiTransform ${SCT_DIR}/bin/osx

cd

cd ${SCT_DIR}/bin/osx

if [ -f isct_ANTSUseLandmarkImagesWithTextFileToGetBSplineDisplacementField ]
then
echo Removing existing binary isct_ANTSUseLandmarkImagesWithTextFileToGetBSplineDisplacementField
rm isct_ANTSUseLandmarkImagesWithTextFileToGetBSplineDisplacementField
fi

if [ -f isct_ANTSUseLandmarkImagesWithTextFileToGetThinPlateDisplacementField ]
then
echo Removing existing binary isct_ANTSUseLandmarkImagesWithTextFileToGetThinPlateDisplacementField
rm isct_ANTSUseLandmarkImagesWithTextFileToGetThinPlateDisplacementField
fi

if [ -f isct_antsApplyTransforms ]
then
echo Removing existing binary isct_antsApplyTransforms
rm isct_antsApplyTransforms
fi

if [ -f isct_antsRegistration ]
then
echo Removing existing binary isct_antsRegistration
rm isct_antsRegistration
fi

if [ -f isct_antsSliceRegularizedRegistration ]
then
echo Removing existing binary isct_antsSliceRegularizedRegistration
rm isct_antsSliceRegularizedRegistration
fi

if [ -f isct_ANTSUseLandmarkImagesToGetAffineTransform ]
then
echo Removing existing binary isct_ANTSUseLandmarkImagesToGetAffineTransform
rm isct_ANTSUseLandmarkImagesToGetAffineTransform
fi

if [ -f isct_ComposeMultiTransform ]
then
echo Removing existing binary isct_ComposeMultiTransform
rm isct_ComposeMultiTransform
fi

echo renaming binaries
mv ANTSUseLandmarkImagesWithTextFileToGetBSplineDisplacementField isct_ANTSUseLandmarkImagesWithTextFileToGetBSplineDisplacementField

mv ANTSUseLandmarkImagesWithTextFileToGetThinPlateDisplacementField isct_ANTSUseLandmarkImagesWithTextFileToGetThinPlateDisplacementField

mv antsApplyTransforms isct_antsApplyTransforms

mv antsRegistration isct_antsRegistration

mv antsSliceRegularizedRegistration isct_antsSliceRegularizedRegistration

mv ANTSUseLandmarkImagesToGetAffineTransform isct_ANTSUseLandmarkImagesToGetAffineTransform

mv ComposeMultiTransform isct_ComposeMultiTransform
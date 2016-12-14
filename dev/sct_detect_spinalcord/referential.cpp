#include "referential.h"
#include <cmath>
#include <iostream>
using namespace std;

// Par défaut crée le référentiel origine. Matrice identité et origine nulle
Referential::Referential()
{
}


Referential::Referential(const CMatrix3x3 &m, const CVector3 &o): axes(m), origine(o)
{
}


Referential::Referential(double **m, const double *o)
{
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) axes(i,j) = m[j][i];
	}
	origine = CVector3(o[0],o[1],o[2]);
}


Referential::Referential(const CVector3 &X, const CVector3 &Y, const CVector3 &Z, const CVector3 &o): origine(o)
{
	axes[0] = X[0];
	axes[1] = X[1];
	axes[2] = X[2];
	axes[3] = Y[0];
	axes[4] = Y[1];
	axes[5] = Y[2];
	axes[6] = Z[0];
	axes[7] = Z[1];
	axes[8] = Z[2];
}


Referential::Referential(const Referential &ref)
{
	origine = ref.origine;
	axes = ref.axes;
}


Referential::~Referential()
{
}


void Referential::setX(const CVector3 &x)
{
	axes(0,0) = x[0];
	axes(0,1) = x[1];
	axes(0,2) = x[2];
}
void Referential::setY(const CVector3 &y)
{
	axes(1,0) = y[0];
	axes(1,1) = y[1];
	axes(1,2) = y[2];
}
void Referential::setZ(const CVector3 &z)
{
	axes(2,0) = z[0];
	axes(2,1) = z[1];
	axes(2,2) = z[2];
}
void Referential::setOrigine(const CVector3 &origine)
{
	this->origine = origine;
}


void Referential::transformation(const CVector3 &translation, const CMatrix3x3 rotation)
{
	origine += translation;
	axes *= rotation;
}

void Referential::transformation(const CMatrix4x4 &transformation)
{
	CMatrix3x3 temp = CMatrix3x3(transformation);
	axes *= temp;
	origine[0] += transformation[12];
	origine[1] += transformation[13];
	origine[2] += transformation[14];
}


Referential* Referential::transformationRef(const CMatrix4x4 &transformation)
{
	Referential* result = new Referential(*this);
	result->transformation(transformation);
	return result;
}


CMatrix4x4 Referential::getTransformation()
{
	// A = T*B où A est la matrice du referentiel next, B est la matrice courante et T est la transformation
	// La solution pour trouver T est donc : T = A*inv(B)
	Referential next;
	CMatrix4x4 result = next.axes*axes.Inverse();
	result[12] = next.origine[0] - origine[0];
	result[13] = next.origine[1] - origine[1];
	result[14] = next.origine[2] - origine[2];

	//cout << "transformation : " << endl << result << endl;

	return result;
}


CMatrix4x4 Referential::getTransformation(const Referential &next)
{
	// A = T*B où A est la matrice du referentiel next, B est la matrice courante et T est la transformation
	// La solution pour trouver T est donc : T = A*inv(B)
	CMatrix4x4 result = next.axes*axes.Inverse();
	result[12] = next.origine[0] - origine[0];
	result[13] = next.origine[1] - origine[1];
	result[14] = next.origine[2] - origine[2];

	//cout << "transformation : " << endl << result << endl;

	return result;
}


CMatrix4x4 Referential::getTransformationInverse()
{
	CMatrix4x4 result = getTransformation();
	CMatrix3x3 resultTemp = result;
	CVector3 translation = CVector3(-result[12],-result[13],-result[14]);
	result = resultTemp.Inverse();
	result[12] = translation[0]; result[13] = translation[1]; result[14] = translation[2];
	return result;
}


CMatrix4x4 Referential::getTransformationInverse(const Referential &last)
{
	CMatrix4x4 result = getTransformation(last);
	CMatrix3x3 resultTemp = result;
	CVector3 translation = CVector3(-result[12],-result[13],-result[14]);
	result = resultTemp.Inverse();
	result[12] = translation[0]; result[13] = translation[1]; result[14] = translation[2];
	return result;
}


Referential Referential::operator+(const Referential& ref)
{
	Referential result = Referential(*this);
	result.axes += ref.axes;
	result.origine += ref.origine;
	return result;
}


Referential Referential::operator/(float value)
{
	Referential result = Referential(*this);
	result.axes /= value;
	result.origine /= value;
	return result;
}


ostream& operator << (std::ostream& os, const Referential& ref )
{
	os << "Origine : ( " << ref.origine[0] << ", " << ref.origine[1] << ", " << ref.origine[2] << " )" << endl;
	os << "Axes : " << endl << ref.axes;

	return os;
}


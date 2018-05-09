#ifndef __REFERENTIAL__
#define __REFERENTIAL__

/*!
 * \file referential.h
 * \brief Class for referential (axes + origin)
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include "util/Matrix4x4.h"
#include "util/Matrix3x3.h"
#include "util/Vector3.h"

/*!
 * \class Referential
 * \brief Class for referential (axes + origin)
 * 
 * This class contains structure and methods for referentials, represented by three axes and an origin.
 * Transformation towards origin referential and other referential are available as well as common operation.
 */
class Referential
{
public:
	Referential();
	Referential(const CMatrix3x3 &m, const CVector3 &o);
	Referential(double **m, const double *o);
	Referential(const CVector3 &X, const CVector3 &Y, const CVector3 &Z, const CVector3 &o);
	Referential(const Referential &ref);
	virtual ~Referential();

	CVector3 getX() const { return axes.getColumn(0); };
	CVector3 getY() const { return axes.getColumn(1); };
	CVector3 getZ() const { return axes.getColumn(2); };
	CVector3 getOrigine() const { return origine; };

	void setX(const CVector3 &x);
	void setY(const CVector3 &y);
	void setZ(const CVector3 &z);
	void setOrigine(const CVector3 &origine);

	void transformation(const CVector3 &translation, const CMatrix3x3 rotation);
	void transformation(const CMatrix4x4 &transformation);
	Referential* transformationRef(const CMatrix4x4 &transformation);

	CMatrix4x4 getTransformation(); // Calcule la transformation du referentiel courant vers le referentiel origine
	CMatrix4x4 getTransformation(const Referential &next); // Calcule la transformation du referentiel courant vers le next
	CMatrix4x4 getTransformationInverse(); // Calcule la transformation du referentiel origine vers le courant
	CMatrix4x4 getTransformationInverse(const Referential &last); // Calcule la transformation du referentiel last vers le courant

	Referential operator+(const Referential& ref);
	Referential operator/(float value);

	/// Stream output operator.
	friend std::ostream& operator << (std::ostream& os, const Referential& ref );

public:
	CMatrix3x3 axes; // axes par colonne
	CVector3 origine;

};

#endif // _REFERENTIAL_H_

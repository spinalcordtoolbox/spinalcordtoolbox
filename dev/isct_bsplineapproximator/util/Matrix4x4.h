//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#ifndef _MATRIX4X4_H_
#define _MATRIX4X4_H_


#include "Vector3.h"
#include <iostream>
using namespace std;

class	CMatrix3x3;


/// Definition of a 4x4 matrix.
/**
The matrix elements are indexed in this way (like OpenGL):
<pre>
0  4  8 12
1  5  9 13
2  6 10 14
3  7 11 15
</pre>
*/
class CMatrix4x4
{
public:
	/// Default contructor.
	CMatrix4x4();
	
	/// Build a matrix from an array of floats.
	CMatrix4x4( const float *mat );


	/// Sets the upper-left submatrix with a 3x3 matrix.
	/**
	All the matrix elements are initialized.  Those that are not
	initialized by the 3x3 matrix are set like an identity matrix.
	*/
	CMatrix4x4( const CMatrix3x3 &mat );

	/// Matrix-matrix addition.
	CMatrix4x4	operator + ( const CMatrix4x4 &m ) const;

	/// Matrix-matrix subtraction.
	CMatrix4x4	operator - ( const CMatrix4x4 &m ) const;

	/// Matrix-matrix product.
	CMatrix4x4	operator * ( const CMatrix4x4 &mat ) const;

	/// Matrix-vector product.
	CVector3	operator * ( const CVector3 &v ) const;

	/// Matrix-scalar multiplication.
	CMatrix4x4	operator * ( const float &a ) const;

	/// Matrix-scalar division.
	CMatrix4x4	operator / ( const float &a ) const;

	/// Matrix-matrix add and assign operator.
	CMatrix4x4&	operator += ( const CMatrix4x4 &m );

	/// Matrix-matrix subtract and assign operator.
	CMatrix4x4&	operator -= ( const CMatrix4x4 &m );

	/// Matrix-matrix multiply and assign operator.
	CMatrix4x4&	operator *= ( const CMatrix4x4 &m );

	/// Matrix-scalar multiply and assign operator.
	CMatrix4x4&	operator *= ( const float &a );

	/// Matrix-scalar divide and assign operator.
	CMatrix4x4&	operator /= ( const float &a );

	/// Returns a reference to the indexed matrix element.
	float&		operator [] ( const int index )			{ return m[index]; }

	/// Returns a reference to the indexed matrix element.
	float		operator [] ( const int index )	const	{ return m[index]; }

	/// Scales the diagonal's elements.
	void		ScaleDiagonal( const float &a );

	/// Returns the matrix's transpose.
	CMatrix4x4	Transpose() const;

	/// Multiplies the matrix with a rotation matrix.
	/**
	@param Axis the rotation axis.
	@param Angle the angle of rotation in radian.
	*/
	void		Rotate( const CVector3& Axis, const float Angle );

	/// Multiplies the matrix with a translation matrix.
	/**
	@param Translation the translation vector.
	*/
	void		Translate( const CVector3& Translation );

	/// Multiplies the matrix with a scaling marix
	/**
	@param Scaling the scaling vector.
	*/
	void		Scale( const CVector3& Scaling );

	/// Returns a pointer to the matrix elements.
	operator	float*()				{ return m; }

	/// Returns a constant pointer to the matrix elements.
	operator	const float*() const	{ return m;	}

	/// Identity matrix.
	static const CMatrix4x4		I;

	/// Zero matrix.
	static const CMatrix4x4		ZERO;

	/// Scalar-matrix multiplication.
	friend CMatrix4x4	operator * ( const float &a, const CMatrix4x4 &m );

	/// Scalar-matrix division.
	friend CVector3		operator * ( const CVector3& v, const CMatrix4x4& m );

	friend ostream& operator << (ostream& os, const CMatrix4x4 &mat );

	CMatrix4x4	Inverse();

protected:
	/// The matrix elements.
	float	m[16];
};


#endif


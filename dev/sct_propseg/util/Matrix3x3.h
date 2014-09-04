//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#ifndef _MATRIX3X3_H_
#define _MATRIX3X3_H_

#include <iostream>
using namespace std;

class CVector3;
class CMatrix4x4;

/// Definition of a 3x3 matrix.
/**
The matrix elements are indexed in this way (like OpenGL):
<pre>
0  3  6
1  4  7
2  5  8
</pre>
*/
class CMatrix3x3
{
public:
	/// Default constructor.
	CMatrix3x3();
	
	/// Build a matrix from an array of floats.
	CMatrix3x3( const float *mat );

	/// Initialize the matrix with the upper-left sub-matrix of a 4x4 matrix.
	CMatrix3x3( const CMatrix4x4 &mat );

	/// Operator =
	CMatrix3x3&	operator = ( const CMatrix3x3 &mat );

	/// Matrix-matrix addition.
	CMatrix3x3	operator + ( const CMatrix3x3 &m ) const;

	/// Matrix-matrix subtraction.
	CMatrix3x3	operator - ( const CMatrix3x3 &m ) const;

	/// Matrix-matrix product.
	CMatrix3x3	operator * ( const CMatrix3x3 &m ) const;

	/// Matrix-vector product.
	CVector3	operator * ( const CVector3 &v ) const;

	/// Matrix-scalar multiplication.
	CMatrix3x3	operator * ( const float &a ) const;

	/// Matrix-scalar division.
	CMatrix3x3	operator / ( const float &a ) const;

	/// Matrix-matrix add and assign operator.
	CMatrix3x3&	operator += ( const CMatrix3x3 &m );

	/// Matrix-matrix subtract and assign operator.
	CMatrix3x3&	operator -= ( const CMatrix3x3 &m );

	/// Matrix-matrix multiply and assign operator.
	CMatrix3x3&	operator *= ( const CMatrix3x3 &m );

	/// Matrix-scalar multiply and assign operator.
	CMatrix3x3&	operator *= ( const float &a );

	/// Matrix-scalar divide and assign operator.
	CMatrix3x3&	operator /= ( const float &a );

	/// Returns a reference to the indexed matrix element.
	float&		operator [] ( const int index )			{ return m[index]; }

	/// Returns the value of the indexed matrix element.
	float		operator [] ( const int index ) const	{ return m[index]; }

	/// Returns the tranpose of the matrix.
	CMatrix3x3	Transpose() const;

	/// Returns the inverse of the matrix.
	CMatrix3x3	Inverse() const;

	/// Returns the matrix determinant.
	float		Determinant() const;

	/// Scales the elements of the diagonal.
	void		ScaleDiagonal( const float &a );

	/// Normalize the columns.
	void		NormalizeColumns();

	/// Returns a pointer to the matrix's elements
	operator	float*()				{ return m; }

	/// Returns a constant pointer to the matrix elements.
	operator	const float*() const	{ return m;	}

	/// Identity matrix.
	static const	CMatrix3x3	I;

	/// Zero matrix.
	static const	CMatrix3x3	ZERO;

	/// Scalar-matrix muliplication operator.
	friend CMatrix3x3	operator * ( const float &a, const CMatrix3x3 &m );

	/// Scalar-matrix division operator.
	friend CMatrix3x3	operator / ( const float &a, const CMatrix3x3 &m );

	float&		operator()(int ligne, int colonne);

	CVector3	getColumn(int i) const;

	friend ostream& operator << (ostream& os, const CMatrix3x3 &mat );

protected:

	/// The matrix elements.
	float	m[9];
};


#endif


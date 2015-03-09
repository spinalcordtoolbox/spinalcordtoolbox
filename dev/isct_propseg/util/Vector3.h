//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#ifndef _VECTOR3_H_
#define _VECTOR3_H_

#if _MSC_VER == 1200
#pragma warning(disable:4786)
#endif

#include <iostream>


class	CMatrix3x3;
class	CVector4;


/// Definition of a vector in 3 dimensions.
class CVector3
{
public:

	/// Default constructor: sets the vector to zero.
	CVector3() { x = y = z = 0.0f; }

	/// Construct a vector with the specified components.
	CVector3( float x_, float y_, float z_ ) { x = x_; y = y_; z = z_; }

	/// Initialize the vector with a 4 dimensions vector.  It is automatically divided by the w component.
	CVector3( const CVector4& v );

	CVector3&	operator = ( const CVector3 &v );

	/// Addition operator.
	CVector3	operator + ( const CVector3 &v ) const;

	/// Subtraction operator.
	CVector3	operator - ( const CVector3 &v ) const;

	/// Dot product operator.
	float		operator * ( const CVector3 &v ) const;

	/// Cross product operator.
	CVector3	operator ^ ( const CVector3 &v ) const;

	/// Vector-scalar multiplication.
	CVector3	operator * ( const float &a ) const;

	/// Vector-scalar multiplication.
	CVector3	operator * ( const CMatrix3x3 &m ) const;

	/// Vector-scalar division.
	CVector3	operator / ( const float &a ) const;

	/// Cross product and assign operator.
	CVector3&	operator ^= ( const CVector3 &v );

	/// Vector-scalar multiplication and assign operator.
	CVector3&	operator *= ( const float &a );

	/// Vector-scalar division and assign operator.
	CVector3&	operator /= ( const float &a );

	/// Add and assign operator.
	CVector3&	operator += ( const CVector3 &v );

	/// Subtract and assign operator.
	CVector3&	operator -= ( const CVector3 &v );

	/// Returns a reference to a vector component.
	float&		operator [] ( const int i )			{ return *(&x + i);	}

	/// Returns a vector component.
	float		operator [] ( const int i )	const	{ return *(&x + i);	}

	/// Inversion.
	CVector3	operator - () const;

	/// change value rapidly.
	void		operator () ( float x, float y, float z );

	/// Equality boolean operator
	bool		operator == ( const CVector3& v ) const;

	/// Inequality boolean operator
	bool		operator != ( const CVector3& v ) const;

	/// Scalar-vector multiplication operator.
	friend		CVector3 operator * ( const float &a, const CVector3 &v );

	/// Scalar-vector division operator.
	friend		CVector3	operator / ( const float &a, const CVector3 &v );

	/// Stream output operator.
	friend		std::ostream& operator << (std::ostream& os, const CVector3& v );

	/// Returns the vector's norm.
	float		Norm() const;

	/// Normalize the vector.
	CVector3	Normalize();

	/// Star operator.
	/**
	Returns the following 3x3 matrix:
	<pre>
	 0 -z  y
	 z  0 -x
	-y  x  0
	</pre>
	*/
	CMatrix3x3	Star() const;

	/// Returns the smallest vector component (if not w).
	float		Max() const;

	/// Returns the greatest vector component (if not w).
	float		Min() const;

	/// A zero vector.
	static const CVector3	ZERO;

	float
	/// The x component.
	x,
	/// The y component.
	y,
	/// The z component.
	z;
};



#endif


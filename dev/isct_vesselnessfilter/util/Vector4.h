//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#ifndef _VECTOR4_H_
#define _VECTOR4_H_

#if _MSC_VER == 1200
#pragma warning(disable:4786)
#endif

#include <iostream>


/// Definition of a vector in 4 dimensions.
class CVector4
{
public:

	/// Default constructor: sets the vector to zero.
	CVector4() { x = y = z = 0.0f; w = 1.0f; }

	/// Initialize the vector with the specified components.
	CVector4( float x_, float y_, float z_, float w_ = 1.0f ) { x = x_; y = y_; z = z_; w = w_; }

	/// Addition operator.
	CVector4	operator + ( const CVector4 &v ) const;

	/// Subtraction operator.
	CVector4	operator - ( const CVector4 &v ) const;

	/// Dot product operator.
	float		operator * ( const CVector4 &v ) const;

	/// Vector-scalar multiplication.
	CVector4	operator * ( const float &a ) const;

	/// Vector-scalar division.
	CVector4	operator / ( const float &a ) const;

	/// Vector-scalar multiplication and assign operator.
	CVector4&	operator *= ( const float &a );

	/// Vector-scalar division and assign operator.
	CVector4&	operator /= ( const float &a );

	/// Add and assign operator.
	CVector4&	operator += ( const CVector4 &v );

	/// Subtract and assign operator.
	CVector4&	operator -= ( const CVector4 &v );

	/// Inversion.
	CVector4	operator - () const;

	/// Returns a reference to a vector component.
	float&		operator [] ( const int i )			{ return *(&x + i);	}

	/// Returns a vector component.
	float		operator [] ( const int i )	const	{ return *(&x + i);	}

	/// Equality boolean operator
	bool		operator == ( const CVector4& v ) const;

	/// Inequality boolean operator
	bool		operator != ( const CVector4& v ) const;

	/// Scalar-vector multiplication operator.
	friend		CVector4 operator * ( const float &a, const CVector4 &v );

	/// Scalar-vector division operator.
	friend		CVector4 operator / ( const float &a, const CVector4 &v );

	/// Stream output operator.
	friend		std::ostream& operator << (std::ostream& os, const CVector4& v );

	/// Returns the vector's norm.
	float		Norm() const;

	/// Normalize the vector.
	CVector4	Normalize();

	/// Returns the smallest vector component.
	float		Max() const;

	/// Returns the greatest vector component.
	float		Min() const;

	/// A zero vector.
	static const CVector4	ZERO;

	float
	/// The x component.
	x,
	/// The y component.
	y,
	/// The z component.
	z,
	/// The w component.
	w;
};



#endif


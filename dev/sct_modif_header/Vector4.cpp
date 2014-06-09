//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#include "Vector4.h"
#include "math.h"

using namespace std;



const CVector4	CVector4::ZERO	= CVector4( 0.0f, 0.0f, 0.0f, 0.0f );


// Addition
CVector4 CVector4::operator + ( const CVector4 &v ) const
{
	return CVector4( x + v.x, y + v.y, z + v.z, w + v.w );
}


// Subtraction
CVector4 CVector4::operator - ( const CVector4 &v ) const
{
	return CVector4( x - v.x, y - v.y, z - v.z, w - v.w );
}


// Dot product
float CVector4::operator * ( const CVector4 &v ) const
{
	return x * v.x + y * v.y + z * v.z + w * v.w;
}


// Scale
CVector4 CVector4::operator * ( const float &a ) const
{
	return CVector4( x*a, y*a, z*a, w*a );
}


// Scale
CVector4 CVector4::operator / ( const float &a ) const
{
	return CVector4( x/a, y/a, z/a, w/a );
}


// Scale
CVector4& CVector4::operator *= ( const float &a )
{
	return ( *this = *this * a );
}


// Scale
CVector4& CVector4::operator /= ( const float &a )
{
	return ( *this = *this / a );
}


CVector4&	CVector4::operator += ( const CVector4 &v )
{
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;

	return *this;
}


CVector4&	CVector4::operator -= ( const CVector4 &v )
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;

	return *this;
}


CVector4	CVector4::operator- () const
{
	return CVector4( -x, -y, -z, -w );
}


bool CVector4::operator == ( const CVector4& v ) const
{
	return ( x == v.x && y == v.y && z == v.z && w == v.w );
}


bool CVector4::operator != ( const CVector4& v ) const
{
	return ( x != v.x || y != v.y || z != v.z || w != v.w );
}


// Scale
CVector4 operator * ( const float &a, const CVector4 &v )
{
	return CVector4( v.x * a, v.y * a, v.z * a, v.w * a );
}


// Scale
CVector4	operator / ( const float &a, const CVector4 &v )
{
	return CVector4( v.x / a, v.y / a, v.z / a, v.w / a );
}


// Return norm
float CVector4::Norm() const
{
	return (float)sqrt( x*x + y*y + z*z + w*w );
}


// Normalize the vector
CVector4 CVector4::Normalize()
{
	return (*this = *this / Norm());
}


float CVector4::Max() const
{
	if ( x > y && x > z && x > w )
		return x;
	else if( y > z && y > w )
		return y;
	else if( z > w )
		return z;
	else
		return w;
}


float CVector4::Min() const
{
	if ( x < y && x < z && x < w )
		return x;
	else if( y < z && y < w )
		return y;
	else if( z < w )
		return z;
	else
		return w;
}


ostream& operator << ( ostream& os, const CVector4& v )
{
	os << "( " << v.x << ", " << v.y << ", " << v.z << ", " << v.w << " )";

	return os;
}

//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#include "Vector3.h"
#include "Vector4.h"
#include "Matrix3x3.h"
#include "math.h"

using namespace std;


const CVector3	CVector3::ZERO	= CVector3( 0.0f, 0.0f, 0.0f );



CVector3::CVector3( const CVector4& v )
{
	x = v.x / v.w;
	y = v.y / v.w;
	z = v.z / v.w;
}


CVector3& CVector3::operator = ( const CVector3 &v )
{
	x = v.x;
	y = v.y;
	z = v.z;
	return *this;
}


// Addition
CVector3 CVector3::operator + ( const CVector3 &v ) const
{
	return CVector3( x + v.x, y + v.y, z + v.z );
}


// Subtraction
CVector3 CVector3::operator - ( const CVector3 &v ) const
{
	return CVector3( x - v.x, y - v.y, z - v.z );
}


// Dot product
float CVector3::operator * ( const CVector3 &v ) const
{
	return x * v.x + y * v.y + z * v.z;
}


// Cross product
CVector3 CVector3::operator ^ ( const CVector3 &v ) const
{
	return CVector3( y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x );
}


// Scale
CVector3 CVector3::operator * ( const float &a ) const
{
	return CVector3( x*a, y*a, z*a );
}


// Multiplication vector * matrix
CVector3 CVector3::operator * ( const CMatrix3x3 &m ) const
{
	return CVector3( x*m[0]+y*m[1]+z*m[2], x*m[3]+y*m[4]+z*m[5], x*m[6]+y*m[7]+z*m[8] );
}


// Scale
CVector3 CVector3::operator / ( const float &a ) const
{
	return CVector3( x/a, y/a, z/a );
}


// Cross product
CVector3& CVector3::operator ^= ( const CVector3 &v )
{
	return ( *this = *this ^ v );
}


// Scale
CVector3& CVector3::operator *= ( const float &a )
{
	return ( *this = *this * a );
}


// Scale
CVector3& CVector3::operator /= ( const float &a )
{
	return ( *this = *this / a );
}


CVector3&	CVector3::operator += ( const CVector3 &v )
{
	x += v.x;
	y += v.y;
	z += v.z;

	return *this;
}


CVector3&	CVector3::operator -= ( const CVector3 &v )
{
	x -= v.x;
	y -= v.y;
	z -= v.z;

	return *this;
}


CVector3	CVector3::operator- () const
{
	return CVector3( -x, -y, -z );
}


void CVector3::operator () ( float x, float y, float z )
{
	this->x = x;
	this->y = y;
	this->z = z;
}


bool CVector3::operator == ( const CVector3& v ) const
{
	return ( x == v.x && y == v.y && z == v.z );
}


bool CVector3::operator != ( const CVector3& v ) const
{
	return ( x != v.x || y != v.y || z != v.z );
}


// Scale
CVector3 operator * ( const float &a, const CVector3 &v )
{
	return CVector3( v.x * a, v.y * a, v.z * a );
}


// Scale
CVector3	operator / ( const float &a, const CVector3 &v )
{
	return CVector3( v.x / a, v.y / a, v.z / a );
}


// Return norm
float CVector3::Norm() const
{
	return (float)sqrt( x*x + y*y + z*z );
}


// Normalize the vector
CVector3 CVector3::Normalize()
{
    if (Norm() <= 1e-12) return CVector3();
	return (*this = *this / Norm());
}


CMatrix3x3	CVector3::Star() const
{
	CMatrix3x3	ret;

	ret[0] = 0;
	ret[1] = z;
	ret[2] = -y;
	ret[3] = -z;
	ret[4] = 0;
	ret[5] = x;
	ret[6] = y;
	ret[7] = -x;
	ret[8] = 0;

	return ret;
}


float CVector3::Max() const
{
	if ( x > y && x > z )
		return x;
	else if( y > z )
		return y;
	else
		return z;
}


float CVector3::Min() const
{
	if ( x < y && x < z )
		return x;
	else if( y < z )
		return y;
	else
		return z;
}


ostream& operator << ( ostream& os, const CVector3& v )
{
	os << "( " << v.x << ", " << v.y << ", " << v.z << " )";

	return os;
}

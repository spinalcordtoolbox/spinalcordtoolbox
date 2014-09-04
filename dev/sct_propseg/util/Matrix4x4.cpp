//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#include "Matrix4x4.h"
#include "Matrix3x3.h"
#include "InverseMatrix.h"
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;


const CMatrix4x4 CMatrix4x4::I		= CMatrix4x4();
const CMatrix4x4 CMatrix4x4::ZERO	= CMatrix4x4() - CMatrix4x4();


CMatrix4x4::CMatrix4x4()
{
	m[0] = m[5] = m[10] = m[15] = 1;
	m[1] = m[2] = m[3] = m[4] = m[6] = m[7] = m[8] = m[9] = m[11]= m[12] = m[13] = m[14] = 0;
}


CMatrix4x4::CMatrix4x4( const float *mat )
{
	memcpy( m, mat, 16 * sizeof( float ) );
}


CMatrix4x4::CMatrix4x4( const CMatrix3x3 &mat )
{
	m[0]	= mat[0];
	m[1]	= mat[1];
	m[2]	= mat[2];
	m[3]	= 0;
	m[4]	= mat[3];
	m[5]	= mat[4];
	m[6]	= mat[5];
	m[7]	= 0;
	m[8]	= mat[6];
	m[9]	= mat[7];
	m[10]	= mat[8];
	m[11]	= 0;
	m[12]	= 0;
	m[13]	= 0;
	m[14]	= 0;
	m[15]	= 1;
}


CVector3	CMatrix4x4::operator * ( const CVector3 &v ) const
{
	CVector3 v2;

	v2.x = m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12];
	v2.y = m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13];
	v2.z = m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14];

	return v2;
}

CVector3		operator * ( const CVector3& v, const CMatrix4x4& m )
{
	CVector3	v2;

	v2.x = m.m[0] * v.x + m.m[1] * v.y + m.m[2] * v.z + m.m[3];
	v2.x = m.m[4] * v.x + m.m[5] * v.y + m.m[6] * v.z + m.m[7];
	v2.x = m.m[8] * v.x + m.m[9] * v.y + m.m[10] * v.z + m.m[11];

	return v2;
}


CMatrix4x4 CMatrix4x4::operator * ( const CMatrix4x4 &mat ) const
{
	CMatrix4x4 ret;

	ret.m[0]	= m[0] * mat.m[0] + m[4] * mat.m[1] + m[8] * mat.m[2] + m[12] * mat.m[3];
	ret.m[4]	= m[0] * mat.m[4] + m[4] * mat.m[5] + m[8] * mat.m[6] + m[12] * mat.m[7];
	ret.m[8]	= m[0] * mat.m[8] + m[4] * mat.m[9] + m[8] * mat.m[10] + m[12] * mat.m[11];
	ret.m[12]	= m[0] * mat.m[12] + m[4] * mat.m[13] + m[8] * mat.m[14] + m[12] * mat.m[15];

	ret.m[1]	= m[1] * mat.m[0] + m[5] * mat.m[1] + m[9] * mat.m[2] + m[13] * mat.m[3];
	ret.m[5]	= m[1] * mat.m[4] + m[5] * mat.m[5] + m[9] * mat.m[6] + m[13] * mat.m[7];
	ret.m[9]	= m[1] * mat.m[8] + m[5] * mat.m[9] + m[9] * mat.m[10] + m[13] * mat.m[11];
	ret.m[13]	= m[1] * mat.m[12] + m[5] * mat.m[13] + m[9] * mat.m[14] + m[13] * mat.m[15];

	ret.m[2]	= m[2] * mat.m[0] + m[6] * mat.m[1] + m[10] * mat.m[2] + m[14] * mat.m[3];
	ret.m[6]	= m[2] * mat.m[4] + m[6] * mat.m[5] + m[10] * mat.m[6] + m[14] * mat.m[7];
	ret.m[10]	= m[2] * mat.m[8] + m[6] * mat.m[9] + m[10] * mat.m[10] + m[14] * mat.m[11];
	ret.m[14]	= m[2] * mat.m[12] + m[6] * mat.m[13] + m[10] * mat.m[14] + m[14] * mat.m[15];

	ret.m[3]	= m[3] * mat.m[0] + m[7] * mat.m[1] + m[11] * mat.m[2] + m[15] * mat.m[3];
	ret.m[7]	= m[3] * mat.m[4] + m[7] * mat.m[5] + m[11] * mat.m[6] + m[15] * mat.m[7];
	ret.m[11]	= m[3] * mat.m[8] + m[7] * mat.m[9] + m[11] * mat.m[10] + m[15] * mat.m[11];
	ret.m[15]	= m[3] * mat.m[12] + m[7] * mat.m[13] + m[11] * mat.m[14] + m[15] * mat.m[15];

	return ret;
}


CMatrix4x4	CMatrix4x4::operator * ( const float &a ) const
{
	CMatrix4x4 ret( *this );

	ret.m[0]	*= a;
	ret.m[1]	*= a;
	ret.m[2]	*= a;
	ret.m[3]	*= a;
	ret.m[4]	*= a;
	ret.m[5]	*= a;
	ret.m[6]	*= a;
	ret.m[7]	*= a;
	ret.m[8]	*= a;
	ret.m[9]	*= a;
	ret.m[10]	*= a;
	ret.m[11]	*= a;
	ret.m[12]	*= a;
	ret.m[13]	*= a;
	ret.m[14]	*= a;
	ret.m[15]	*= a;

	return ret;
}


CMatrix4x4 operator * ( const float &a, const CMatrix4x4 &m )
{
	CMatrix4x4	ret( m );

	ret.m[0]	*= a;
	ret.m[1]	*= a;
	ret.m[2]	*= a;
	ret.m[3]	*= a;
	ret.m[4]	*= a;
	ret.m[5]	*= a;
	ret.m[6]	*= a;
	ret.m[7]	*= a;
	ret.m[8]	*= a;
	ret.m[9]	*= a;
	ret.m[10]	*= a;
	ret.m[11]	*= a;
	ret.m[12]	*= a;
	ret.m[13]	*= a;
	ret.m[14]	*= a;
	ret.m[15]	*= a;

	return ret;
}


CMatrix4x4	CMatrix4x4::operator / ( const float &a ) const
{
	CMatrix4x4 ret( *this );

	ret.m[0]	/= a;
	ret.m[1]	/= a;
	ret.m[2]	/= a;
	ret.m[3]	/= a;
	ret.m[4]	/= a;
	ret.m[5]	/= a;
	ret.m[6]	/= a;
	ret.m[7]	/= a;
	ret.m[8]	/= a;
	ret.m[9]	/= a;
	ret.m[10]	/= a;
	ret.m[11]	/= a;
	ret.m[12]	/= a;
	ret.m[13]	/= a;
	ret.m[14]	/= a;
	ret.m[15]	/= a;

	return ret;
}


CMatrix4x4	CMatrix4x4::operator + ( const CMatrix4x4 &m ) const
{
	CMatrix4x4 ret(*this);

	ret.m[0] += m.m[0];
	ret.m[1] += m.m[1];
	ret.m[2] += m.m[2];
	ret.m[3] += m.m[3];
	ret.m[4] += m.m[4];
	ret.m[5] += m.m[5];
	ret.m[6] += m.m[6];
	ret.m[7] += m.m[7];
	ret.m[8] += m.m[8];
	ret.m[9] += m.m[9];
	ret.m[10] += m.m[10];
	ret.m[11] += m.m[11];
	ret.m[12] += m.m[12];
	ret.m[13] += m.m[13];
	ret.m[14] += m.m[14];
	ret.m[15] += m.m[15];

	return ret;
}


CMatrix4x4&	CMatrix4x4::operator += ( const CMatrix4x4 &m )
{
	return ( *this = *this + m );
}


CMatrix4x4	CMatrix4x4::operator - ( const CMatrix4x4 &m ) const
{
	CMatrix4x4 ret(*this);

	ret.m[0] -= m.m[0];
	ret.m[1] -= m.m[1];
	ret.m[2] -= m.m[2];
	ret.m[3] -= m.m[3];
	ret.m[4] -= m.m[4];
	ret.m[5] -= m.m[5];
	ret.m[6] -= m.m[6];
	ret.m[7] -= m.m[7];
	ret.m[8] -= m.m[8];
	ret.m[9] -= m.m[9];
	ret.m[10] -= m.m[10];
	ret.m[11] -= m.m[11];
	ret.m[12] -= m.m[12];
	ret.m[13] -= m.m[13];
	ret.m[14] -= m.m[14];
	ret.m[15] -= m.m[15];

	return ret;
}


CMatrix4x4&	CMatrix4x4::operator -= ( const CMatrix4x4 &m )
{
	return ( *this = *this - m );
}


CMatrix4x4&	CMatrix4x4::operator *= ( const CMatrix4x4 &m )
{
	return ( *this = *this * m );
}


CMatrix4x4&	CMatrix4x4::operator *= ( const float &a )
{
	return ( *this = *this * a );
}


CMatrix4x4&	CMatrix4x4::operator /= ( const float &a )
{
	return ( *this = *this * a );
}


void	CMatrix4x4::ScaleDiagonal( const float &a )
{
	m[0]	*= a;
	m[5]	*= a;
	m[10]	*= a;
	m[15]	*= a;
}


CMatrix4x4 CMatrix4x4::Transpose() const
{
	CMatrix4x4	ret;

	ret.m[0]	= m[0];
	ret.m[1]	= m[4];
	ret.m[2]	= m[8];
	ret.m[3]	= m[12];
	ret.m[4]	= m[1];
	ret.m[5]	= m[5];
	ret.m[6]	= m[9];
	ret.m[7]	= m[13];
	ret.m[8]	= m[2];
	ret.m[9]	= m[6];
	ret.m[10]	= m[10];
	ret.m[11]	= m[14];
	ret.m[12]	= m[3];
	ret.m[13]	= m[7];
	ret.m[14]	= m[11];
	ret.m[15]	= m[15];

	return ret;
}


void CMatrix4x4::Rotate( const CVector3& Axis, const float Angle )
{
	// Code inspired from Mesa3D
	float s, c;
	float xx, yy, zz, xy, yz, zx, xs, ys, zs, one_c;
	CMatrix4x4 mat;

	s = sinf( Angle );
	c = cosf( Angle );

	xx = Axis.x * Axis.x;
	yy = Axis.y * Axis.y;
	zz = Axis.z * Axis.z;
	xy = Axis.x * Axis.y;
	yz = Axis.y * Axis.z;
	zx = Axis.z * Axis.x;
	xs = Axis.x * s;
	ys = Axis.y * s;
	zs = Axis.z * s;
	one_c = 1.0F - c;

	mat[0] = (one_c * xx) + c;
	mat[4] = (one_c * xy) - zs;
	mat[8] = (one_c * zx) + ys;
	mat[12] = 0.0F;

	mat[1] = (one_c * xy) + zs;
	mat[5] = (one_c * yy) + c;
	mat[9] = (one_c * yz) - xs;
	mat[13] = 0.0F;

	mat[2] = (one_c * zx) - ys;
	mat[6] = (one_c * yz) + xs;
	mat[10] = (one_c * zz) + c;
	mat[14] = 0.0F;

	mat[3] = 0.0F;
	mat[7] = 0.0F;
	mat[11] = 0.0F;
	mat[15] = 1.0F;

	*this = (*this) * mat;
}


void CMatrix4x4::Translate( const CVector3& Translation )
{
	m[12]	= m[0] * Translation.x + m[4] * Translation.y + m[8] * Translation.z + m[12];
	m[13]	= m[1] * Translation.x + m[5] * Translation.y + m[9] * Translation.z + m[13];
	m[14]	= m[2] * Translation.x + m[6] * Translation.y + m[10] * Translation.z + m[14];
	m[15]	= m[3] * Translation.x + m[7] * Translation.y + m[11] * Translation.z + m[15];
}


void CMatrix4x4::Scale( const CVector3& Scaling )
{
	m[0]	*= Scaling.x;
	m[4]	*= Scaling.y;
	m[8]	*= Scaling.z;

	m[1]	*= Scaling.x;
	m[5]	*= Scaling.y;
	m[9]	*= Scaling.z;

	m[2]	*= Scaling.x;
	m[6]	*= Scaling.y;
	m[10]	*= Scaling.z;

	m[3]	*= Scaling.x;
	m[7]	*= Scaling.y;
	m[11]	*= Scaling.z;
}


CMatrix4x4 CMatrix4x4::Inverse()
{
	int order = 4;
	float **A = new float*[order], **Y = new float*[order];
	for (int i=0; i<order; i++) {
		A[i] = new float[order];
		Y[i] = new float[order];
	}
	for (int i=0; i<order; i++) {
		for (int j=0; j<order; j++)
			A[j][i] = m[order*i+j];
	}

	MatrixInversion inv(A,order,Y);

	CMatrix4x4 result;
	for (int i=0; i<order; i++) {
		for (int j=0; j<order; j++)
			result[order*i+j] = Y[j][i];
	}
	return result;
}


ostream& operator << (ostream& os, const CMatrix4x4 &mat )
{
	os << " ( " << mat.m[0] << ", " << mat.m[4] << ", " << mat.m[8] << ", " << mat.m[12] << " )" << endl;
	os << " ( " << mat.m[1] << ", " << mat.m[5] << ", " << mat.m[9] << ", " << mat.m[13] << " )" << endl;
	os << " ( " << mat.m[2] << ", " << mat.m[6] << ", " << mat.m[10] << ", " << mat.m[14] << " )" << endl;
	os << " ( " << mat.m[3] << ", " << mat.m[7] << ", " << mat.m[11] << ", " << mat.m[15] << " )" << endl;

	return os;
}

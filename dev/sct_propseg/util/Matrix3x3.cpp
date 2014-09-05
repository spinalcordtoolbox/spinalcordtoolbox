//-----------------------------------------------------------------------------
// This file is part of the b-reality project:
// http://sourceforge.net/projects/b-reality
//
// (C) Francis Page  2003
//-----------------------------------------------------------------------------

#include "Matrix3x3.h"
#include "Matrix4x4.h"
#include "Vector3.h"
#include "InverseMatrix.h"
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;


const CMatrix3x3 CMatrix3x3::I		= CMatrix3x3();
const CMatrix3x3 CMatrix3x3::ZERO	= CMatrix3x3() - CMatrix3x3();


CMatrix3x3::CMatrix3x3()
{
	m[0] = m[4] = m[8] = 1;
	m[1] = m[2] = m[3] = m[5] = m[6] = m[7] = 0;
}


CMatrix3x3::CMatrix3x3( const float *mat )
{
	memcpy( m, mat, 9 * sizeof( float ) );
}


CMatrix3x3::CMatrix3x3( const CMatrix4x4 &mat )
{
	m[0]	= mat[0];
	m[1]	= mat[1];
	m[2]	= mat[2];
	m[3]	= mat[4];
	m[4]	= mat[5];
	m[5]	= mat[6];
	m[6]	= mat[8];
	m[7]	= mat[9];
	m[8]	= mat[10];
}


CMatrix3x3& CMatrix3x3::operator=( const CMatrix3x3 &mat )
{
	m[0]	= mat.m[0];
	m[1]	= mat.m[1];
	m[2]	= mat.m[2];
	m[3]	= mat.m[3];
	m[4]	= mat.m[4];
	m[5]	= mat.m[5];
	m[6]	= mat.m[6];
	m[7]	= mat.m[7];
	m[8]	= mat.m[8];
	return *this;
}


CVector3	CMatrix3x3::operator * ( const CVector3 &v ) const
{
	CVector3 v2;

	v2.x = m[0] * v.x + m[3] * v.y + m[6] * v.z;
	v2.y = m[1] * v.x + m[4] * v.y + m[7] * v.z;
	v2.z = m[2] * v.x + m[5] * v.y + m[8] * v.z;

	return v2;
}


CMatrix3x3 CMatrix3x3::operator * ( const CMatrix3x3 &mat ) const
{
	CMatrix3x3 ret;

	ret.m[0]	= m[0] * mat.m[0]	+ m[3] * mat.m[1] + m[6] * mat.m[2];
	ret.m[1]	= m[1] * mat.m[0]	+ m[4] * mat.m[1] + m[7] * mat.m[2];
	ret.m[2]	= m[2] * mat.m[0]	+ m[5] * mat.m[1] + m[8] * mat.m[2];
	ret.m[3]	= m[0] * mat.m[3]	+ m[3] * mat.m[4] + m[6] * mat.m[5];
	ret.m[4]	= m[1] * mat.m[3]	+ m[4] * mat.m[4] + m[7] * mat.m[5];
	ret.m[5]	= m[2] * mat.m[3]	+ m[5] * mat.m[4] + m[8] * mat.m[5];
	ret.m[6]	= m[0] * mat.m[6]	+ m[3] * mat.m[7] + m[6] * mat.m[8];
	ret.m[7]	= m[1] * mat.m[6]	+ m[4] * mat.m[7] + m[7] * mat.m[8];
	ret.m[8]	= m[2] * mat.m[6]	+ m[5] * mat.m[7] + m[8] * mat.m[8];

	return ret;
}


CMatrix3x3& CMatrix3x3::operator *= ( const CMatrix3x3 &mat )
{
	return ( *this = *this * mat );
}


CMatrix3x3	CMatrix3x3::operator * ( const float &a ) const
{
	CMatrix3x3 ret( *this );

	ret.m[0]	*= a;
	ret.m[1]	*= a;
	ret.m[2]	*= a;
	ret.m[3]	*= a;
	ret.m[4]	*= a;
	ret.m[5]	*= a;
	ret.m[6]	*= a;
	ret.m[7]	*= a;
	ret.m[8]	*= a;

	return ret;
}


CMatrix3x3&	CMatrix3x3::operator *= ( const float &a )
{
	return ( *this = *this * a );
}


CMatrix3x3 operator * ( const float &a, const CMatrix3x3 &m )
{
	CMatrix3x3	ret( m );

	ret.m[0]	*= a;
	ret.m[1]	*= a;
	ret.m[2]	*= a;
	ret.m[3]	*= a;
	ret.m[4]	*= a;
	ret.m[5]	*= a;
	ret.m[6]	*= a;
	ret.m[7]	*= a;
	ret.m[8]	*= a;

	return ret;
}


CMatrix3x3	CMatrix3x3::operator / ( const float &a ) const
{
	CMatrix3x3 ret( *this );

	ret.m[0]	/= a;
	ret.m[1]	/= a;
	ret.m[2]	/= a;
	ret.m[3]	/= a;
	ret.m[4]	/= a;
	ret.m[5]	/= a;
	ret.m[6]	/= a;
	ret.m[7]	/= a;
	ret.m[8]	/= a;

	return ret;
}


CMatrix3x3&	CMatrix3x3::operator /= ( const float &a )
{
	return ( *this = *this * a );
}


CMatrix3x3	CMatrix3x3::operator + ( const CMatrix3x3 &m ) const
{
	CMatrix3x3 ret(*this);

	ret.m[0] += m.m[0];
	ret.m[1] += m.m[1];
	ret.m[2] += m.m[2];
	ret.m[3] += m.m[3];
	ret.m[4] += m.m[4];
	ret.m[5] += m.m[5];
	ret.m[6] += m.m[6];
	ret.m[7] += m.m[7];
	ret.m[8] += m.m[8];

	return ret;
}


CMatrix3x3&	CMatrix3x3::operator += ( const CMatrix3x3 &m )
{
	return ( *this = *this + m );
}


CMatrix3x3	CMatrix3x3::operator - ( const CMatrix3x3 &m ) const
{
	CMatrix3x3 ret(*this);

	ret.m[0] -= m.m[0];
	ret.m[1] -= m.m[1];
	ret.m[2] -= m.m[2];
	ret.m[3] -= m.m[3];
	ret.m[4] -= m.m[4];
	ret.m[5] -= m.m[5];
	ret.m[6] -= m.m[6];
	ret.m[7] -= m.m[7];
	ret.m[8] -= m.m[8];

	return ret;
}


CMatrix3x3&	CMatrix3x3::operator -= ( const CMatrix3x3 &m )
{
	return ( *this = *this - m );
}


CMatrix3x3 operator / ( const float &a, const CMatrix3x3 &m )
{
	CMatrix3x3	ret( m );

	ret.m[0]	/= a;
	ret.m[1]	/= a;
	ret.m[2]	/= a;
	ret.m[3]	/= a;
	ret.m[4]	/= a;
	ret.m[5]	/= a;
	ret.m[6]	/= a;
	ret.m[7]	/= a;
	ret.m[8]	/= a;

	return ret;
}


void	CMatrix3x3::ScaleDiagonal( const float &a )
{
	m[0]	*= a;
	m[4]	*= a;
	m[8]	*= a;
}


void	CMatrix3x3::NormalizeColumns()
{
	float	c1 = sqrtf( m[0] * m[0] + m[1] * m[1] + m[2] * m[2] );
	float	c2 = sqrtf( m[3] * m[3] + m[4] * m[4] + m[5] * m[5] );
	float	c3 = sqrtf( m[6] * m[6] + m[7] * m[7] + m[8] * m[8] );

	m[0] /= c1;
	m[1] /= c1;
	m[2] /= c1;
	m[3] /= c2;
	m[4] /= c2;
	m[5] /= c2;
	m[6] /= c3;
	m[7] /= c3;
	m[8] /= c3;
}


CMatrix3x3	CMatrix3x3::Transpose() const
{
	CMatrix3x3	ret;

	ret.m[0]	= m[0];
	ret.m[1]	= m[3];
	ret.m[2]	= m[6];
	ret.m[3]	= m[1];
	ret.m[4]	= m[4];
	ret.m[5]	= m[7];
	ret.m[6]	= m[2];
	ret.m[7]	= m[5];
	ret.m[8]	= m[8];

	return ret;
}


CMatrix3x3	CMatrix3x3::Inverse() const
{
	CMatrix3x3	ret;
	float det = Determinant();
	ret[0] = (m[4] * m[8] - m[7] * m[5])/det;
	ret[1] = (m[2] * m[7] - m[1] * m[8])/det;
	ret[2] = (m[1] * m[5] - m[2] * m[4])/det;
	ret[3] = (m[5] * m[6] - m[3] * m[8])/det;
	ret[4] = (m[0] * m[8] - m[2] * m[6])/det;
	ret[5] = (m[2] * m[3] - m[0] * m[5])/det;
	ret[6] = (m[3] * m[7] - m[4] * m[6])/det;
	ret[7] = (m[1] * m[6] - m[0] * m[7])/det;
	ret[8] = (m[0] * m[4] - m[3] * m[1])/det;

	return ret;

	/*int order = 3;
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

	CMatrix3x3 result;
	for (int i=0; i<order; i++) {
		for (int j=0; j<order; j++)
			result[order*i+j] = Y[j][i];
	}

	for (int i=0; i<order; i++) {
		delete [] A[i];
		delete [] Y[i];
	}
	delete [] A;
	delete [] Y;

	return result;*/
}


float		CMatrix3x3::Determinant() const
{
	return	m[0] * ( m[4] * m[8] - m[5] * m[7] ) -
			m[3] * ( m[1] * m[8] - m[2] * m[7] ) +
			m[6] * ( m[1] * m[5] - m[2] * m[4] );
}


float& CMatrix3x3::operator()(int ligne, int colonne)
{
	return m[3*colonne+ligne];
}


CVector3 CMatrix3x3::getColumn(int i) const
{
	return CVector3(m[3*i], m[3*i+1], m[3*i+2]);
}


ostream& operator << (ostream& os, const CMatrix3x3 &mat )
{
	os << " ( " << mat.m[0] << ", " << mat.m[3] << ", " << mat.m[6] << " )" << endl;
	os << " ( " << mat.m[1] << ", " << mat.m[4] << ", " << mat.m[7] << " )" << endl;
	os << " ( " << mat.m[2] << ", " << mat.m[5] << ", " << mat.m[8] << " )" << endl;

	return os;
}

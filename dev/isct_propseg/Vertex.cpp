#include "Vertex.h"
#include <iostream>
#include <cstdlib>
#include <cmath> 

using namespace std;

Vertex::Vertex()
{
	point_ = CVector3();
	label_ = 1;
	label_s_ = "1";
	normal_ = CVector3();
	deform_ = true;
}

Vertex::Vertex(double x, double y, double z)
{
	point_ = CVector3(x,y,z);
	label_ = 1;
	label_s_ = "1";
	normal_ = CVector3();
	deform_ = true;
}

Vertex::Vertex(double x, double y, double z, string labels)
{
	point_ = CVector3(x,y,z);
	label_ = 1;
	label_s_ = labels;
	normal_ = CVector3();
	deform_ = true;
}

Vertex::Vertex(string labels)
{
	point_ = CVector3();
	label_ = 1;
	label_s_ = labels;
	normal_ = CVector3();
	deform_ = true;
}

Vertex::Vertex(CVector3 point)
{
	point_ = point;
	label_ = 1;
	label_s_ = "1";
	normal_ = CVector3();
	deform_ = true;
}

Vertex::Vertex(CVector3 point, bool deform)
{
	point_ = point;
	label_ = 1;
	label_s_ = "1";
	normal_ = CVector3();
	deform_ = deform;
}

Vertex::Vertex(CVector3 point, CVector3 normal)
{
	point_ = point;
	label_ = 1;
	label_s_ = "1";
	normal_ = normal;
	deform_ = true;
}

Vertex::Vertex(CVector3 point, CVector3 normal, int label)
{
	point_ = point;
	label_ = label;
	label_s_ = "1";
	normal_ = normal;
	deform_ = true;
}

Vertex::Vertex(const Vertex &v)
{
	point_ = v.point_;
	label_ = v.label_;
	label_s_ = v.label_s_;
	normal_ = v.normal_;
	deform_ = v.deform_;
}

Vertex::Vertex(const Vertex &v, string labels)
{
	point_ = v.point_;
	label_ = v.label_;
	label_s_ = labels;
	normal_ = v.normal_;
	deform_ = v.deform_;
}

Vertex::~Vertex(void)
{
}

CVector3 Vertex::getPosition()
{
	return point_;
}

void Vertex::setPosition(CVector3 pos)
{
	point_ = pos;
}

void Vertex::setLabel(int label)
{
	label_ = label;
}

void Vertex::setLabelS(string labels)
{
	label_s_ = labels;
}

int Vertex::getLabel()
{
	return label_;
}

string Vertex::getLabelString()
{
	return label_s_;
}

void Vertex::setNormal(double x, double y, double z)
{
	normal_ = CVector3(x,y,z);
}

CVector3 Vertex::getNormal()
{
	return normal_;
}

double Vertex::distance(double x, double y, double z)
{
	return sqrt(pow((point_[0]-x),2) + pow((point_[1]-y),2) + pow((point_[2]-z),2));
}

double Vertex::distance(Vertex& v)
{
	CVector3 pos = v.getPosition();
	return sqrt(pow((point_[0]-pos[0]),2) + pow((point_[1]-pos[1]),2) + pow((point_[2]-pos[2]),2));
}

CVector3 Vertex::transform(CMatrix4x4& transformation)
{
	// La matrice transformation est une matrice 4X4 de type transformation homogène
	return transformation*point_;
}

Vertex& Vertex::operator=(const Vertex &v)
{
	if(this != &v) {
		point_ = v.point_;
		label_ = v.label_;
		label_s_ = v.label_s_;
		normal_ = v.normal_;
	}
	return *this;
}

// deux points sont égaux si leurs coordonnées et leur normale sont égales
bool Vertex::operator==(const Vertex &v)
{
	return point_==v.point_ && normal_==v.normal_;
}

ostream& operator<<( ostream &flux, const Vertex &v )
{
	flux << "[ " << v.point_[0] << " ; " << v.point_[1] << " ; " << v.point_[2] << " ] ; [ " << v.normal_[0] << " ; " << v.normal_[1] << " ; " << v.normal_[2] << " ]";
	return flux;
}

#ifndef __VERTEX__
#define __VERTEX__

/*!
 * \file Vertex.h
 * \brief Point of a mesh, with normal and label.
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <ostream>
#include <string>

#include "util/Vector3.h"
#include "util/Matrix4x4.h"


/*!
 * \class Vertex
 * \brief Point of a mesh, with normal and label.
 * 
 * This class represents a vertex in Mesh. It contains a point and a normal.
 * Several methods are available, like computation of the distance between two vertices, transformations and basic operator.
 */
class Vertex
{
public:
	Vertex();
	Vertex(double x, double y, double z);
	Vertex(double x, double y, double z, std::string labels);
	Vertex(std::string labels);
	Vertex(CVector3 point);
	Vertex(CVector3 point, bool deform);
	Vertex(CVector3 point, CVector3 normal);
	Vertex(CVector3 point, CVector3 normal, int label);
	Vertex(const Vertex &v);
	Vertex(const Vertex &v, std::string labels);
	~Vertex(void);
	CVector3 getPosition();
	void setPosition(CVector3 pos);
	void setLabel(int label);
	void setLabelS(std::string labels);
	int getLabel();
	std::string getLabelString();
	void setNormal(double x, double y, double z);
	CVector3 getNormal();
	double distance(double x, double y, double z);
	double distance(Vertex& v);
	CVector3 transform(CMatrix4x4& transformation);

	Vertex& operator=(const Vertex &v);
	bool operator==(const Vertex &v);
	friend std::ostream& operator<<( std::ostream &flux, const Vertex &v );

	void setDeform(bool deform) { deform_ = deform; };
	bool hasToBeDeform() { return deform_; };
private:
	CVector3 point_;
	int label_;
	std::string label_s_;
	CVector3 normal_;

	bool deform_;
};

#endif // VERTEX_H

#ifndef __FONCTEUR_PLAN__
#define __FONCTEUR_PLAN__

/*!
 * \file foncteurPlan.h
 * \brief Compute the plan minimizing the distance between points and the plan
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <vector>

#include "Vertex.h"


/*!
 * \struct FoncteurPlan
 * \brief Compute the plan minimizing the distance between points and the plan
 *
 * This structure allow to compute the equation of the plan which minimizing the distance (RMS) between a set of points and the plan.
 */
struct FoncteurPlan {
	std::vector<CVector3>* points_;

	//! Constructor of functor
    /*!
	  \param points Set of points used to compute plan equation
    */
	FoncteurPlan(std::vector<CVector3>* points)
	{
		points_ = points;
	}

	//! Operator() computing distance (RMS) between set of points and the plan of equation ax+by+cz+d=0
    /*!
      \param p Parameters of the plan equation (p[0]=a, p[1]=b, p[2]=c, p[3]=d)
	  \return The root mean square (RMS) distance between the plan and the set of points
    */
	double& operator()(std::vector<double> p)
	{
		double result = 0.0;
		CVector3 pos;
		for (unsigned int i=0; i<points_->size(); i++) {
			pos = (*points_)[i];
			result += pow(abs(p[0]*pos[0] + p[1]*pos[1] + p[2]*pos[2] + p[3]),2)/(pow(p[0],2) + pow(p[1],2) + pow(p[2],2) + 1.0e-10);
		}
		return result;
	}
};

#endif

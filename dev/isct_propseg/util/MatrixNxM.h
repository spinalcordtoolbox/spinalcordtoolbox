/****************************************************************************
 * Fichier:     Matrice.h
 * Auteurs:     Benjamin De Leener
 * Date:        03/03/2012
 * Modifié:		08/03/2013
 * Description: Definition et Implementation de la classe generique Matrice
 ****************************************************************************/

#ifndef _MATRICE_H_
#define _MATRICE_H_

#include <iostream>
#include <cmath>
#include <alglib/ap.h>
using namespace ap;
using namespace std;


class Matrice
{
public:
	Matrice();
	Matrice(unsigned int lignes, unsigned int colonnes);
    Matrice(real_2d_array m, unsigned int lignes, unsigned int colonnes, bool newMatrice=false);
	Matrice(const Matrice& uneMatrice);
	~Matrice();

	unsigned int getNombreLignes() { return lignes_; };
	unsigned int getNombreColonnes() { return colonnes_; };
	real_2d_array& getMatrice() { return matrice_; };

	double& operator()(unsigned int ligne, unsigned int colonne);
	void operator=(const Matrice& uneMatrice);
	Matrice operator*(const Matrice& uneMatrice);
	Matrice operator+(const Matrice& uneMatrice);
	Matrice operator-(const Matrice& uneMatrice);
	Matrice operator/(double valeur);
	Matrice transpose();
	double norm(); // si vecteur!!

	friend ostream& operator<< (ostream& stream, const Matrice& uneMatrice);
	friend istream& operator>> (istream& stream, Matrice& uneMatrice);
    
    Matrice pinv(double tol=1e-15);

private:
	real_2d_array matrice_;
	unsigned int lignes_;
	unsigned int colonnes_;
};

#endif
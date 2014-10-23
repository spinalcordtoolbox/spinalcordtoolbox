package pottslab;

/**
 * @author Martin Storath
 *
 */

public class JavaTools {

    public static int OR_HORIZONTAL = 1,
	    OR_VERTICAL = 2,
	    OR_DIAGONAL = 3,
	    OR_ANTIDIAGONAL = 4;



    /**
     * minimization of univariate scalar valued Potts functional
     * @param f
     * @param gamma
     * @param weights
     * @return
     */
    public static double[] minL2Potts(double[] f, double gamma, double[] weights) {
	PLVector[] plVec = PLVector.array1DToVector1D(f); 
	L2Potts pottsCore = new L2Potts(plVec, weights, gamma);
	pottsCore.call();
	return PLVector.vector1DToArray1D(plVec);
    }

    /**
     * minimization of univariate vector-valued Potts functional
     * @param f
     * @param gamma
     * @param weights
     * @return
     */
    public static double[][] minL2Potts(double[][] f, double gamma, double[] weights) {
	PLVector[] plVec = PLVector.array2DToVector1D(f); 
	L2Potts pottsCore = new L2Potts(plVec, weights, gamma);
	pottsCore.call();
	return PLVector.vector1DToArray2D(plVec);
    }

    /**
     * Minimization of univariate L2-Potts along indicated orientation
     * @param img
     * @param gamma
     * @param weights
     * @param orientation
     * @return
     */
    public static PLImage minL2PottsOrientation(PLImage img, double gamma, double[][] weights, String orientation) {
	PLProcessor proc = new PLProcessor();
	proc.setMultiThreaded(true);
	proc.setGamma(gamma);
	proc.set(img, weights);

	switch (orientation) {
	case "horizontal":
	    proc.applyHorizontally();
	    break;
	case "vertical":
	    proc.applyVertically();
	    break;
	case "diagonal":
	    proc.applyDiag();
	    break;
	case "antidiagonal":
	    proc.applyAntiDiag();
	    break;

	}
	return img;
    }

    /**
     * ADMM strategy to the scalar-valued 2D Potts problem anisotropic neighborhood (4-connected)
     * @param f
     * @param gamma
     * @param weights
     * @param muInit
     * @param muStep
     * @param stopTol
     * @param verbose
     * @return
     */
    public static PLImage minL2PottsADMM4(PLImage img, double gamma, double[][] weights, double muInit, double muStep, double stopTol,  boolean verbose, boolean multiThreaded, boolean useADMM) {
	// init
	int m = img.mRow;
	int n = img.mCol;
	int l = img.mLen; 
	PLImage u = PLImage.zeros(m,n,l);
	PLImage v = img.copy();
	PLImage lam = PLImage.zeros(m,n,l);
	PLImage temp = PLImage.zeros(m,n,l);
	double[][] weightsPrime = new double[m][n];
	double error = Double.POSITIVE_INFINITY;
	double mu = muInit;
	double gammaPrime;
	int nIter = 0;
	PLProcessor proc = new PLProcessor();
	proc.setMultiThreaded(multiThreaded);
	double fNorm = img.normQuad();

	// a shortcut
	if (fNorm == 0) {
	    return img;
	}

	// main loop
	while (error >= stopTol * fNorm) {
	    // set Potts parameters
	    gammaPrime = 2 * gamma; 

	    // set weights
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++){ 
		weightsPrime[i][j] = weights[i][j] + mu;
	    }

	    // solve horizontal univariate Potts problems 
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++) {
		u.get(i, j).set(k, (img.get(i, j).get(k) * weights[i][j] + v.get(i, j).get(k) * mu - lam.get(i, j).get(k)) / weightsPrime[i][j]);
	    }
	    proc.set(u, weightsPrime);
	    proc.setGamma(gammaPrime);
	    proc.applyHorizontally();

	    // solve vertical univariate Potts problems 
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++) {
		v.get(i, j).set(k, (img.get(i, j).get(k) * weights[i][j] + u.get(i, j).get(k) * mu + lam.get(i, j).get(k)) / weightsPrime[i][j]);
	    }
	    proc.set(v, weightsPrime);
	    proc.setGamma(gammaPrime);
	    proc.applyVertically();

	    // update Lagrange multiplier and calculate difference between u and v
	    error = 0;
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++){
		temp.get(i, j).set(k, u.get(i, j).get(k) - v.get(i, j).get(k));
		if (useADMM) {
		    lam.get(i, j).set(k, lam.get(i, j).get(k) + temp.get(i, j).get(k) * mu); // update Lagrange multiplier
		}
		error += Math.pow(temp.get(i, j).get(k), 2); // compute error
	    }
	    // update coupling
	    mu *= muStep;

	    // count iterations
	    nIter++;

	    // show some information
	    if (verbose) {
		System.out.print("*");
		if (nIter % 50 == 0) {
		    System.out.print("\n");
		}
	    }
	}

	// show some information
	if (verbose) {
	    System.out.println("\n Total number of iterations " + nIter + "\n");
	}
	return u;
    }



    /**
     * ADMM strategy to the scalar-valued 2D Potts problem with near-isotropic neighborhood (8-connected)
     * @param img
     * @param gamma
     * @param weights
     * @param muInit
     * @param muStep
     * @param stopTol
     * @param verbose
     * @return
     */
    public static PLImage minL2PottsADMM8(PLImage img, double gamma, double[][] weights, double muInit, double muStep, double stopTol,  boolean verbose, boolean multiThreaded, boolean useADMM, double[] omega) {
	// init image dimensions
	int m = img.mRow;
	int n = img.mCol;
	int l = img.mLen;
	// init ADMM variables
	PLImage u = PLImage.zeros(m,n,l);
	PLImage v = img.copy();
	PLImage w = img.copy();
	PLImage z = img.copy();
	PLImage lam1 = PLImage.zeros(m,n,l);
	PLImage lam2 = PLImage.zeros(m,n,l);
	PLImage lam3 = PLImage.zeros(m,n,l);
	PLImage lam4 = PLImage.zeros(m,n,l);
	PLImage lam5 = PLImage.zeros(m,n,l);
	PLImage lam6 = PLImage.zeros(m,n,l);
	// init modified weight vector
	double[][] weightsPrime = new double[m][n];
	// init coupling
	double mu = muInit;
	// auxiliary variables
	double gammaPrimeC, gammaPrimeD;
	// set neighborhood weights
	double omegaC = omega[0];
	double omegaD = omega[1];
	// init up the Potts processer
	PLProcessor proc = new PLProcessor();
	proc.setMultiThreaded(multiThreaded);
	// compute the norm of the input image
	double fNorm = img.normQuad();
	if (fNorm == 0) {
	    return img;
	}
	double error = Double.POSITIVE_INFINITY;
	// init number of iterations 
	int nIter = 0;

	// main ADMM iteration
	while (error >= stopTol * fNorm) {
	    // set jump penalty
	    gammaPrimeC = 4.0 * omegaC * gamma;
	    gammaPrimeD = 4.0 * omegaD * gamma;

	    // set weights
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++){ 
		weightsPrime[i][j] = weights[i][j] + 6 * mu;
	    }

	    // solve univariate Potts problems horizontally
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++){
		u.get(i, j).set(k, ( img.get(i, j).get(k) * weights[i][j] + 2 * mu * (w.get(i, j).get(k) + v.get(i, j).get(k) + z.get(i, j).get(k)) 
			+ 2 * (-lam1.get(i, j).get(k) - lam2.get(i, j).get(k) - lam3.get(i, j).get(k)) ) / weightsPrime[i][j] );
	    }
	    proc.set(u, weightsPrime);
	    proc.setGamma(gammaPrimeC);
	    proc.applyHorizontally();

	    // solve 1D Potts problems diagonally
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++){
		w.get(i,j).set(k, (img.get(i,j).get(k) * weights[i][j] + 2 * mu * (u.get(i,j).get(k) + v.get(i,j).get(k) + z.get(i,j).get(k)) 
			+ 2 * (lam2.get(i,j).get(k) + lam4.get(i,j).get(k) - lam6.get(i,j).get(k))) / weightsPrime[i][j] );
	    }
	    proc.set(w, weightsPrime);
	    proc.setGamma(gammaPrimeD);
	    proc.applyDiag();

	    // solve 1D Potts problems vertically
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++){
		v.get(i,j).set(k, (img.get(i,j).get(k) * weights[i][j] + 2 * mu * (u.get(i,j).get(k) + w.get(i,j).get(k) + z.get(i,j).get(k)) 
			+ 2 * (lam1.get(i,j).get(k) - lam4.get(i,j).get(k) - lam5.get(i,j).get(k))) / weightsPrime[i][j]);
	    }
	    proc.set(v, weightsPrime);
	    proc.setGamma(gammaPrimeC);
	    proc.applyVertically();

	    // solve 1D Potts problems antidiagonally
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++){
		z.get(i,j).set(k, (img.get(i,j).get(k) * weights[i][j] + 2 *mu * (u.get(i,j).get(k) + w.get(i,j).get(k) + v.get(i,j).get(k)) 
			+ 2 * (lam3.get(i,j).get(k) + lam5.get(i,j).get(k) + lam6.get(i,j).get(k))) / weightsPrime[i][j]);
	    }
	    proc.set(z, weightsPrime);
	    proc.setGamma(gammaPrimeD);
	    proc.applyAntiDiag();

	    // update Lagrange multiplier and calculate difference between u and v
	    error = 0;
	    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++){
		if (useADMM) {
		    lam1.get(i,j).set(k, lam1.get(i,j).get(k) + mu * (u.get(i,j).get(k) - u.get(i,j).get(k)) ); 
		    lam2.get(i,j).set(k, lam2.get(i,j).get(k) + mu * (u.get(i,j).get(k) - v.get(i,j).get(k)) ); 
		    lam3.get(i,j).set(k, lam3.get(i,j).get(k) + mu * (u.get(i,j).get(k) - z.get(i,j).get(k)) ); 
		    lam4.get(i,j).set(k, lam4.get(i,j).get(k) + mu * (v.get(i,j).get(k) - w.get(i,j).get(k)) ); 
		    lam5.get(i,j).set(k, lam5.get(i,j).get(k) + mu * (v.get(i,j).get(k) - z.get(i,j).get(k)) ); 
		    lam6.get(i,j).set(k, lam6.get(i,j).get(k) + mu * (w.get(i,j).get(k) - z.get(i,j).get(k)) );
		}
		error += Math.pow(u.get(i,j).get(k) - v.get(i,j).get(k), 2);
	    }

	    // increase coupling parameter
	    mu *= muStep;

	    // increase iteration counter
	    nIter++;

	    // show some information
	    if (verbose) {
		System.out.print("*");
		if (nIter % 50 == 0) {
		    System.out.print("\n");
		}
	    }
	}
	// show some information
	if (verbose) {
	    System.out.println("\n Total number of iterations " + nIter + "\n");
	}
	return u;
    }
}

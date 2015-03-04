/**
 * 
 */
package pottslab;

import java.util.concurrent.Callable;

/**
 * @author Martin Storath
 *
 */
@SuppressWarnings("rawtypes")
public class L2Potts implements Callable {

    PLVector[] mData;
    double mGamma; 
    double[] mWeights;
    int mExcludedIntervalSize;
    
    public L2Potts (PLVector[] data, double[] weights, double gamma) {
	set(data, weights, gamma);
    }

    /**
     * Set up the data and parameters
     * @param data
     * @param weights
     * @param gamma
     */
    public void set(PLVector[] data, double[] weights, double gamma) {
	mData = data;
	mWeights = weights;
	mGamma = gamma;
	mExcludedIntervalSize = 0;
    }
    
    /**
     * Set up the data and parameters
     * @param data
     * @param weights
     * @param gamma
     */
    public void setExcludedIntervalSize(int excludedIntervalSize) {
	mExcludedIntervalSize = excludedIntervalSize;
    }
    

    /**
     * Solve one-dimensional Potts problem in situ on mData 
     */
    public Object call() {

	int n = mData.length;
	int[] arrJ = new int[n];
	int nVec = mData[0].length();

	double[] arrP = new double[n]; // array of Potts values
	double d = 0, p = 0, dpg = 0; // temporary variables to save some float operation
	PLVector[] m = new PLVector[n + 1]; // cumulative first moments
	double[] s = new double[n + 1]; // cumulative second moments (summed up in 3 dimension)
	double[] w = new double[n + 1]; // cumulative weights

	// precompute cumulative moments
	m[0] = PLVector.zeros(nVec);
	s[0] = 0;
	double wTemp, mTemp, wDiffTemp;
	for (int j = 0; j < n; j++) {
	    wTemp = mWeights[j];
	    m[j + 1] = mData[j].mult(wTemp);
	    m[j + 1].plusAssign(m[j]);
	    s[j + 1] = mData[j].normQuad() * wTemp + s[j];
	    w[j + 1] = w[j] + wTemp;
	}

	// main loop
	for (int r = 1; r <= n; r++) {
	    arrP[r-1] = s[r] - m[r].normQuad() / (w[r]); // set Potts value of constant solution
	    arrJ[r-1] = 0; // set jump location of constant solution
	    for (int l = r - mExcludedIntervalSize; l >= 2 ; l--) {
		// compute squared deviation from mean value d
		mTemp = 0;
		for (int k = 0; k < nVec; k++) {
		    mTemp = mTemp + Math.pow(m[r].get(k) - (m[l - 1].get(k)), 2);
		}
		wDiffTemp = (w[r] - w[l - 1]);
		if (wDiffTemp == 0) {
		    d = 0;
		} else {
		    d = s[r] - s[l - 1] - mTemp / wDiffTemp;
		}
		dpg = d + mGamma; // temporary variable
		if (dpg > arrP[r-1]) {
		    // Acceleration: if deviation plus jump penalty is larger than best
		    // Potts functional value, we can break the loop
		    break;
		}
		p = arrP[l - 2] + dpg; // candidate Potts value
		if (p < arrP[r-1]) {
		    arrP[r-1] = p; // set optimal Potts value
		    arrJ[r-1] = l - 1; // set jump location
		}
	    }
	}
	// Reconstruction from best partition
	int r = n;
	int l = arrJ[r-1];
	PLVector mu = PLVector.zeros(nVec);
	while (r > 0) {
	    // compute mean value on interval [l+1, r]
	    for (int k = 0; k < nVec; k++) {
		mu.set(k, (m[r].get(k) - m[l].get(k)) / (w[r] - w[l]));
	    }
	    // set mean value on interval [l+1, r]
	    for (int j = l; j < r; j++) {
		for  (int k = 0; k < mu.length(); k++) { 
		    mData[j].set(k, mu.get(k));
		}
	    }
	    r = l;
	    if (r < 1) break;
	    // go to next jump
	    l = arrJ[r-1]; 
	}
	return mData;
    }

}

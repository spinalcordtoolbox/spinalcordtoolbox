/**
 * 
 */
package pottslab;

/**
 * Data structure for a vector with some frequently used methods  
 * 
 * @author Martin Storath
 *
 */
public class PLVector {

    private double[] mData;

    public PLVector(double data) {
	mData = new double[1];
	mData[0] = data;
    }

    public PLVector(double[] data) {
	mData = data;
    }

    public double get(int idx) {
	return mData[idx];
    }

    public double[] get() {
	return mData;
    }

    public void set(double[] data) {
	mData = data;
    }

    public void set(int k, double d) {
	mData[k] = d;
    }

    public double norm() {
	return Math.sqrt(this.normQuad());
    }

    public void normalize() {
	double norm = this.norm();
	if (norm == 0) {
	    mData = new double[mData.length];
	} else {
	    for (int i=0; i < mData.length; i++) {
		mData[i] /= norm; 
	    }
	}
    }

    public double normQuad() { 
	double norm = 0;
	for (int i=0; i < mData.length; i++) {
	    norm += Math.pow(mData[i], 2);
	}
	return norm;
    }


    public double sum() {
	double sum = 0;
	for (int i=0; i < mData.length; i++) {
	    sum += mData[i];
	}
	return sum;
    }

    public int length() {
	return mData.length;
    }

    public PLVector mult(double x) {
	double[] data = new double[length()]; 
	for (int i=0; i < mData.length; i++) {
	    data[i] = mData[i] * x;
	}
	return new PLVector(data);
    }

    public PLVector multAssign(double x) {
	for (int i=0; i < mData.length; i++) {
	    mData[i] *= x;
	}
	return this;
    }

    public PLVector divAssign(double x) {
	for (int i=0; i < mData.length; i++) {
	    mData[i] /= x;
	}
	return this;
    }

    public PLVector pow(double p) {
	double[] data = new double[length()]; 
	for (int i=0; i < mData.length; i++) {
	    data[i] = Math.pow(mData[i], p);
	}
	return new PLVector(data);
    }

    public PLVector powAssign(double p) {
	for (int i=0; i < mData.length; i++) {
	    mData[i] = Math.pow(mData[i], p);
	}
	return this;
    }

    public PLVector plus(PLVector x) {
	assert x.length() == this.length();
	double[] data = new double[length()]; 
	for (int i=0; i < mData.length; i++) {
	    data[i] = mData[i] + x.get(i);
	}
	return new PLVector(data);
    }

    public PLVector plusAssign(PLVector x) {
	assert x.length() == this.length();
	for (int i=0; i < mData.length; i++) {
	    mData[i] += x.get(i);
	}
	return this;
    }

    public PLVector minus(PLVector x) {
	assert x.length() == this.length();
	double[] data = new double[length()]; 
	for (int i=0; i < mData.length; i++) {
	    data[i] = mData[i] - x.get(i);
	}
	return new PLVector(data);
    }

    public PLVector minusAssign(PLVector x) {
	assert x.length() == this.length();
	for (int i=0; i < mData.length; i++) {
	    mData[i] -= x.get(i);
	}
	return this;
    }

    /**
     * Deep copy
     * @return
     */
    public PLVector copy() {
	double datac[] = new double[this.length()];
	for (int i=0; i < mData.length; i++) {
	    datac[i] = mData[i];
	}
	return new PLVector(datac);
    }

    public static PLVector zeros(int length) {
	return new PLVector(new double[length]);
    }

    public static PLVector[] array1DToVector1D(double[] arr) {
	int m = arr.length;
	PLVector[] vec = new PLVector[m];
	for (int i = 0; i < m; i++)  {
	    vec[i] =  new PLVector(arr[i]); 
	}
	return vec;
    }

    public static PLVector[] array2DToVector1D(double[][] arr) {
	int m = arr.length;
	PLVector[] vec = new PLVector[m];
	for (int i = 0; i < m; i++) {
	    vec[i] =  new PLVector(arr[i]); 
	}
	return vec;
    }

    public static PLVector[][] array3DToVector2D(double[][][] arr) {
	int m = arr.length;
	int n = arr[0].length;
	PLVector[][] vec = new PLVector[m][n];
	for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)  {
	    vec[i][j] =  new PLVector(arr[i][j]); 
	}
	return vec;
    }

    public static PLVector[][] array2DToVector2D(double[][] arr) {
	int m = arr.length;
	int n = arr[0].length;
	PLVector[][] vec = new PLVector[m][n];
	for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)  {
	    vec[i][j] =  new PLVector(arr[i][j]); 
	}
	return vec;
    }


    public static double[] vector1DToArray1D(PLVector[] vec) {
	int m = vec.length;
	double[] arr = new double[m];
	for (int i = 0; i < m; i++)  {
	    arr[i]=  vec[i].get(0); 
	}
	return arr;
    }

    public static double[][] vector1DToArray2D(PLVector[] vec) {
	int m = vec.length;
	int l = vec[0].length();
	double[][] arr = new double[m][l];
	for (int i = 0; i < m; i++) for (int k = 0; k < l; k++)  {
	    arr[i][k] =  vec[i].get(k); 
	}
	return arr;
    }

    public static double[][][] vector2DToArray3D(PLVector[][] vec) {
	int m = vec.length;
	int n = vec[0].length;
	int l = vec[0][0].length();
	double[][][] arr = new double[m][n][l];
	for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) for (int k = 0; k < l; k++)  {
	    arr[i][j][k] =  vec[i][j].get(k); 
	}
	return arr;
    }


    public static double[][] vector2DToArray2D(PLVector[][] vec) {
	int m = vec.length;
	int n = vec[0].length;
	double[][] arr = new double[m][n];
	for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)  {
	    arr[i][j] =  vec[i][j].get(0); 
	}
	return arr;
    }

    public String toString() {
	String str = "(";
	int l = this.length();
	for (int i=0; i < l; i++) {
	    str += mData[i];
	    if (i == l-1) {
		str += ")";
	    } else {
		str += ", ";
	    }
	}
	return str; 
    }

}

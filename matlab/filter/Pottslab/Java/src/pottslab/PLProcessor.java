/**
 * 
 */
package pottslab;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


/**
 * @author Martin Storath
 *
 */

@SuppressWarnings("rawtypes")
public class PLProcessor {

    public PLImage mImg;
    public double[][] mWeights;
    public int mCol, mRow;
    public int mProcessors;
    public double mGamma;
    public ExecutorService eservice;

    public PLProcessor() {
	init();
    }

    public PLProcessor(PLImage img, double[][] weights) {
	init();
	set(img, weights);
    }

    protected void init() {
	setMultiThreaded(true);
    }

    public void set(PLImage img, double[][] weights) {
	mImg = img;
	mWeights = weights;
	mCol = img.mCol;
	mRow = img.mRow;
    }

    /**
     * Enable/disable parallel execution
     * @param bool
     */
    public void setMultiThreaded(boolean bool) {
	if (bool) {
	    mProcessors = Runtime.getRuntime().availableProcessors();
	} else {
	    mProcessors = 1;
	}
	eservice = Executors.newFixedThreadPool(mProcessors);
    }

    public void setGamma(double gamma) {
	mGamma = gamma;
    }


    /**
     * Applies Potts algorithm in horizontal direction
     * 
     * @param gamma
     */
    public void  applyHorizontally() {
	List<Future> futuresList = new ArrayList<Future>(mRow);
	PLVector[] array; 
	double[] weightsArr;
	for (int i = 0; i < mRow; i++) {
	    // set up arrays
	    array = new PLVector[mCol]; 
	    weightsArr = new double[mCol];
	    for (int j = 0; j < mCol; j++) {
		array[j] = mImg.get(i, j);
		weightsArr[j] = mWeights[i][j];
	    }
	    // add to task list
	    futuresList.add(eservice.submit(createTask(array, weightsArr)));
	}
	for(Future future:futuresList) {
	    try {
		future.get();
	    }
	    catch (InterruptedException e) {}
	    catch (ExecutionException e) {}
	}
    }

    /**
     * Applies Potts algorithm in vertical direction
     * 
     * @param gamma
     */
    public void  applyVertically() {
	List<Future> futuresList = new ArrayList<Future>(mCol);
	PLVector[] array;
	double[] weightsArr;
	for (int j = 0; j < mCol; j++) {
	    // set up arrays
	    array = new PLVector[mRow]; 
	    weightsArr = new double[mRow];
	    for (int i = 0; i < mRow; i++) {
		array[i] = mImg.get(i, j);
		weightsArr[i] = mWeights[i][j];
	    }
	    futuresList.add(eservice.submit(createTask(array, weightsArr)));
	}

	for(Future future:futuresList) {
	    try {
		future.get();
	    }
	    catch (InterruptedException e) {}
	    catch (ExecutionException e) {}
	}
    }

    /**
     * Applies Potts algorithm in diagonal direction (top left to bottom right)
     * 
     * @param gamma
     */
    public void  applyDiag() {
	int row, col;
	List<Future> futuresList = new ArrayList<Future>(mRow+mCol-1);
	PLVector[] array;
	double[] weightsArr;

	for (int k = 0; k < (mCol); k++) {

	    int sDiag = Math.min(mRow, mCol - k);

	    array = new PLVector[sDiag];
	    weightsArr = new double[sDiag];
	    for (int j = 0; j < sDiag; j++) {

		row = j;
		col = j + k;
		array[j] = mImg.get(row, col);
		weightsArr[j] = mWeights[row][col];
	    }
	    futuresList.add(eservice.submit(createTask(array, weightsArr)));
	}
	for (int k = mRow-1; k > 0; k--) {
	    int sDiag = Math.min(mRow - k, mCol);
	    array = new PLVector[sDiag];
	    weightsArr = new double[sDiag];
	    for (int j = 0; j < sDiag; j++) {
		row = j + k;
		col = j;
		array[j] = mImg.get(row, col);
		weightsArr[j] = mWeights[row][col];
	    }
	    futuresList.add(eservice.submit(createTask(array, weightsArr)));
	}
	for(Future future:futuresList) {
	    try {
		future.get();
	    }
	    catch (InterruptedException e) {}
	    catch (ExecutionException e) {}
	}
    }

    /**
     * Applies Potts algorithm in antidiagonal direction (top right to bottom left)
     * 
     * @param gamma
     */
    public void applyAntiDiag() {
	int row, col;
	List<Future> futuresList = new ArrayList<Future>(mRow+mCol-1);
	PLVector[] array;
	double[] weightsArr;
	for (int k = 0; k < (mCol); k++) {
	    int sDiag = Math.min(mRow, mCol - k);
	    array = new PLVector[sDiag];
	    weightsArr = new double[sDiag];
	    for (int j = 0; j < sDiag; j++) {
		row = j;
		col = mCol - 1 - (j + k);
		array[j] = mImg.get(row, col);
		weightsArr[j] = mWeights[row][col];
	    }
	    futuresList.add(eservice.submit(createTask(array, weightsArr)));
	}
	for (int k = mRow-1; k > 0; k--) {
	    int sDiag = Math.min(mRow - k, mCol);
	    array = new PLVector[sDiag];
	    weightsArr = new double[sDiag];
	    for (int j = 0; j < sDiag; j++) {
		row = (j + k);
		col = mCol - 1 - j;
		array[j] = mImg.get(row, col);
		weightsArr[j] = mWeights[row][col];
	    }
	    futuresList.add(eservice.submit(createTask(array, weightsArr)));
	}
	for(Future future:futuresList) {
	    try {
		future.get();
	    }
	    catch (InterruptedException e) {}
	    catch (ExecutionException e) {}
	}
    }

    /**
     * Applies univariate Potts algorithm into indicated direction
     * 
     * @param gamma
     */
    public void applyToDirection(int[] direction) {
	int pCol = direction[0];
	int pRow = direction[1];
	if (pRow == 0) {
	    this.applyHorizontally();
	    return;
	}
	if (pCol == 0) {
	    this.applyVertically();
	    return;
	}
	int row, col;
	List<Future> futuresList = new LinkedList<Future>();
	PLVector[] array;
	double[] weightsArr;
	int sDiag;
	for(int rOffset = 0; rOffset < pRow ; rOffset++ ) {
	    for(int cOffset = rOffset; cOffset < mCol; cOffset++ ) {
		sDiag = Math.min( (mCol - cOffset - 1)/pCol, (mRow - rOffset-1)/pRow ) + 1;  
		array = new PLVector[sDiag];
		weightsArr = new double[sDiag];
		row = rOffset;
		col = cOffset;
		for (int j = 0; j < sDiag; j++) {
		    array[j] = mImg.get(row, col);
		    weightsArr[j] = mWeights[row][col];
		    row += pRow;
		    col += pCol;
		}
		//futuresList.add(eservice.submit(createTask(array, weightsArr)));
		Callable l2Potts = createTask(array, weightsArr);
		futuresList.add(eservice.submit(l2Potts));
	    }
	}

	for(int cOffset = 0; cOffset < pCol ; cOffset++ ) {
	    for(int rOffset = cOffset; rOffset < mRow; rOffset++ ) {
		//int cOffset = 0;
		sDiag = Math.min( (mCol - cOffset - 1)/pCol, (mRow - rOffset-1)/pRow ) + 1;  
		array = new PLVector[sDiag];
		weightsArr = new double[sDiag];
		row = rOffset;
		col = cOffset;
		for (int j = 0; j < sDiag; j++) {
		    array[j] = mImg.get(row, col);
		    weightsArr[j] = mWeights[row][col];
		    row += pRow;
		    col += pCol;
		}
		//futuresList.add(eservice.submit(createTask(array, weightsArr)));
		Callable l2Potts = createTask(array, weightsArr);
		futuresList.add(eservice.submit(l2Potts));
	    }
	}

	for(Future future:futuresList) {
	    try {
		future.get();
	    }
	    catch (InterruptedException e) {}
	    catch (ExecutionException e) {}
	}
    }

    public Callable<?> createTask(PLVector[] array, double[] weightsArr) {
	return new L2Potts(array, weightsArr, mGamma);
    }

}

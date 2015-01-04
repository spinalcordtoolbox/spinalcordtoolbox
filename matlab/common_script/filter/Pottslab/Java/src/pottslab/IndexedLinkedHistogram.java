package pottslab;

import java.util.ArrayList;

/**
 * @author martinstorath
 * 
 */
public class IndexedLinkedHistogram {

    ArrayList<HistNode> originalOrder; // stores the nodes in order of the incoming insertions
    double[] weights; // stores all weights

    HistNode first, last, median; // list pointers 
    HistNode firstTemp, lastTemp; // temporary list pointers 

    double totalDeviation; // stores the deviation from the median
    double weightAboveMedian, weightBelowMedian; 
    double totalWeight;

    public double[] currentMedians;
    /**
     * Constructor
     * @param initialSize
     * the expected list size
     */
    public IndexedLinkedHistogram(double[] weights) {
	// initialize the array which indexes the nodes in their original order
	originalOrder = new ArrayList<HistNode>(weights.length);
	// init weights
	this.weights = weights;
    }

    /**
     * Node inner class
     */
    class HistNode {
	// the value
	double value; 
	double weight, weightTemp;
	int count, countTemp;
	// pointer to the next and previous elements
	HistNode next, prev;
	// temporary pointers to the next and previous elements
	HistNode nextTemp, prevTemp; 

	/**
	 * Node constructor
	 * @param element
	 */
	HistNode(double element, double weight) {
	    this.value = element;
	    this.weight = weight;
	    this.count = 1;
	}

	void addWeight(double weight) {
	    this.weight += weight;
	    count++;
	}

	void removeWeightTemp(double weight) {
	    this.weightTemp -= weight;
	    countTemp--;
	    if (countTemp < 0) {
		throw new RuntimeException("Error in List.");
	    }
	    if (weightTemp < 0) {
		weightTemp = 0;
	    }
	}

	/**
	 * Reset all temporary values to the non-temporary ones
	 */
	void resetTemp() {
	    nextTemp = next;
	    prevTemp = prev;
	    weightTemp = weight;
	    countTemp = count;
	}
    }

    /**
     * inserts an element sorted by value in ascending order from first to last
     */
    public void insertSorted(double elem) {
	double weight = weights[this.size()];
	HistNode pivot = null; 
	HistNode iterator = null;
	if (first == null) { // empty list
	    pivot = new HistNode(elem, weight);
	    first = pivot;
	    last = first;
	    median = first;
	    totalDeviation = 0;
	    weightAboveMedian = 0;
	    weightBelowMedian = 0;
	    totalWeight = weight;
	} else { // non-empty list
	    iterator = first;
	    while (iterator != null) {
		if (iterator.value >= elem) {
		    break;
		}
		iterator.resetTemp();
		iterator = iterator.next;
	    }
	    // insert node at right place or add weight to existing node
	    if (iterator != null) {
		if (iterator.value == elem) {
		    pivot = iterator;
		    pivot.addWeight(weight);
		} else {
		    pivot = new HistNode(elem, weight);
		    insertBefore(iterator, pivot);
		}
	    } else {
		pivot = new HistNode(elem, weight);
		insertAfter(last, pivot);
	    }
	    // continue loop to reset temporary pointers
	    while (iterator != null) {
		iterator.resetTemp();
		iterator = iterator.next;
	    }
	    // add weight to total weight
	    totalWeight += weight;
	} 
	// add the pivot node to the original order (allows removeTemp in O(1)!)
	originalOrder.add(pivot);
	// reset temporary first and last

	
	/*  OBSOLETE: leads to loss of significance
	// update weights
	if (elem > median.value) {
	    weightAboveMedian += weight;
	} else if (elem < median.value) {
	    weightBelowMedian += weight;
	}
	 shifting median and updating deviations
		totalDeviation += Math.abs(median.value - elem) * weight; // new element increases deviation
		double oldMedian = median.value; // store old median
		double medDiff; // difference between old and new median
		// if weight above the median is to large, execute the following loop
		while ((weightAboveMedian > totalWeight/2) && (median.next != null)) {
		    // the null check is important if loss of significance occurs
		    weightBelowMedian += median.weight; // update weight above
		    median = median.next; // shift median
		    // update median deviation
		    medDiff = Math.abs(oldMedian - median.value); // difference between old and new median
		    totalDeviation -= medDiff * Math.abs(weightBelowMedian - weightAboveMedian); // update deviation
		    weightAboveMedian -= median.weight; // update weight below
		}
		// if the weight below the median is to large, execute the following loop
		while ((weightBelowMedian > totalWeight/2)  && (median.prev != null)) {
		    // the null check is important if loss of significance occurs
		    weightAboveMedian += median.weight;
		    // shift median to the left
		    median = median.prev;
		    // update median deviation
		    medDiff = Math.abs(oldMedian - median.value);
		    totalDeviation -= medDiff * Math.abs(weightAboveMedian - weightBelowMedian);
		    weightBelowMedian -= median.weight;
		}
	*/
	
	// determine median
	iterator = first;
	double wbm = 0;
	double twh = totalWeight/2.0;
	while (iterator != null) {
	    wbm += iterator.weight;
	    if (wbm > twh)  {
		median = iterator.prev;
		break;
	    }
	    iterator = iterator.next;
	}
	if (median == null) {
	    median = first;
	}
	// determine weight below and above
	weightAboveMedian = 0;
	weightBelowMedian = 0;
	totalDeviation = 0;
	iterator = first;
	while (iterator != null) {
	    if (iterator.value < median.value) {
		weightBelowMedian += iterator.weight;
	    } else if (iterator.value > median.value) {
		weightAboveMedian += iterator.weight;
	    }
	    totalDeviation += Math.abs(median.value - iterator.value) * iterator.weight;
	    iterator = iterator.next;
	}
    }

    /**
     * 
     * @return
     * the number of elements currently in the list, equals the (outer) r-loop index
     */
    public int size() {
	return originalOrder.size();
    }

    /**
     * inserts the newNode after the pivot
     * @param pivot
     * @param newNode
     */
    private void insertAfter(HistNode pivot, HistNode newNode) {
	newNode.next = pivot.next;
	pivot.next = newNode;
	newNode.prev = pivot;
	if (pivot == last) {
	    last = newNode;
	} else {
	    newNode.next.prev = newNode;
	    newNode.next.resetTemp();
	}
	// reset temporary pointers
	newNode.prev.resetTemp();
	newNode.resetTemp();	
    }

    /**
     * inserts the newNode before the pivot
     * @param pivot
     * @param newNode
     */
    private void insertBefore(HistNode pivot, HistNode newNode) {
	newNode.prev = pivot.prev;
	pivot.prev = newNode;
	newNode.next = pivot;
	if (pivot == first) {
	    first = newNode;
	} else {
	    newNode.prev.next = newNode;
	    newNode.prev.resetTemp();
	}
	// reset temporary pointers
	newNode.resetTemp();	
	newNode.next.resetTemp();
    }

    /**
     * Computes the deviations from the median of every connected interval in the original data vector (stored in indices)
     * @return
     */
    public double[] computeDeviations() {
	// init the output array, i.e., deviation between d_[l,r]
	double[] deviationsArray = new double[this.size()];
	// init the corresponding median array, i.e., deviation between d_[l,r]
	currentMedians = new double[this.size()];
	// init all temporary values
	firstTemp = first;
	lastTemp = last;
	HistNode medianTemp = median;
	double deviationTemp = totalDeviation;
	double weightAboveMedianTemp = weightAboveMedian;
	double weightBelowMedianTemp = weightBelowMedian;
	double totalWeightTemp = totalWeight;
	double medDiff, oldMedian; 
	for (int l = 1; l < this.size(); l++) {
	    // set the deviation array entry to the temporary distance
	    deviationsArray[l-1] = deviationTemp;
	    // set the median array entry to the temporary median
	    currentMedians[l-1] = medianTemp.value;
	    // pointer to the node to be removed
	    HistNode nodeToRemove = originalOrder.get(l-1); 
	    // removes the node temporarily
	    double weightToRemove = weights[l-1];
	    // remove weight from node
	    nodeToRemove.removeWeightTemp(weightToRemove);
	    // update deviation
	    deviationTemp -= weightToRemove * Math.abs(nodeToRemove.value - medianTemp.value);
	    // update total weight
	    totalWeightTemp -= weightToRemove;
	    // update weights above, below
	    if (nodeToRemove.value > medianTemp.value) {
		weightAboveMedianTemp -= weightToRemove;
	    } else if (nodeToRemove.value < medianTemp.value) {
		weightBelowMedianTemp -= weightToRemove;
	    } 
	    
	    
	    // if weights are unbalanced, the median pointer has to be shifted and the deviations require updates
	    // if weight above is too large, shift right
	    double twth = totalWeightTemp/2.0;
	    while ((weightAboveMedianTemp > twth) && (medianTemp.nextTemp != null)) {
		oldMedian = medianTemp.value;
		weightBelowMedianTemp += medianTemp.weightTemp; // update weight below
		medianTemp = medianTemp.nextTemp;  
		// calculate difference between old and new median
		medDiff = Math.abs(oldMedian - medianTemp.value);
		// decrease deviation according median and weight differences
		deviationTemp -= medDiff * Math.abs(weightBelowMedianTemp - weightAboveMedianTemp);
		weightAboveMedianTemp -= medianTemp.weightTemp; // update weight above
	    }
	    // if weight below is too large, shift right
	    while ((weightBelowMedianTemp > twth) && (medianTemp.prevTemp != null)) {
		oldMedian = medianTemp.value; // store old median
		weightAboveMedianTemp += medianTemp.weightTemp; // update weight above
		medianTemp = medianTemp.prevTemp; // shift median to left
		// calculate difference between old and new median
		medDiff = Math.abs(oldMedian - medianTemp.value);
		// decrease deviation according median and weight differences
		deviationTemp -= medDiff * Math.abs(weightAboveMedianTemp - weightBelowMedianTemp); 
		weightBelowMedianTemp -= medianTemp.weightTemp; // update weight below
	    }

	    // remove node temporary, if it contains no more weights
	    if (nodeToRemove.countTemp == 0) {
		removeTemp(nodeToRemove);
	    }
	}
	return deviationsArray;
    }

    /**
     * remove temporarily a HistNode from the histogram
     * 
     * @param index
     */
    private void removeTemp(HistNode nodeToRemove) {
	// remove node temporarily
	if (nodeToRemove == lastTemp) {
	    lastTemp = lastTemp.prevTemp;
	    lastTemp.nextTemp = null;
	} else if (nodeToRemove == firstTemp) {
	    firstTemp = firstTemp.nextTemp;
	    firstTemp.prevTemp = null;
	} else {
	    nodeToRemove.nextTemp.prevTemp = nodeToRemove.prevTemp;
	    nodeToRemove.prevTemp.nextTemp = nodeToRemove.nextTemp;
	}

    }
}

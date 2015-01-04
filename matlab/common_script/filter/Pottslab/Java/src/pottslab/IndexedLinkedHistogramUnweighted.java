package pottslab;

import java.util.ArrayList;

/**
 * @author martinstorath
 * 
 */
public class IndexedLinkedHistogramUnweighted {

    private ArrayList<HistNode> originalOrder; // stores the nodes in order of the incoming insertions

    private HistNode first, last, median; // list pointers 
    private HistNode firstTemp, lastTemp; // temporary list pointers

    private double currentDeviation; // stores the deviation from the median
    
    /**
     * Constructor
     * @param initialSize
     * the expected list size
     */
    public IndexedLinkedHistogramUnweighted(int initialSize) {
	// initialize the array which indexes the nodes in their original order
	originalOrder = new ArrayList<HistNode>(initialSize);
    }

    /**
     * Node inner class
     */
    class HistNode {
	// the value
	double value; 
	// pointer to the next and previous elements
	HistNode next, prev;
	// temporary pointers to the next and previous elements
	HistNode nextTemp, prevTemp; 

	/**
	 * Node constructor
	 * @param element
	 */
	HistNode(double element) {
	    this.value = element;
	}
	
	void resetTemp() {
	    nextTemp = next;
	    prevTemp = prev;
	}
    }

    /**
     * inserts an element sorted in ascending order from first to last
     */
    public void insertSorted(double elem) {
	HistNode node = new HistNode(elem);
	if (first == null) { // empty list
	    first = node;
	    last = node;
	    median = node;
	    currentDeviation = 0;
	} else { // non empty list
	    HistNode iterator = first;
	    // isEven is true, if the size of the list is even
	    boolean isEven = (this.size() % 2 == 0);
	    // for speed-up, we insert and reset the temporary pointers in one sweep
	    HistNode pivot = null;
	    while (iterator != null) {
		// find pivot element
		if ((pivot == null) && (iterator.value > node.value)) {
		   pivot = iterator; 
		}
		// set temporary pointers
		iterator.prevTemp = iterator.prev;
		iterator.nextTemp = iterator.next; 
		iterator = iterator.next;
	    }
	    // insert node at right place 
	    if (pivot != null) {
		insertBefore(pivot, node);
	    } else { // iterator is null if elem is greater than all elements in the list 
		insertAfter(last, node);
	    }
	    // update deviation from (old) median  
	    currentDeviation = currentDeviation + Math.abs(node.value - median.value);
	    HistNode medianOld = median; 	    // store old median
	    // update median and distances
	    boolean insertAboveMedian =  median.value <= node.value;
	    if (!insertAboveMedian && isEven) {
		median = median.prev;
		currentDeviation = currentDeviation - medianOld.value + median.value;
	    } else if (insertAboveMedian && !isEven) {
		median = median.next;
	    }
	}
	// add the node to the original order (allows removeTemp in O(1)!)
	originalOrder.add(node);
	// set temporary first and last
	firstTemp = first;
	lastTemp = last;
    }

    /**
     * 
     * @return
     * the number of elements currently in the list
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
	// Initialization
	double[] deviationsArray = new double[this.size()];
	HistNode medianTemp = median;
	double deviationTemp = currentDeviation;
	// reset all temporary values and pointers to the non-temporary ones
	//resetTemp();
	for (int l = 1; l < this.size(); l++) {
	    // set the deviation to the temporary distance
	    deviationsArray[l-1] = deviationTemp;
	    // pointer to the node to be removed
	    HistNode nodeToRemove = originalOrder.get(l-1); 
	    // removes the node temporarily
	    removeTemp(nodeToRemove);
	    // update median and distances
	    deviationTemp = deviationTemp - Math.abs(medianTemp.value - nodeToRemove.value);
	    /* the correct update of medTemp and deviationTemp requires to differentiate
	    between even and odd temporary length and if removedNode was
	    above or below medianTemp */
	    double medianValueOld = medianTemp.value;
	    boolean isEven = ((this.size() - l + 1) % 2 == 0);
	    if (isEven) {  
		if ((medianValueOld < nodeToRemove.value) || (medianTemp == nodeToRemove)) {
		    medianTemp = medianTemp.prevTemp;
		    deviationTemp = deviationTemp - medianValueOld + medianTemp.value;
		}
	    } else if ((nodeToRemove.value <= medianValueOld) ) {
		medianTemp = medianTemp.nextTemp; 
	    }
	}
	return deviationsArray;
    }

    /**
     * remove temporarily
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
    
    public String printList(boolean temp) {
	String str = "";
	HistNode iterator = first;
	while (iterator != null) { 
		// find pivot element
		str = str + ", " + iterator.value;
		if (temp) {
			iterator = iterator.nextTemp;
		} else {
			iterator = iterator.next;
		}
	    }
	return str;
    }
    
    public static void main(String[] args) {
	IndexedLinkedHistogramUnweighted list = new IndexedLinkedHistogramUnweighted(10);
	list.insertSorted(1);
	list.insertSorted(0);
	list.insertSorted(0);
	list.insertSorted(0);
	list.computeDeviations();
	System.out.println(list.printList(true));
	System.out.println(list.printList(false));

	list.insertSorted(2);

	System.out.println(list.printList(true));
	System.out.println(list.printList(false));
    }
    
}

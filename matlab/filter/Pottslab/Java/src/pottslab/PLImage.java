/**
 * 
 */
package pottslab;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;


/**
 * A structure for a vector valued image
 * 
 * @author Martin Storath
 *
 */
public class PLImage {

    public  PLVector[][] mData;
    public int mRow, mCol, mLen;

    public PLImage(PLVector[][] vecArr) {
	mRow = vecArr.length;
	mCol = vecArr[0].length;
	mLen = vecArr[0][0].length();
	mData = vecArr;
    }


    public PLImage(double[][] img) {
	set(img);
    }

    public PLImage(double[][][] img) {
	set(img);
    }
    
    public void set(double[][] img) {
	mRow = img.length;
	mCol = img[0].length;
	mLen = 1;
	mData = new PLVector[mRow][mCol];
	for (int i = 0; i < mRow; i++) for (int j = 0; j < mCol; j++)  {
	    this.set(i, j, new PLVector(img[i][j])); 
	}
    }
    
    public void set(double[][][] img) {
	mRow = img.length;
	mCol = img[0].length;
	mLen = img[0][0].length;
	mData = new PLVector[mRow][mCol];
	for (int i = 0; i < mRow; i++) for (int j = 0; j < mCol; j++)  {
	    this.set(i, j, new PLVector(img[i][j])); 
	}
    }

    public PLVector get(int i, int j) {
	return mData[i][j];
    }

    public void set(int i, int j, PLVector data) {
	mData[i][j] = data;
    }

    public double[][][] toDouble3D() {
	double[][][] arr = new double[mRow][mCol][mLen];
	for (int i = 0; i < mRow; i++) for (int j = 0; j < mCol; j++) for (int k = 0; k < mLen; k++)  {
	    arr[i][j][k] =  get(i,j).get(k); 
	}
	return arr;
    }

    public double[] toDouble() {
	double[] arr = new double[mRow * mCol * mLen];
	for (int i = 0; i < mRow; i++) for (int j = 0; j < mCol; j++) for (int k = 0; k < mLen; k++)  {
	    arr[mRow * mCol * k + mRow * j + i] =  get(i,j).get(k); 
	}
	return arr;
    }


    public double normQuad() {
	double norm = 0;
	for (int i = 0; i < mRow; i++) for (int j = 0; j < mCol; j++) {
	    norm += get(i,j).normQuad();
	}
	return norm;
    }

    public PLImage copy() {
	PLVector[][] newData = new PLVector[mRow][mCol];
	for (int i = 0; i < mRow; i++) for (int j = 0; j < mCol; j++)   {
	    newData[i][j] = get(i, j).copy();
	}
	return new PLImage(newData);
    }

    public static PLImage zeros(int rows, int cols, int len) {
	return new PLImage(new double[rows][cols][len]);
    }


    public void show() {
	final BufferedImage img = new BufferedImage(mRow, mCol, BufferedImage.TYPE_INT_RGB);
	Graphics2D g = (Graphics2D)img.getGraphics();
	for(int i = 0; i < mRow; i++) {
	    for(int j = 0; j < mCol; j++) {
		float c = (float) Math.min(Math.abs(mData[i][j].get(0)), 1.0);
		g.setColor(new Color(c, c, c));
		g.fillRect(i, j, 1, 1);
	    }
	}

	JFrame frame = new JFrame("Image test");
	//frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	JPanel panel = new JPanel() {
	    @Override
	    protected void paintComponent(Graphics g) {
		Graphics2D g2d = (Graphics2D)g;
		g2d.clearRect(0, 0, getWidth(), getHeight());
		g2d.setRenderingHint(
			RenderingHints.KEY_INTERPOLATION,
			RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		// Or _BICUBIC
		g2d.scale(2, 2);
		g2d.drawImage(img, 0, 0, this);
	    }
	};
	panel.setPreferredSize(new Dimension(mRow*4, mCol*4));
	frame.getContentPane().add(panel);
	frame.pack();
	frame.setVisible(true);
    }

}

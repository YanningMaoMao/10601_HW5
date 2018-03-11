
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CSVReader {
	
	private static final int IMG_HEIGHT = 16;
	private static final int IMG_WIDTH = 8;
	private static final int NUM_PIXELS = IMG_HEIGHT * IMG_WIDTH;
	
	
	/* --------------------   Class Variables   ------------------- */
	

	private String source;
	private String delimiter;
	
	private int nData;
	
	private List<Integer> labels;
	private List<List<Double>> letters;
	
	
	/* ----------------------   Initializer   ---------------------- */
	
	
	public CSVReader(String source, String delimeter) throws FileNotFoundException {
		
		// set class variable
		this.source = source;
		this.delimiter = delimeter;
		
		// initialize the lists
		labels = new ArrayList<>();
		letters = new ArrayList<>();
		
		// initialize the number of letters
		nData = 0;
		
		// open a scanner to scan the file
		Scanner sc = new Scanner(new File(source));
		
		while (sc.hasNext()) {
			
			// read a line
			String line = sc.nextLine();
			String[] row = line.split(delimiter);
			
			// parse the label
			int label = Integer.parseInt(row[0]);
			labels.add(label);
			
			// parse the pixels
			List<Double> pixels = new ArrayList<>();
			for (int i = 1; i < row.length; i ++) {
				pixels.add(Double.parseDouble(row[i]));
			}
			assert(pixels.size() == NUM_PIXELS);
			letters.add(pixels);
			
			// update the number of data
			nData += 1;

		}
		assert(labels.size() == nData);
		assert(letters.size() == nData);
		
		// close the scanner
		sc.close();
		
	}

	
	/* --------------------------   Getters   --------------------------*/
	
	public static int getImgHeight() {
		return IMG_HEIGHT;
	}
	
	public static int getImgWidth() {
		return IMG_WIDTH;
	}
	
	public static int getNumPixels() {
		return NUM_PIXELS;
	}

	public String getSource() {
		return source;
	}
	
	public int getNumberOfData() {
		return nData;
	}
	
	public int getLabel(int letterIdx) {
		assert(0 <= letterIdx && letterIdx < nData);
		return labels.get(letterIdx);
	}
	
	public List<Integer> getLabels() {
		
		List<Integer> labelsCopy = new ArrayList<>();
		for (Integer label : labels) {
			labelsCopy.add(label);
		}
		assert(labelsCopy.size() == nData);
		
		return labelsCopy;
	}
	
	public double getPixel(int letterIdx, int pixelRow, int pixelCol) {
		assert(0 <= letterIdx && letterIdx < nData);
		assert(0 <= pixelRow && pixelRow < IMG_HEIGHT);
		assert(0 <= pixelCol && pixelCol < IMG_WIDTH);
		
		int pixelIdx = pixelRow * IMG_HEIGHT + pixelCol;
		return letters.get(letterIdx).get(pixelIdx);
	}
	
	public List<Double> getPixelsForLetter(int letterIdx) {
		assert(0 <= letterIdx && letterIdx < nData);
		
		List<Double> pixelsCopy = new ArrayList<>();
		for (Double pixel : letters.get(letterIdx)) {
			pixelsCopy.add(pixel);
		}
		assert(pixelsCopy.size() == NUM_PIXELS);
		return pixelsCopy;
	}

}




import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class neuralnet {
	
	/* ------------ constants --------------- */
	
	private static final int NUM_ARGS = 9;
	private static final double RANDOM_MIN = -1;
	private static final double RANDOM_MAX = 1;
	
	/* ------------ class variables --------------- */

	private static String trainInPath;
	private static String validInPath;
	private static String trainOutPath;
	private static String validOutPath;
	private static String metricsOutPath;
	
	private static int numEpochs;
	private static int hiddenUnits;
	private static int initFlag;
	private static double learnRate;
	
	// parameter matrix dimensions
	private static int D;
	private static int M;
	private static int K;
	
	// parameter matrices
	private static List<List<Double>> alpha;
	private static List<List<Double>> beta;
	
	// input data
	private static CSVReader trainData;
	private static CSVReader validData;
	
	private static double uniformRandom(Random r, double min, double max) {
		return min + (max - min) * r.nextDouble();
	}
	
	private static List<List<Double>> initRandomMatrix(int nRow, int nCol) {
		
		List<List<Double>> matrix = new ArrayList<>();
		
		// random generator
		Random r = new Random();
		
		for (int i = 0; i < nRow; i ++) {
			
			List<Double> row = new ArrayList<>();
			// the bias term is always zero
			row.add(new Double(0));
			
			for (int j = 1; j < nCol; j ++) {
				double d = uniformRandom(r, RANDOM_MIN, RANDOM_MAX);
				row.add(new Double(d));
			}
			
			assert(row.size() == nCol);
			matrix.add(row);
		}
		
		assert(matrix.size() == nRow);
		return matrix;
	}
	
	private static List<List<Double>> initZeroMatrix(int nRow, int nCol) {
		
		List<List<Double>> matrix = new ArrayList<>();
		
		for (int i = 0; i < nRow; i ++) {
			List<Double> row = new ArrayList<>();
			for (int j = 0; j < nCol; j++) {
				row.add(new Double(0));
			}
			assert(row.size() == nCol);
			matrix.add(row);
		}
		
		assert(matrix.size() == nRow);
		return matrix;
	}
	
	private static List<Double> multMatrixWithVector(List<List<Double>> m, List<Double> v,
							 int nRow, int nCol) {
		assert(m != null && v != null);
		assert(m.size() == nRow);
		assert(v.size() == nCol);
		
		// result vector
		List<Double> result = new ArrayList<>();
		
		// multiply each row of the matrix with the vector
		for (int i = 0; i < nRow; i ++) {
			double d = 0;
			assert(m.get(i).size() == nCol);
			for (int j = 0; j < nCol; j ++) {
				d += m.get(i).get(j) * v.get(j);
			}
			result.add(d);
		}
		
		// return the result
		assert(result.size() == nRow);
		return result;
	}
	
	private static void trainModel() {
		// TODO
		for (int epoch = 0; epoch < numEpochs; epoch ++) {
			for (int i = 0; i < trainData.getNumberOfData(); i ++) {
				
			}
		}
	}
	
	
	public static void main(String[] args) throws Exception {
		
		// check number of command line arguments
		if (args == null || args.length != NUM_ARGS) {
			throw new Exception("Wrong number of command line arguments.");
		}
		
		// read the command line arguments
		trainInPath = args[0];
		validInPath = args[1];
		trainOutPath = args[2];
		validOutPath = args[3];
		metricsOutPath = args[4];
		numEpochs = Integer.parseInt(args[5]);
		hiddenUnits = Integer.parseInt(args[6]);
		initFlag = Integer.parseInt(args[7]);
		learnRate = Double.parseDouble(args[8]);
		
		// read the data
		trainData = new CSVReader(trainInPath, ",");
		validData = new CSVReader(validInPath, ",");
		
		// set the dimensions
		K = 10;
		D = hiddenUnits;
		M = CSVReader.getNumPixels();
		
		// initialize the matrices
		if (initFlag == 1) {
			alpha = initRandomMatrix(D, M + 1);
			beta = initRandomMatrix(K, D + 1);
		}
		else if (initFlag == 2) {
			alpha = initZeroMatrix(D, M + 1);
			beta = initRandomMatrix(K, D + 1);
		}
		else {
			throw new Exception ("Wrong init flag.");
		}
		
		// train the model
		trainModel();
	}
}




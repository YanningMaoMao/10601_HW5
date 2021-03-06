import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class empirical {
	
	/* ------------ constants --------------- */
	
	private static final int NUM_ARGS = 9;
	private static final double RANDOM_MIN = -0.1;
	private static final double RANDOM_MAX = 0.1;
	
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
	
	// cross entropies
	private static List<Double> trainEntropies = new ArrayList<>();
	private static List<Double> validEntropies = new ArrayList<>();
	
	// prediction error rates
	private static double trainError;
	private static double validError;
	
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
	
	/**
	 * Multiply two vectors to get a matrix
	 */
	private static List<List<Double>> multVectorWithVector(List<Double> v1, List<Double> v2,
								int nRow, int nCol) {
		assert(v1 != null && v2 != null);
		assert(v1.size() == nRow && v2.size() == nCol);
		
		// the result matrix
		List<List<Double>> m = new ArrayList<>();
		
		// calculate the matrix
		for (int i = 0; i < nRow; i ++) {
			List<Double> row = new ArrayList<>();
			for (int j = 0; j < nCol; j ++) {
				double d = v1.get(i) * v2.get(j);
				row.add(d);
			}
			assert(row.size() == nCol);
			m.add(row);
		}
		
		assert(m.size() == nRow);
		return m;
	}
	
	private static List<Double> multRevMatrixWithVector(List<List<Double>> m, List<Double> v,
								int nMatrixRow, int nMatrixCol) {
		assert(m != null && v != null);
		assert(m.size() == nMatrixRow);
		assert(v.size() == nMatrixRow);
		
		// the result vector
		List<Double> result = new ArrayList<>();
		
		// calculate the vector
		for (int j = 0; j < nMatrixCol; j ++) {
			double d = 0;
			for (int i = 0; i < nMatrixRow; i ++) {
				d += m.get(i).get(j) * v.get(i);
			}
			result.add(d);
		}
		
		assert(result.size() == nMatrixCol);
		return result;
	}
	
	private static List<Double> elemWiseMultVectors(List<Double> v1, List<Double> v2) {
		assert(v1 != null && v2 != null);
		assert(v1.size() == v2.size());
		
		// the result vector
		List<Double> result = new ArrayList<>();
		
		// element wise multiply the vectors
		for (int i = 0; i < v1.size(); i ++) {
			result.add(v1.get(i) * v2.get(i));
		}
		
		assert(result.size() == v1.size());
		return result;
	}
	
	private static void updateAlpha(List<List<Double>> gAlpha) {
		assert(gAlpha.size() == alpha.size() && alpha.size() == D);
		
		for (int i = 0; i < D; i ++) {
			assert(gAlpha.get(i).size() == M + 1);
			assert(alpha.get(i).size() == M + 1);
			for (int j = 0 ; j < M + 1; j ++) {
				double oldVal = alpha.get(i).get(j);
				double newVal = oldVal - learnRate * gAlpha.get(i).get(j);
				alpha.get(i).set(j, newVal);
			}
		}
	}
	
	private static void updateBeta(List<List<Double>> gBeta) {
		assert(beta != null && gBeta != null);
		assert(beta.size() == K && gBeta.size() == K);
		
		for (int i = 0; i < K; i ++) {
			assert(beta.get(i).size() == D + 1);
			assert(gBeta.get(i).size() == D + 1);
			for (int j = 0; j < D + 1; j ++) {
				double oldVal = beta.get(i).get(j);
				double newVal = oldVal - learnRate * gBeta.get(i).get(j);
				beta.get(i).set(j, newVal);
			}
		}
	}
	
	private static List<Double> applySigmoidToList(List<Double> lst) {
		assert(lst != null);
		
		List<Double> newLst = new ArrayList<>();
		for (Double d : lst) {
			double sigmoidD = 1.0 / (1 + Math.exp((-1.0) * d.doubleValue()));
			newLst.add(sigmoidD);
		}
		
		return newLst;
	}
	
	private static List<Double> applySoftmaxToList(List<Double> lst) {
		assert(lst != null);
		
		List<Double> expLst = new ArrayList<>();
		
		for (Double d : lst) {
			expLst.add(Math.exp(d));
		}
		
		double expSum = 0;
		for (Double d : expLst) {
			expSum += d;
		}
		
		List<Double> softmaxLst = new ArrayList<>();
		for (Double d : expLst) {
			softmaxLst.add(d / expSum);
		}
		
		return softmaxLst;
	}
	
	private static double getEntropy(CSVReader data) {
		
		double entropy = 0;
		
		for (int i = 0; i < data.getNumberOfData(); i ++) {
			
			// get input data
			List<Double> x = data.getPixelsForLetter(i);
			x.add(0, new Double(1));
			assert(x.size() == M + 1);
			
			// calculate y hat
			List<Double> yHatTemp = multMatrixWithVector(alpha, x, D, M + 1);
			yHatTemp = applySigmoidToList(yHatTemp);
			yHatTemp.add(0, new Double(1));
			List<Double> yHat = multMatrixWithVector(beta, yHatTemp, K, D + 1);
			yHat = applySoftmaxToList(yHat);
			
			// calculate y_k times log y_hat
			int label = data.getLabel(i);
			entropy += Math.log(yHat.get(label));
			
		}
		
		// calculate mean entropy
		entropy = (-1) * entropy / data.getNumberOfData();
		
		return entropy;
		
	}
	
	private static void trainModel() {
		
		// train the neural network for the given number of epochs
		for (int epoch = 0; epoch < numEpochs; epoch ++) {
			
			// update the matrix parameters by each letter instance
			for (int i = 0; i < trainData.getNumberOfData(); i ++) {
				
				// the pixel vector
				List<Double> x = trainData.getPixelsForLetter(i);
				x.add(0, new Double(1));
				assert(x.size() == M + 1);
				
				// ------ NNForward ------ //
				
				// a = alpha times x
				List<Double> a = multMatrixWithVector(alpha, x, D, M + 1);
				
				// z = sigmoid(a)
				List<Double> z = applySigmoidToList(a);
				z.add(0, new Double(1));
				
				// b = beta times z
				assert(z.size() == D + 1);
				List<Double> b = multMatrixWithVector(beta, z, K, D + 1);
				
				// yHat = softmax(b)
				assert(b.size() == K);
				List<Double> yHat = applySoftmaxToList(b);
				assert(yHat.size() == K);
				
				// J = - y_transpose log(yHat)
				int label = trainData.getLabel(i);
				
				// ------ NNBackward ------ //
				
				// g_yHat = - y divided by yHat
				double gYHat = (-1) * (1.0 / yHat.get(label));
				
				// gb
				List<Double> gb = new ArrayList<>();
				for (int col = 0; col < yHat.size(); col ++) {
					double temp = yHat.get(label) * yHat.get(col);
					if (col == label) {
						temp = yHat.get(label) - temp;
					}
					else {
						temp = 0 - temp;
					}
					temp = temp * gYHat;
					gb.add(temp);
				}
				assert(gb.size() == K);
				
				// gBeta
				List<List<Double>> gBeta = multVectorWithVector(gb, z, K, D + 1);
				
				// gz
				List<Double> gz = multRevMatrixWithVector(beta, gb, K, D + 1);
				assert(gz.size() == D + 1);
				
				// ga = gz bit-wise times z bit-wise times (1 - z)
				List<Double> oneMinusZ = z.stream().map(d -> 1 - d).collect(Collectors.toList());
				List<Double> ga = elemWiseMultVectors(gz, elemWiseMultVectors(z, oneMinusZ));
				
				// gAlpha = ga times x_transpose
				ga.remove(0); // remove bias term
				List<List<Double>> gAlpha = multVectorWithVector(ga, x, D, M + 1);
				
				// ------ update parameter matrices ------ //
				
				// update alpha
				updateAlpha(gAlpha);
				// update beta
				updateBeta(gBeta);
			
			}

			trainEntropies.add(getEntropy(trainData));
			validEntropies.add(getEntropy(validData));

		}
		
		// ------ calculate cross entropy ------ //
		// trainEntropies.add(getEntropy(trainData));
	}
	
	private static double predictLabels(CSVReader data, String outPath) throws FileNotFoundException {
		
		// number of wrong predictions
		double numWrongPred = 0;
		
		// open the writer
		File outFile = new File(outPath);
		PrintWriter wr = new PrintWriter(outFile);
		
		// predict label for each instance
		for (int i = 0; i < data.getNumberOfData(); i ++) {
			
			// the input vector
			List<Double> x = data.getPixelsForLetter(i);
			x.add(0, new Double(1));
			
			// get hidden layer
			List<Double> hidden = multMatrixWithVector(alpha, x, D, M + 1);
			hidden = applySigmoidToList(hidden);
			hidden.add(0, new Double(1));
			
			// predict the labels
			List<Double> labelProbs = multMatrixWithVector(beta, hidden, K, D + 1);
			labelProbs = applySoftmaxToList(labelProbs);
			
			// find the label with maximum probability
			int predLabel = 0;
			double maxProb = labelProbs.get(0);
			for (int k = 1; k < K; k ++) {
				if (labelProbs.get(k) > maxProb) {
					maxProb = labelProbs.get(k);
					predLabel = k;
				}
			}
			
			// compare the predicted label with the original label
			if (predLabel != data.getLabel(i)) {
				numWrongPred += 1;
			}
			
			// write the prediction
			wr.println(predLabel);
		}
		
		wr.close();
		return numWrongPred / new Double(data.getNumberOfData());
	}
	
	private static void outputMetrics() throws FileNotFoundException {
		
		// output the entropies and the error rates
		File metricsOut = new File(metricsOutPath);
		PrintWriter wr = new PrintWriter(metricsOut);
		
		// output the entropies
		for (int epoch = 0; epoch < numEpochs; epoch ++) {
			// output entropy for training data
			// String trainStr = crossentropy(train): ";
			// trainStr += trainEntropies.get(epoch);
			wr.print(trainEntropies.get(epoch) + ",");
			// output entropy for validation data
			// String validStr = "epoch=" + (epoch + 1) + " crossentropy(validation): ";
			// validStr += validEntropies.get(epoch);
			wr.println(validEntropies.get(epoch));
		}
		
		// wr.println(trainStr);
		
		wr.close();
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
			beta = initZeroMatrix(K, D + 1);
		}
		else {
			throw new Exception ("Wrong init flag.");
		}
		
		// train the model
		trainModel();
		
		// print the entropies
		System.out.println("Train Entropy : " + trainEntropies.get(0));
		System.out.println("Validation Entropy : " + validEntropies.get(0));
		
		// predict the labels
		// trainError = predictLabels(trainData, trainOutPath);
		// validError = predictLabels(validData, validOutPath);

		// output the entropies and prediction error rates
		outputMetrics();
		
	}
	
}




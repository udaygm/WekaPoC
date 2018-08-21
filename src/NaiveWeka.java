import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class NaiveWeka {
	public static void main(String[] args) throws Exception {
		// ConverterUtils.DataSource source1 = new
		// ConverterUtils.DataSource("./data/train.arff");
		BufferedReader source1 = readDataFile("weather.txt");
		Instances train = new Instances(source1);
		// Instances train = source1.getDataSet();
		// setting class attribute if the data format does not provide this information
		// For example, the XRFF format saves the class attribute information as well
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);

		// ConverterUtils.DataSource source2 = new
		// ConverterUtils.DataSource("./data/test.arff");
		// Instances test = source2.getDataSet();

		BufferedReader source2 = readDataFile("test.txt");
		Instances test = new Instances(source2);
		// setting class attribute if the data format does not provide this information
		// For example, the XRFF format saves the class attribute information as well
		if (test.classIndex() == -1)
			test.setClassIndex(train.numAttributes() - 1);

		System.out.println("Test Instances size:::" + test.size());

		// model

		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(train);

		// this does the trick
		for (int i = 0; i < test.size(); i++) {
			double label = naiveBayes.classifyInstance(test.instance(i));
			test.instance(i).setClassValue(label);

			System.out.println(test.instance(i).stringValue(4));
		}
	}

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}
}

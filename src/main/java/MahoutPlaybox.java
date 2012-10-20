import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.DefaultTreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.DescriptorException;
import org.apache.mahout.classifier.df.ref.SequentialBuilder;
import org.apache.mahout.common.RandomUtils;

import java.io.IOException;
import java.util.Random;

public class MahoutPlaybox {
    private static final int NUM_ATTRIBUTES = 10;

    public static void main(String[] args) throws IOException, DescriptorException {
//        Dataset dataset = Dataset.read(new DataInputStream(new FileInputStream(new File(""))));
        Random rng = RandomUtils.getRandom();

//        Data data = Utils.randomData(rng, NUM_ATTRIBUTES, false, 100);

        String descriptor = "N N N L ";


        String[] sData = new String[] {
                "7,6,2,1,",
                "5,7,4,1,",
                "1,7,3,2,",
                "4,1,8,3,",
                "5,5,5,1,",
                "4,4,6,3,",
                "4,5,2,2,",
                "2,2,7,3,",
                "6,6,6,2,",
                "2,9,2,2,",
                "3,3,8,3,",
                "3,3,7,3,",
                "3,3,6,3,",
                "3,3,7,3,",
                "3,3,7,3,"
        };

        Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);

        Data data =  DataLoader.loadData(dataset, sData);

        Data train = data.clone();

        Data test = train.rsplit(rng, (int) (data.size() * 0.1));

        SequentialBuilder forestBuilder = new SequentialBuilder(rng, new DefaultTreeBuilder(), train);

        DecisionForest tree = forestBuilder.build(5);

        double[] predictions = new double[test.size()];
        tree.classify(test, predictions);

        for (double prediction : predictions) {
            System.out.println("prediction = " + prediction);
        }
    }
}

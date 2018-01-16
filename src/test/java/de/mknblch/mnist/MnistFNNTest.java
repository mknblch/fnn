package de.mknblch.mnist;

import de.mknblch.fnn.DataSet;
import de.mknblch.fnn.FFN;
import de.mknblch.fnn.Trainer;
import org.junit.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

/**
 * @author mknblch
 */
public class MnistFNNTest {


    public static final Path IMAGE_PATH = Paths.get("/Users/mknobloch/Downloads/mnist/train-images-idx3-ubyte");
    public static final Path LABEL_PATH = Paths.get("/Users/mknobloch/Downloads/mnist/train-labels-idx1-ubyte");

    @Test
    public void test() throws Exception {

        final Trainer trainer = Trainer.builder(28 * 28, 10)
                .withLearningRate(1)
                .addHiddenLayer(1000)
                .addHiddenLayer(500)
                .build(42L);

        final MnistDataSets dataSets = new MnistDataSets(IMAGE_PATH, LABEL_PATH);

        final DataSet dataSet = dataSets.create(100);
        final FFN net = trainer.train(dataSet, 0.01, 100);

        matchUnknown(dataSets, net);
//        matchKnown(dataSet, net);
    }

    private void matchKnown(DataSet dataSet, FFN net) {
        final double[][] inputs = dataSet.inputs();
        final double[][] expected = dataSet.expected();
        for (int i = 0; i < dataSet.size(); i++) {
            final double[] results = net.eval(inputs[i]);
            for (double result : results) {
                System.out.printf("%.0f", result);
            }
            System.out.print(" | ");
            for (double result : expected[i]) {
                System.out.printf("%.0f", result);
            }
            System.out.println();

        }
    }

    private void matchUnknown(MnistDataSets dataSets, FFN net) {
        final Iterator<MnistImage> imageIterator = dataSets.getImageIterator();

        for (int i = 0; i < 100 && imageIterator.hasNext(); i++) {
            final MnistImage image = imageIterator.next();
            final double[] results = net.eval(image.toArray());
            for (double result : results) {
                System.out.printf("%.0f ", result);
            }
            System.out.println(": " + image.label);
        }
    }
}

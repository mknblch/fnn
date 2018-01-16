package de.mknblch.mnist;

import de.mknblch.fnn.DataSet;
import de.mknblch.fnn.Trainer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;

/**
 * @author mknblch
 */
public class MnistDataSets {

    private static final double[][] expected = new double[][]{
            new double[] {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
    };

    private final Iterator<MnistImage> imageIterator;

    public MnistDataSets(Path imagePath, Path labelPath) throws IOException {
        this.imageIterator = new MnistImageIterator(imagePath, labelPath);
    }

    public Iterator<MnistImage> getImageIterator() {
        return imageIterator;
    }

    public DataSet create(int imageCount) {
        final double[][] inputs = new double[imageCount][];
        final double[][] outputs = new double[imageCount][];
        for (int i = 0; i < imageCount && imageIterator.hasNext(); i++) {
            final MnistImage image = imageIterator.next();
            inputs[i] = image.toArray();
            outputs[i] = expectedOutputForLabel(image.label);
        }
        return DataSet.fromArray(inputs, outputs);
    }

    private static double[] expectedOutputForLabel(int label) {
        return expected[label];
    }
}

package de.mknblch.fnn;

import org.junit.Ignore;
import org.junit.Test;
import static de.mknblch.fnn.TestData.*;
import static org.junit.Assert.*;

/**
 * @author mknblch
 */
public class FNNTest {

    public static final double RATE = 0.45;
    public static final long RANDOM_SEED = 42L;
    public static final int MAX_ITERATIONS = 1000;
    public static final double EXPECTED_ERROR = 0.01;

    /**
     * Exclusive Or (hidden-layer with 3 units)
     */
    @Test
    public void testXOR() throws Exception {
        assertClassify(train(true, XOR), XOR);
    }

    /**
     * Equality (hidden-layer with 3 units)
     */
    @Test
    public void testEQ() throws Exception {
        assertClassify(train(true, EQ), EQ);
    }

    /**
     * And (no hidden layer)
     */
    @Test
    public void testAND() throws Exception {
        assertClassify(train(false, AND), AND);
    }

    /**
     * Or (no hidden layer)
     */
    @Test
    public void testOR() throws Exception {
        assertClassify(train(false, OR), OR);
    }

    /**
     * Implication (no hidden layer)
     */
    @Test
    public void testIMP() throws Exception {
        assertClassify(train(false, IMP), IMP);
    }

    /**
     * XOR can't be trained without hidden layer
     */
    @Ignore("TODO")
    @Test(expected = IllegalStateException.class)
    public void testNotTrainable() throws Exception {
        Trainer.builder(2, 1)
                .withLearningRate(RATE)
                .build()
                .train(XOR, 0.01, 10_000);
    }

    private static Trainer train(boolean hiddenLayer, DataSet dataSet) {
        final Trainer.Builder trainer = Trainer
                .builder(2, 1)
                .withLearningRate(RATE);

        if (hiddenLayer) {
            trainer.addHiddenLayer(3);
        }

        return trainer
                .build(RANDOM_SEED)
                .train(dataSet, EXPECTED_ERROR, MAX_ITERATIONS);
    }

    /**
     * check if the network verifies the data set
     */
    public static void assertClassify(FFN network, DataSet dataSet) {
        final double[][] input = dataSet.inputs();
        final double[][] expected = dataSet.expected();
        for (int i = 0; i < input.length; i++) {
            final double[] output = network.eval(input[i]);
            for (int j = 0; j < output.length; j++) {
                if (Math.abs(output[j] - expected[i][j]) > 0.5) {
                    fail("Network did not match");
                }
            }
        }
    }
}
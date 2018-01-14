package de.mknblch.nnet;

import org.junit.Test;
import static de.mknblch.nnet.DataSets.*;
import static org.junit.Assert.*;

/**
 * @author mknblch
 */
public class NetTest {

    public static final double RATE = 0.3;
    public static final long RANDOM_SEED = 42L;
    public static final int MAX_ITERATIONS = 2000;
    public static final double EXPECTED_ERROR = 0.1;

    /**
     * Exclusive Or (hidden-layer with 3 units)
     */
    @Test
    public void testXOR() throws Exception {
        final FeedForwardNetwork net = train(true, XOR);
        evalBinary(net, XOR);
    }

    /**
     * Equality (hidden-layer with 3 units)
     */
    @Test
    public void testEQ() throws Exception {
        final FeedForwardNetwork net = train(true, EQ);
        evalBinary(net, EQ);
    }

    /**
     * And (no hidden layer)
     */
    @Test
    public void testAND() throws Exception {
        final FeedForwardNetwork net = train(false, AND);
        evalBinary(net, AND);
    }

    /**
     * Or (no hidden layer)
     */
    @Test
    public void testOR() throws Exception {
        final FeedForwardNetwork net = train(false, OR);
        evalBinary(net, OR);
    }

    /**
     * Implication (no hidden layer)
     */
    @Test
    public void testIMP() throws Exception {
        final FeedForwardNetwork net = train(false, IMP);
        evalBinary(net, IMP);
    }

    private static FeedForwardNetwork train(boolean hiddenLayer, DataSet dataSet) {
        final TrainableNetwork.Builder trainer = TrainableNetwork
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
    public static void evalBinary(FeedForwardNetwork network, DataSet dataSet) {
        final double[][] input = dataSet.inputs();
        final double[][] expected = dataSet.expected();
        for (int i = 0; i < input.length; i++) {
            final double[] output = network.feed(input[i]);
            for (int j = 0; j < output.length; j++) {
                if (Math.abs(output[j] - expected[i][j]) > 0.5) {
                    fail("Network did not match");
                }
            }
        }
    }
}
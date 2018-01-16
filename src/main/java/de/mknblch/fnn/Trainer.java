package de.mknblch.fnn;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Trainable Network
 *
 * @author mknblch
 */
public class Trainer extends FFN {

    // layer array including input and output layers
    private final Layer[] layers;
    // learning rate
    private double rate = 0.1;
    // number of iterations the training took
    private int iterations = -1;

    private DoubleUnaryOperator rateFunction = v -> 0.1;

    /**
     * create a new Builder
     * @param inputs number of input units
     * @param outputs number of output units
     * @return a new builder
     */
    public static Builder builder(int inputs, int outputs) {
        return new Builder(inputs, outputs);
    }

    private Trainer(Layer[] layers, double rate) {
        super(layers);
        this.layers = layers;
        this.rate = rate;
    }

    /**
     * get the number of iterations the
     * training took or -1 if untrained
     * @return number of iterations
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * train the network with the given parameters
     * @param dataSet the dataSet
     * @param converge error threshold for convergence
     * @param maxIterations maximum count of iterations before Exception is thrown
     * @return itself for method chaining
     * @throws IllegalStateException if iteration limit exceeds
     */
    public Trainer train(DataSet dataSet, double converge, int maxIterations) {
        final double[][] inputs = dataSet.inputs();
        final double[][] expected = dataSet.expected();
        for (iterations = 0; iterations < maxIterations; iterations++) {
            final double train = train(inputs, expected);
            System.out.println("error = " + train);
            if (train <= converge) {
                return this;
            }
        }
        return this;
//        throw new IllegalStateException("Network did not converge in " + maxIterations + " iterations");
    }

    /**
     * do a single training step with a batch of values
     * @param input array of input arrays
     * @param expected array of expected output arrays
     * @return sum of the errors of the output layer
     */
    public double train(double[][] input, double[][] expected) {
        double error = 0;
        for (int i = 0; i < input.length; i++) {
            error = Math.max(error, train(input[i], expected[i]));
        }
        return error;
    }

    /**
     * do a single training step with the given values
     * @param input the input values
     * @param expected expected values
     * @return error of last layer
     */
    public double train(double[] input, double[] expected) {
        eval(input);
        backward(expected);
        return error(layers[layers.length - 1].values, expected);
    }

    /**
     * calculate error
     * @param values the output values of a layer
     * @param expected expected values
     * @return the error
     */
    private double error(double[] values, double[] expected) {
        double e = 0.0;
        for (int j = 0; j < values.length; j++) {
            e += Math.pow(values[j] - expected[j], 2.0);
        }
        return e / 2.0;
    }

    /**
     * backpropagation
     * @param expected expected values
     * @return deltas
     */
    private double[][] backward(double[] expected) {
        final double[][] deltas = new double[layers.length][];
        calcOutputDeltas(deltas, expected);
        calcHiddenDeltas(deltas);
        update(deltas);
        return deltas;
    }

    /**
     * append deltas of the output layer to the delta array
     * @param delta the delta array
     * @param expected expected values
     */
    private void calcOutputDeltas(double[][] delta, double[] expected) {
        final double[] outValues = layers[layers.length - 1].values;
        delta[layers.length - 1] = new double[outValues.length];
        Arrays.setAll(delta[layers.length - 1], i -> outValues[i] * (1.0 - outValues[i]) * (outValues[i] - expected[i]));
    }

    /**
     * appends deltas of hidden layers (if any) to the delta array
     * @param delta the delta array
     */
    private void calcHiddenDeltas(double[][] delta) {
        for (int l = layers.length - 2; l >= 1; l--) {
            final Layer layer = layers[l];
            final Layer next = layers[l + 1];
            delta[l] = new double[layer.values.length];
            for (int j = 0; j < layer.values.length; j++) {
                double t = 0;
                for (int i = 0; i < next.values.length; i++) {
                    t += delta[l + 1][i] * next.weights[i * layer.values.length + j];
                }
                delta[l][j] = layer.values[j] * (1.0 - layer.values[j]) * t;
            }
        }
    }

    /**
     * update weights and biases
     * @param delta completely calculated delta array
     */
    private void update(double[][] delta) {
        for (int k = 1; k < layers.length; k++) {
            final Layer layer = layers[k];
            final Layer previous = layers[k - 1];
            for (int i = 0; i < layer.values.length; i++) {
                final double fc = -rate * delta[k][i];
                for (int j = 0; j < previous.values.length; j++) {
                    final int index = j * layer.values.length + i;
                    layer.weights[index] += fc * previous.values[j];
                }
                layer.bias[i] += fc;
            }
        }
    }

    /**
     * builder to ease setup and addition of hidden layers.
     */
    public static class Builder {

        // list of layers including input and output layers
        private final List<Layer> layers = new ArrayList<>();
        // reference to the input values
        private final double[] input;
        // size of last added layer
        private int layerSize;
        // number of output units
        private final int outputSize;
        // learning learningRate
        private double learningRate = 0.1;

        private Builder(int inputSize, int outputSize) {
            input = new double[inputSize];
            layerSize = inputSize;
            this.outputSize = outputSize;
            layers.add(new Layer(input));
        }

        /**
         * add a new hidden layer
         * @param size number of neurons / units in the layer
         * @return this builder
         */
        public Builder addHiddenLayer(int size) {
            Layer temp = new Layer(layerSize, size);
            layerSize = size;
            layers.add(temp);
            return this;
        }

        /**
         * set learning learningRate
         * @param rate learning learningRate
         * @return this builder
         */
        public Builder withLearningRate(double rate) {
            this.learningRate = rate;
            return this;
        }

        /**
         * build a trainable network
         * @return a trainable eval forward network
         */
        public Trainer build() {
            return build(System.currentTimeMillis());
        }

        /**
         * build a trainable network
         * @param randomSeed seed value for weight randomization or -1L to skip
         * @return a trainable eval forward network
         */
        public Trainer build(long randomSeed) {
            addHiddenLayer(outputSize);
            if (randomSeed != -1L) {
                initialize(randomSeed);
            }
            return new Trainer(this.layers.toArray(new Layer[0]), learningRate);
        }

        /**
         * initialize weights using the given seed
         * @param seed seed value
         */
        private void initialize(long seed) {
            final SecureRandom random = new SecureRandom(
                    new byte[]{
                            (byte) ((seed >> 56) & 0xFF),
                            (byte) ((seed >> 48) & 0xFF),
                            (byte) ((seed >> 40) & 0xFF),
                            (byte) ((seed >> 32) & 0xFF),
                            (byte) ((seed >> 24) & 0xFF),
                            (byte) ((seed >> 16) & 0xFF),
                            (byte) ((seed >> 8) & 0xFF),
                            (byte) (seed & 0xFF)
                    }
            );

            for (int i = 1; i < layers.size(); i++) {
                Arrays.setAll(layers.get(i).weights, k -> 1.0 - random.nextDouble() * 2.0);
            }
        }
    }
}

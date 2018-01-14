package de.mknblch.nnet;

import java.util.Arrays;

/**
 * Simple Feed Forward Network
 * @author mknblch
 */
public class FeedForwardNetwork {

    // layer array including input and output layers
    final Layer[] layers;

    /**
     * build a network with (2 or more) layers
     * @param layers the network layers
     */
    FeedForwardNetwork(Layer[] layers) {
        this.layers = layers;
    }

    /**
     * number of hidden layers
     * @return number of hidden layers
     */
    public int numLayer() {
        return layers.length - 2;
    }

    /**
     * feed the network with given values
     * @param input the input values
     * @return output of the network
     */
    public double[] feed(double[] input) {
        layers[0].values = input;
        for (int i = 1; i < layers.length; i++) {
            forward(i);
        }
        return layers[layers.length - 1].values;
    }

    /**
     * get current output of the last layer
     * @return output of the layer
     */
    public double[] values() {
        return values(layers.length - 1);
    }

    /**
     * get output of the given layer
     * @return output of the layer
     */
    public double[] values(int layer) {
        return layers[layer].values;
    }

    /**
     * get weights of the given layer
     * @return weights of the layer
     */
    public double[] weights(int layer) {
        return layers[layer].values;
    }

    /**
     * get biases of the given layer
     * @return biases of the layer
     */
    public double[] bias(int layer) {
        return layers[layer].values;
    }

    /**
     * do feed forward step for the given layer
     * @param layer the index of the layer
     */
    private void forward(int layer) {
        final double[] precursor = layers[layer - 1].values;
        final double[] weights = layers[layer].weights;
        final double[] values = layers[layer].values;
        final double[] bias = layers[layer].bias;
        Arrays.setAll(values, j -> {
            double t = bias[j];
            for (int i = 0; i < precursor.length; i++) {
                t += precursor[i] * weights[i * values.length + j];
            }
            return sigmoid(t);
        });
    }

    /**
     * sigmoid function
     * @param x parameter
     * @return the sigmoid of x
     */
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * simple VO representing one layer of neurons
     */
    static class Layer {

        // actual output values
        double[] values;
        // biases (initially set to 1.0)
        final double[] bias;
        // weight matrix (laid out as sequential array)
        final double[] weights;

        /**
         * Input layer ctor
         * @param values array for input values
         */
        Layer(double[] values) {
            this.values = values;
            weights = null;
            bias = null;
        }

        /**
         * Hidden layer ctor
         * @param previousUnits number of output units in the previous layer
         * @param units desired number of units in the layer
         */
        Layer(int previousUnits, int units) {
            this.values = new double[units];
            this.bias = new double[units];
            weights = new double[previousUnits * units];
            Arrays.fill(bias, 1.0);
        }
    }
}

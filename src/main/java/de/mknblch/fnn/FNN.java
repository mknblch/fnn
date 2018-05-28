package de.mknblch.fnn;

import java.util.Arrays;
import java.util.function.DoubleFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;

/**
 * Feedforward Network
 *
 * @author mknblch
 */
public class FNN {

    // layer array including input and output layers
    final Layer[] layers;

    /**
     * sigmoid function
     */
    DoubleUnaryOperator activationFunction = (x) -> 1.0 / (1.0 + Math.exp(-x));

    /**
     * build a network with given (2 or more) layers
     * @param layers the network layers
     */
    FNN(Layer[] layers) {
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
    public double[] eval(double[] input) {
        return eval(input, false);
    }

    /**
     * feed the network with given values
     * @param input the input values
     * @param parallel use {@link Arrays#parallelSetAll(double[], IntToDoubleFunction)} instead of
     *                 {@link Arrays#setAll(double[], IntToDoubleFunction)}
     * @return output of the network
     */
    public double[] eval(double[] input, boolean parallel) {
        layers[0].values = input;
        for (int i = 1; i < layers.length; i++) {
            forward(i, parallel);
        }
        return layers[layers.length - 1].values;
    }

    public FNN withActivationFunction(DoubleUnaryOperator activationFunction) {
        this.activationFunction = activationFunction;
        return this;
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
     * do eval forward step for the given layer
     * @param layer the index of the layer
     */
    private void forward(int layer, boolean parallel) {
        if (parallel) {
            parallelForward(layers[layer].values, layers[layer - 1].values, layers[layer].bias, layers[layer].weights);
        } else {
            sequentialForward(layers[layer].values, layers[layer - 1].values, layers[layer].bias, layers[layer].weights);
        }
    }

    private void sequentialForward(double[] values, double[] precursor, double[] bias, double[] weights) {
        Arrays.setAll(values, j -> {
            double t = bias[j];
            for (int i = 0; i < precursor.length; i++) {
                t += precursor[i] * weights[i * values.length + j];
            }
            return activationFunction.applyAsDouble(t);
        });
    }

    private void parallelForward(double[] values, double[] precursor, double[] bias, double[] weights) {
        Arrays.parallelSetAll(values, j -> {
            double t = bias[j];
            for (int i = 0; i < precursor.length; i++) {
                t += precursor[i] * weights[i * values.length + j];
            }
            return activationFunction.applyAsDouble(t);
        });
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
            weights = new double[0];
            bias = new double[0];
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

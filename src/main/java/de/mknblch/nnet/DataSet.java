package de.mknblch.nnet;

/**
 * basic training data set
 *
 * @author mknblch
 */
public interface DataSet {

    /**
     * number of elements
     * @return elements in the set
     */
    int size();

    /**
     * get the input values
     * @return input values
     */
    double[][] inputs();

    /**
     * get output values
     * @return output values
     */
    double[][] expected();

    /**
     * build a DataSet from the given arrays.
     * the size of input and expected must be equal
     * and the size of an input element must match the number of
     * input neurons of the network while the size of an expected
     * element must match the number of its output neurons.
     *
     * @param inputs array of input elements
     * @param expected array of output elements
     * @return a DataSet
     */
    static DataSet fromArray(double[][] inputs, double[][] expected) {

        return new DataSet() {
            @Override
            public int size() {
                return inputs.length;
            }

            @Override
            public double[][] inputs() {
                return inputs;
            }

            @Override
            public double[][] expected() {
                return expected;
            }
        };
    }
}

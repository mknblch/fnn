package de.mknblch.fnn;

/**
 * @author mknblch
 */
public class TestData {

    public static final double[][] INPUTS = {
            new double[]{0.0, 0.0},
            new double[]{0.0, 1.0},
            new double[]{1.0, 0.0},
            new double[]{1.0, 1.0}
    };

    public static DataSet OR = DataSet.fromArray(
            INPUTS,
            new double[][]{
                    new double[]{0.0},
                    new double[]{1.0},
                    new double[]{1.0},
                    new double[]{1.0}
            }
    );

    public static DataSet AND = DataSet.fromArray(
            INPUTS,
            new double[][]{
                    new double[]{0.0},
                    new double[]{0.0},
                    new double[]{0.0},
                    new double[]{1.0}
            }
    );

    public static DataSet XOR = DataSet.fromArray(
            INPUTS,
            new double[][]{
                    new double[]{0.0},
                    new double[]{1.0},
                    new double[]{1.0},
                    new double[]{0.0}
            }
    );

    public static DataSet EQ = DataSet.fromArray(
            INPUTS,
            new double[][]{
                    new double[]{1.0},
                    new double[]{0.0},
                    new double[]{0.0},
                    new double[]{1.0}
            }
    );

    public static DataSet IMP = DataSet.fromArray(
            INPUTS,
            new double[][]{
                    new double[]{1.0},
                    new double[]{1.0},
                    new double[]{0.0},
                    new double[]{1.0}
            }
    );
}

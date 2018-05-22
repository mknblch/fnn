package de.mknblch.fnn;

/**
 * @author mknblch
 */
public class ScaleData {

    public static double[] scaleInPlace(double[] data, double min, double max) {
        return scale(data, data, min, max);
    }

    public static double[] scaleInPlace(double[] data, int fromIndex, int length, double min, double max) {
        return scale(data, data, fromIndex, length, 0, min, max);
    }

    public static double[] scaleToNew(double[] data, double min, double max) {
        return scale(data, new double[data.length], 0, data.length, 0, min, max);
    }

    public static double[] scale(double[] data, double[] out, double min, double max) {
        return scale(data, out, 0, data.length, 0, min, max);
    }

    public static double[] scaleToNew(double[] data, int fromIndex, int length, double min, double max) {
        return scale(data, new double[length], fromIndex, length, -fromIndex, min, max);
    }

    public static double[] scale(double[] data, double[] out, int fromIndex, int length, int targetOffset, double min, double max) {
        double cmin = Double.MAX_VALUE;
        double cmax = -Double.MAX_VALUE;
        final int end = length + fromIndex;
        for (int i = fromIndex; i < end; i++) {
            final double v = data[i];
            if (v < cmin) {
                cmin = v;
            }
            if (v > cmax) {
                cmax = v;
            }
        }
        final double p = (max - min) / (cmax - cmin);
        for (int i = fromIndex; i < end; i++) {
            out[i + targetOffset] = min + p * (data[i] - cmin);
        }
        return out;
    }
}

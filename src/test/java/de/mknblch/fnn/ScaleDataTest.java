package de.mknblch.fnn;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author mknblch
 */
public class ScaleDataTest {

    public static final double[] in0 = {1, 1, 2, 2, 1, 1};

    public static final double[] e0 = {-1, -1, 1, 1, -1, -1};
    public static final double[] e1 = {0, 0, 1, 1, 0, 0};

    @Test
    public void test_e0() throws Exception {
        final double[] doubles = ScaleData.scaleToNew(in0, -1, 1);
        System.out.println(Arrays.toString(doubles));
        assertArrayEquals(e0, doubles, 0.1);
    }

    @Test
    public void test_e1() throws Exception {
        final double[] doubles = ScaleData.scaleToNew(in0, 0, 1);
        System.out.println(Arrays.toString(doubles));
        assertArrayEquals(e1, doubles, 0.1);
    }

}
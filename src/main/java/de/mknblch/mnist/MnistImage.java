package de.mknblch.mnist;

import javax.imageio.ImageIO;
import java.awt.image.*;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * @author mknblch
 */
public class MnistImage {

    public final int label;

    public final int width;
    public final int height;
    public final byte[] data;

    public MnistImage(int label, int width, int height, byte[] data) {
        this.label = label;
        this.width = width;
        this.height = height;
        this.data = data;
    }

    public void save(Path path) throws IOException {
        ImageIO.write(toBufferedImage(), "png", path.toFile());
    }

    public BufferedImage toBufferedImage() {
        final BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        final int[] data = ((DataBufferInt) bufferedImage.getRaster().getDataBuffer()).getData();
        Arrays.setAll(data, this::color);
        return bufferedImage;
    }

    private int color(int i) {
        return (((int) data[i] & 0xFF) << 16) | (((int) data[i] & 0xFF) << 8) | ((int) data[i] & 0xFF);
    }

    public double[] toArray() {
        final double[] doubles = new double[data.length];
        Arrays.setAll(doubles, i -> data[i] / 255.0);
        return doubles;
    }
}

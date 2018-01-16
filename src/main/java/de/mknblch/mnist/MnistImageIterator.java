package de.mknblch.mnist;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Iterator;

/**
 * @author mknblch
 */
public class MnistImageIterator implements Iterator<MnistImage> {

    private final MappedByteBuffer buffer;
    private final MnistLabels mnistLabels;

    public final int magicValue;
    public final int numImages;
    public final int height;
    public final int width;

    private int offset = 0;

    public MnistImageIterator(Path imagePath, Path labelPath) throws IOException {
        buffer = open(imagePath);
        magicValue = buffer.getInt();
        numImages = buffer.getInt();
        height = buffer.getInt();
        width = buffer.getInt();
        mnistLabels = new MnistLabels(labelPath);
    }

    @Override
    public boolean hasNext() {
        return buffer.hasRemaining();
    }

    @Override
    public MnistImage next() {
        final byte[] data = new byte[width * height];
        buffer.get(data);
        return new MnistImage(mnistLabels.label(offset++), width, height, data);
    }

    private static MappedByteBuffer open(Path path) throws IOException {
        final File file = path.toFile();
        return new RandomAccessFile(file, "r")
                .getChannel()
                .map(FileChannel.MapMode.READ_ONLY, 0, file.length());
    }
}

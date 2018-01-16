package de.mknblch.mnist;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

/**
 * @author mknblch
 */
public class MnistLabels {

    private final MappedByteBuffer buffer;
    public final int magicValue;
    public final int numLabels;
    private final byte[] labels;

    public MnistLabels(Path path) throws IOException {
        buffer = open(path);
        magicValue = buffer.getInt();
        numLabels = buffer.getInt();
        labels = new byte[numLabels];
        buffer.get(labels);
    }

    public int label(int index) {
        return labels[index] & 0xFF;
    }

    private static MappedByteBuffer open(Path path) throws IOException {
        final File file = path.toFile();
        return new RandomAccessFile(file, "r")
                .getChannel()
                .map(FileChannel.MapMode.READ_ONLY, 0, file.length());
    }
}

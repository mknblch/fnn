package de.mknblch.mnist;

import org.junit.Test;

import java.nio.file.Paths;
import java.util.Iterator;

/**
 * @author mknblch
 */
public class MnistTest {

    @Test
    public void test() throws Exception {

        final Iterator<MnistImage> iterator = new MnistImageIterator(
                Paths.get("/Users/mknobloch/Downloads/mnist/train-images-idx3-ubyte"),
                Paths.get("/Users/mknobloch/Downloads/mnist/train-labels-idx1-ubyte"));

        for (int i = 0; i < 10; i++) {
            if (!iterator.hasNext()) {
                break;
            }
            final MnistImage image = iterator.next();
            image.save(Paths.get("/Users/mknobloch/Downloads/", image.label + ".png"));
        }
    }

}
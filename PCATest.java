package cn.nsoc.ml.tf.model.local;

import junit.framework.TestCase;

public class PCATest extends TestCase{
    private PCA pca;

    @Override
    protected void setUp() throws Exception {
        pca = new PCA();
        super.setUp();
    }


    public void testPca() {
        double[][] v = {{1.0, 2.0, 3.0}, {3.0, 4.0, 3.0}, {5.0, 6.0, 3.0}};
        show(pca.pcaNormalized(v));
    }


    private void show(double[][] v) {
        for (double[] aV : v) {
            for (int column = 0; column < v[0].length; column++) {
                System.out.print(aV[column] + "    ");
            }
            System.out.println();
        }
    }
}

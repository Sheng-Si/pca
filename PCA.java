import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import Jama.Matrix;

/*
 * 算法步骤:
 * 1)将原始数据按列组成n行m列矩阵X
 * 2)特征中心化。即每一维的数据都减去该维的均值，使每一维的均值都为0
 * 3)求出协方差矩阵
 * 4)求出协方差矩阵的特征值及对应的特征向量
 * 5)将特征向量按对应的特征值大小从上往下按行排列成矩阵，取前k行组成矩阵p
 * 6)Y=XP' 即为降维到k维后的数据
 */
public class PCA {

    private double threshold;// 特征值阈值
    private Matrix matrix;
    private int principalComponentNum = 0;

    public PCA(double featureThreshold) {
        this.threshold = featureThreshold;
    }

    public PCA() {
        this(0.95);
    }

    /**
     * 获取主成份数
     * @return 主成份数
     */
    public int getPrincipalComponentNum() {
        return principalComponentNum;
    }

    /**
     *
     * 使每个样本的均值为0
     * @param initiallyMatrix 原始二维数组矩阵
     * @return zeroCenteredMatrix 中心化后的矩阵
     */
    public double[][] zeroCentered(double[][] initiallyMatrix) {
        int n = initiallyMatrix.length;
        int m = initiallyMatrix[0].length;
        double[] sum = new double[m];
        double[] average = new double[m];
        double[][] zeroCenteredMatrix = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (double[] aPrimary : initiallyMatrix) {
                sum[i] += aPrimary[i];
            }
            average[i] = sum[i] / n;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                zeroCenteredMatrix[j][i] = initiallyMatrix[j][i] - average[i];
            }
        }
        return zeroCenteredMatrix;
    }

    /**
     *
     * 计算协方差矩阵
     * @param matrix 中心化后的矩阵
     * @return result 协方差矩阵
     */
    public double[][] cov(double[][] matrix) {
        int n = matrix.length;// 行数
        int m = matrix[0].length;// 列数
        double[][] covMatrix = new double[m][m];// 协方差矩阵
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double temp = 0;
                for (double[] aMatrix : matrix) {
                    temp += aMatrix[i] * aMatrix[j];
                }
                covMatrix[i][j] = temp / (n - 1);
            }
        }
        return covMatrix;
    }

    /**
     * 求特征值矩阵
     * @param matrix 协方差矩阵
     * @return result 向量的特征值二维数组矩阵
     */
    public double[][] featureValueMatrix(double[][] matrix) {
        Matrix m = getOrCreateMatrix(matrix);
        return m.eig().getD().getArray();
    }

    /**
     * 标准化矩阵（特征向量矩阵）
     * @param matrix 特征值矩阵
     * @return result 标准化后的二维数组矩阵
     */
    public double[][] featureVectorMatrix(double[][] matrix) {
        Matrix m = getOrCreateMatrix(matrix);
        return m.eig().getV().getArray();
    }

    /**
     * 创建Matrix对象,创建过的则不再创建
     * @param matrix 矩阵
     * @return Matrix对象
     */
    private synchronized Matrix getOrCreateMatrix(double[][] matrix) {
        if(this.matrix == null) {
            this.matrix = new Matrix(matrix);
        }

        return this.matrix;
    }

    /**
     * 寻找主成分
     * @param featureValueMatrix 特征值二维数组
     * @param featureVectorMatrix 特征向量二维数组
     * @return principalMatrix 主成分矩阵
     */
    public Matrix getPrincipalComponent(double[][] featureValueMatrix, double[][] featureVectorMatrix) {
        Matrix A = new Matrix(featureVectorMatrix);// 定义一个特征向量矩阵
        double[][] featureVectorR = A.transpose().getArray();// 特征向量转置
        Map<Integer, double[]> principalMap = new HashMap<>();// key=主成分特征值，value=该特征值对应的特征向量
        TreeMap<Double, double[]> featureMap = new TreeMap<>(Collections.reverseOrder());// key=特征值，value=对应的特征向量；初始化为翻转排序，使map按key值降序排列

        double[] featureValueArray = extractDiagonal(featureValueMatrix);// 把特征值矩阵对角线上的元素放到数组featureValueArray里

        for (int i = 0; i < featureVectorR.length; i++) {
            double[] value = featureVectorR[i];
            featureMap.put(featureValueArray[i], value);
        }

        // 求特征总和
        double total = 0;// 存储特征值总和
        for (double anEigenvalueArray : featureValueArray) {
            total += anEigenvalueArray;
        }
        // 选出前几个主成分
        double temp = 0;
        this.principalComponentNum = 0;// 主成分数
        List<Double> plist = new ArrayList<>();// 主成分特征值
        for (double key : featureMap.keySet()) {
            if (temp / total <= threshold) {
                temp += key;
                plist.add(key);
                this.principalComponentNum ++;
            }
        }

        // 往主成分map里输入数据
        for (int i = 0; i < plist.size(); i++) {
            if (featureMap.containsKey(plist.get(i))) {
                principalMap.put(i, featureMap.get(plist.get(i)));
            }
        }

        // 把map里的值存到二维数组里
        double[][] principalArray = new double[principalMap.size()][];
        Iterator<Entry<Integer, double[]>> it = principalMap.entrySet()
                .iterator();
        for (int i = 0; it.hasNext(); i++) {
            principalArray[i] = it.next().getValue();
        }

        return new Matrix(principalArray);
    }

    /**
     * 归一化
     * @param matrix 将要归一化的矩阵
     * @return 归一化的矩阵
     */
    public static double[][] normalized(double[][] matrix) {
        if(matrix.length == 0) throw new IllegalArgumentException("is empty");
        double[][] r = new double[matrix.length][matrix[0].length];

        double max = matrix[0][0];
        double min = matrix[0][0];
        for (double[] aV : matrix) {
            for (int column = 0; column < matrix[0].length; column++) {
                double current = aV[column];
                if (current > max) {
                    max = current;
                } else if (current < min) {
                    min = current;
                }
            }
        }

        for (int row = 0; row < matrix.length; row ++) {
            for (int column = 0; column < matrix[0].length; column++) {
                r[row][column] = (matrix[row][column] - min) / (max - min);
            }
        }

        return r;
    }

    /**
     * 提取对角线元素
     * @param matrix 要提取的矩阵
     * @return 对角线元素数组
     */
    private double[] extractDiagonal(double[][] matrix) {
        int n = matrix.length;
        if(n > matrix[0].length) {
            n = matrix[0].length;
        }
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j)
                    r[i] = matrix[i][j];
            }
        }

        return r;
    }

    /**
     * 矩阵相乘
     * @param initiallyMatrix 原始二维数组
     * @param principalMatrix 主成分矩阵
     * @return result 结果矩阵
     */
    public Matrix multiply(double[][] initiallyMatrix, double[][] principalMatrix) {
        Matrix initially = new Matrix(initiallyMatrix);
        Matrix principal = new Matrix(principalMatrix);
        return initially.times(principal);
    }

    /**
     * pca过程,计算出降维后的矩阵
     * @param initiallyMatrix 原始矩阵
     * @param normalized 是否归一化
     * @return 降维后的矩阵
     */
    private double[][] pca(double[][] initiallyMatrix, boolean normalized) {
        double[][] averageArray = zeroCentered(initiallyMatrix);

        double[][]  covMatrix = cov(averageArray);

        double[][] featureValueMatrix = featureValueMatrix(covMatrix);
        double[][] featureVectorMatrix = featureVectorMatrix(covMatrix);

        if(normalized) {
            featureValueMatrix = normalized(featureValueMatrix);
            featureVectorMatrix = normalized(featureVectorMatrix);
        }


        Matrix principalMatrix = getPrincipalComponent(featureValueMatrix, featureVectorMatrix);

        return multiply(initiallyMatrix, principalMatrix.transpose().getArray()).getArray();
    }


    /**
     * 带有归一化的pca
     * @param initiallyMatrix 原始矩阵
     * @return 降维后的矩阵
     */
    public double[][] pcaNormalized(double[][] initiallyMatrix) {
        return pca(initiallyMatrix, true);
    }

    /**
     * 不进行归一化的pca
     * @param initiallyMatrix 原始矩阵
     * @return 降维后的矩阵
     */
    public double[][] pca(double[][] initiallyMatrix) {
        return pca(initiallyMatrix, false);
    }
}

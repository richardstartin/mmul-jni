package com.openkappa.mmul;

import java.nio.FloatBuffer;

public class MatrixMultiplication {

  public static native void multiply(FloatBuffer left, FloatBuffer right, FloatBuffer result, int width);

}

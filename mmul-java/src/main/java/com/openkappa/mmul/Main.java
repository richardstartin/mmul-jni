package com.openkappa.mmul;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.ThreadLocalRandom;

public class Main {

  static {
    System.loadLibrary("mmul-win32");
  }

  public static void main(String[] args) {
    int size = 512;
    FloatBuffer left = ByteBuffer.allocateDirect(size * size * 4).asFloatBuffer().put(data(size * size));
    FloatBuffer right = ByteBuffer.allocateDirect(size * size * 4).asFloatBuffer().put(data(size * size));
    FloatBuffer result = ByteBuffer.allocateDirect(size * size * 4).asFloatBuffer();
    MatrixMultiplication.multiply(left, right, result, size);
    for (int i = 0; i < size * size; ++i) {
      System.out.println(result.get(i));
    }
  }


  private static float[] data(int size) {
    float[] data = new float[size];
    for (int i = 0; i < data.length; ++i) {
      data[i] = ThreadLocalRandom.current().nextFloat();
    }
    return data;
  }
}

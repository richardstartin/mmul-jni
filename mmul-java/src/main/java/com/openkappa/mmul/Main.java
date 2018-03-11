package com.openkappa.mmul;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.ThreadLocalRandom;

import static java.lang.Integer.bitCount;

public class Main {

  static {
    System.loadLibrary("mmul-win32");
  }

  public static void main(String[] args) {
    int size = 512;
    FloatBuffer left = allocateDirectAligned(size * size * 4, 64).asFloatBuffer().put(data(size * size));
    FloatBuffer right = allocateDirectAligned(size * size * 4, 64).asFloatBuffer().put(data(size * size));
    FloatBuffer result = allocateDirectAligned(size * size * 4, 64).asFloatBuffer();
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


  /**
   * Taken from agrona
   * @param capacity
   * @param alignment
   * @return
   */
  public static ByteBuffer allocateDirectAligned(final int capacity, final int alignment) {
    if (bitCount(alignment) != 1) {
      throw new IllegalArgumentException("Must be a power of 2: alignment=" + alignment);
    }

    final ByteBuffer buffer = ByteBuffer.allocateDirect(capacity + alignment);

    final long address = address(buffer);
    final int remainder = (int)(address & (alignment - 1));
    final int offset = alignment - remainder;

    buffer.limit(capacity + offset);
    buffer.position(offset);

    return buffer.slice();
  }

  /**
   * Get the address at which the underlying buffer storage begins.
   *
   * @param buffer that wraps the underlying storage.
   * @return the memory address at which the buffer storage begins.
   */
  public static long address(final ByteBuffer buffer) {
    return ((sun.nio.ch.DirectBuffer)buffer).address();
  }
}

# -*- coding: utf-8 -*-

"""Functions to interleave and deinterleave integers together."""
from __future__ import division
from math import ceil
import pandas as pd

def show_bits(x, n):
    return format(x, '0{}b'.format(n))


def part1by1(n):
    """
    Inserts one 0 bit between each bit in `n`.

    n: 16-bit integer
    """
    n &= 65535
    n = (n | n << 8) & 16711935
    n = (n | n << 4) & 252645135
    n = (n | n << 2) & 858993459
    n = (n | n << 1) & 1431655765
    return n


def unpart1by1(n):
    """
    Gets every other bit from `n`.

    n: 32-bit integer
    """
    n &= 1431655765
    n = (n | n >> 1) & 858993459
    n = (n | n >> 2) & 252645135
    n = (n | n >> 4) & 16711935
    n = (n | n >> 8) & 65535
    return n


def part1by1_32(n):
    """
    Inserts one 0 bit between each bit in `n`.

    n: 32-bit integer
    """
    n &= 4294967295
    n = (n | n << 16) & 281470681808895
    n = (n | n << 8) & 71777214294589695
    n = (n | n << 4) & 1085102592571150095
    n = (n | n << 2) & 3689348814741910323
    n = (n | n << 1) & 6148914691236517205
    return int(n)


def part1by2(n):
    """
    Inserts two 0 bits between each bit in `n`.

    n: 16-bit integer
    """
    n &= 65535
    n = (n | n << 16) & 4278190335
    n = (n | n << 8) & 1031043870735
    n = (n | n << 4) & 13403570319555
    n = (n | n << 2) & 40210710958665
    return n


def unpart1by2(n):
    """
    Gets every third bit from `n`.

    n: 32-bit integer
    """
    n &= 40210710958665
    n = (n | n >> 2) & 13403570319555
    n = (n | n >> 4) & 1031043870735
    n = (n | n >> 8) & 4278190335
    n = (n | n >> 16) & 65535
    return n


def part1by3(n):
    """
    Inserts three 0 bits between each bit in `n`.

    n: 16-bit integer
    """
    n &= 65535
    n = (n | n << 27) & 35115652612607
    n = (n | n << 9) & 17733404569366535
    n = (n | n << 9) & 8072421338238709767
    n = (n | n << 3) & 1166717146542313521
    n = (n | n << 3) & 1229782938247303441
    return int(n)


def part1by3_cheat(n):
    """
    Inserts three 0 bits between each bit in `n`.

    Does so by inserting one 0 bit between each bit in `n`,
    then doing the same on that output.

    n: 16-bit integer
    """
    return part1by1_32(part1by1(n))


def unpart1by3(n):
    """
    Gets every fourth bit from `n`.

    n: 32-bit integer
    """
    n &= 1229782938247303441
    n = (n | n >> 3) & 1166717146542313521
    n = (n | n >> 3) & 8072421338238709767
    n = (n | n >> 9) & 17733404569366535
    n = (n | n >> 9) & 35115652612607
    n = (n | n >> 27) & 65535
    return n


def part1by4(n):
    """
    Inserts four 0 bits between each bit in `n`.

    n: 16-bit integer
    """
    n &= 65535
    n = (n | n << 32) & 280375465083135
    n = (n | n << 16) & 17293839061792849935
    n = (n | n << 8) & 3545237007667534236675
    n = (n | n << 4) & 38997607084342876603425
    return int(n)


def unpart1by4(n):
    """
    Gets every fifth bit from `n`.

    n: 32-bit integer
    """
    n &= 38997607084342876603425
    n = (n | n >> 4) & 3545237007667534236675
    n = (n | n >> 8) & 17293839061792849935
    n = (n | n >> 16) & 280375465083135
    n = (n | n >> 32) & 65535
    return n


def part1bym(n, m):
    """
    Inserts `m` 0 bits between each bit in `n`.

    n: integer
    m: integer
    """
    return int(''.join([e + m * '0' for e in format(n, 'b')])[:-m], base=2)


def unpart1bym(n, m):
    """
    Gets every `m + 1`th bit from `n`.

    n: integer
    m: integer
    """
    return int(format(n, 'b')[::-1][::m + 1][::-1], base=2)


def interleavem_naive(*nums):
    """
    Interleaves several numbers in binary representation.

    Uses a na\xc3\xafve method of picking from each number in turn.

    nums: iterable of integers
    """
    nums = nums[::-1]
    max_bits = max([num.bit_length() for num in nums])
    if max_bits == 0:
        return 0
    binary_strings = [show_bits(num, max_bits) for num in nums]
    s = ''
    for i in range(max_bits):
        for bs in binary_strings:
            s += bs[i]

    return int(s, base=2)


def interleavem(*nums):
    """
    Interleaves several numbers in binary representation.

    Uses a method of spacing out the numbers, filling with intermediate zeros,
    then slightly shifting them and combining.

    nums: iterable of integers
    """
    nums = nums[::-1]
    n = 0
    m = len(nums) - 1
    for i, num in enumerate(nums):
        n |= part1bym(num, m) << m - i

    return n


def interleave2(x, y):
    """
    Interleaves two numbers in binary representation.

    Uses a method of spacing out the numbers, filling with intermediate zeros,
    then slightly shifting them and combining.

    x, y: 16-bit integers
    """
    max_bits = max(x.bit_length(), y.bit_length())
    iterations = int(ceil(max_bits / 16))
    n = 0
    for i in range(iterations):
        interleaved = part1by1(x & 65535) | part1by1(y & 65535) << 1
        n |= interleaved << 32 * i
        x = x >> 16
        y = y >> 16

    return n


def interleave3(x, y, z):
    """
    Interleaves three numbers in binary representation.

    Uses a method of spacing out the numbers, filling with intermediate zeros,
    then slightly shifting them and combining.

    x, y, z: 16-bit integers
    """
    return part1by2(x) | part1by2(y) << 1 | part1by2(z) << 2


def interleave4(w, x, y, z):
    """
    Interleaves four numbers in binary representation.

    Uses a method of spacing out the numbers, filling with intermediate zeros,
    then slightly shifting them and combining.

    w, x, y, z: 16-bit integers
    """
    return part1by3(w) | part1by3(x) << 1 | part1by3(y) << 2 | part1by3(z) << 3


def interleave4_cheat(w, x, y, z):
    """
    Interleaves four numbers in binary representation.

    Uses a method of spacing out the numbers, filling with intermediate zeros,
    then slightly shifting them and combining.

    w, x, y, z: 16-bit integers
    """
    return part1by3_cheat(w) | part1by3_cheat(x) << 1 | part1by3_cheat(y) << 2 | part1by3_cheat(z) << 3


def interleave5(v, w, x, y, z):
    """
    Interleaves five numbers in binary representation.

    Uses a method of spacing out the numbers, filling with intermediate zeros,
    then slightly shifting them and combining.

    v, w, x, y, z: 16-bit integers
    """
    return part1by4(v) | part1by4(w) << 1 | part1by4(x) << 2 | part1by4(y) << 3 | part1by4(z) << 4


def deinterleave2(n):
    """Deinterleaves an integer into two integers."""
    iterations = int(ceil(n.bit_length() / 32))
    x = y = 0
    for i in range(iterations):
        x |= unpart1by1(n) << 16 * i
        y |= unpart1by1(n >> 1) << 16 * i
        n = n >> 32

    return [x, y]


def deinterleave3(n):
    """Deinterleaves an integer into three integers."""
    return [unpart1by2(n), unpart1by2(n >> 1), unpart1by2(n >> 2)]


def deinterleave4(n):
    """Deinterleaves an integer into four integers."""
    return [unpart1by3(n),
            unpart1by3(n >> 1),
            unpart1by3(n >> 2),
            unpart1by3(n >> 3)]


def deinterleave5(n):
    """Deinterleaves an integer into five integers."""
    return [unpart1by4(n),
            unpart1by4(n >> 1),
            unpart1by4(n >> 2),
            unpart1by4(n >> 3),
            unpart1by4(n >> 4)]


def deinterleavem(n, m):
    """Deinterleaves an integer into `m` integers."""
    return [unpart1bym(n >> i, m - 1) for i in range(m)]


if __name__ == "__main__":
    zvalue = interleavem(*[8, 5])
    bits = show_bits(interleavem(*[8, 5]), 64)
    print(zvalue, bits)

    zvalue = interleavem(*[1,1,2,2,1,77,8,2,1,19,21,11,1,1,1])
    print(zvalue)

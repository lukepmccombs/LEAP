#! /usr/bin/env python

##############################################################################
#
#   LEAP - Library for Evolutionary Algorithms in Python
#   Copyright (C) 2004  Jeffrey K. Bassett
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
##############################################################################

# Python 2 & 3 compatibility
from __future__ import print_function

import sys
import string
import copy
import random
import math
import numpy

from LEAP.encoding import BinaryEncoding



#############################################################################
#
# g2pSubEncoding
#
#############################################################################

class g2pSubEncoding(BinaryEncoding):
    """
    This class defines a binary to real-valued genetic encoding using a
    genotype to phenotype mapping.  This allows the mapping to be evolved in a
    meta-EA (or perhaps co-evolved).  Each mapping is then evaluated in a
    sub-EA that uses this encoding.
    """
    def __init__(self, problem, mapping):
        BinaryEncoding.__init__(self, problem, len(mapping))
        self.Nvec = len(mapping)
        self.Nelem = len(mapping[0]) - 1
        assert(self.Nvec > 0)
        assert(self.Nelem > 0)

        self.mapping = mapping

        # Preprocess the vectors to speed things up
        #self.vectors = []
        #for v in mapping:
        #    vector = 2**v[0] * numpy.array(v[1:])
        #    self.vectors += [vector]

        #vectors = [[0.0] * self.Nelem for v in range(self.Nvec)]
        vectors = [[0.0 for e in range(self.Nelem)] for v in range(self.Nvec)]
        for v in range(self.Nvec):
            for e in range(self.Nelem):
                vectors[v][e] = 2**mapping[v][0] * mapping[v][e+1]

        self.vectors = vectors


#    def decodeGenome(self, genome):
#        phenome = numpy.array([0.0] * self.Nelem)
#        for i in range(len(genome)):
#            if genome[i] == '1':
#                phenome += self.vectors[i]
#            
#        #print("g2pSubEncoding.decodeGenome")
#        return phenome.tolist()


    def decodeGenome(self, genome):
        phenome = [0.0] * self.Nelem
        for v in range(self.Nvec):
            if genome[v] == '1':
                for e in range(self.Nelem):
                    phenome[e] += self.vectors[v][e]
            
        return phenome



#############################################################################
#
# unit_test
#
#############################################################################
def myFunction(phenome):
   return(sum(abs(phenome)))


def unit_test():
    """
    Test the rule interpreter.
    """
    from LEAP.problem import FunctionOptimization

    mapping = [[3.0, 1.0, 0.0],
               [2.0, 1.0, 0.0],
               [1.0, 1.0, 0.0],
               [0.0, 1.0, 0.0],
               [3.0, 0.0, 1.0],
               [2.0, 0.0, 1.0],
               [1.0, 0.0, 1.0],
               [0.0, 0.0, 1.0]]

    problem = FunctionOptimization(myFunction, maximize = False)

    encoding = g2pSubEncoding(problem, mapping)
    genome = encoding.randomGenome()

    assert(len(genome) == 8)
    passed = True

    # Test the encoding
    genome = '10000101'
    phenome = encoding.decodeGenome(genome)
    print("phenome =", phenome)
    passed = passed and (phenome == [8.0, 5.0])

    genome = '00000001'
    phenome = encoding.decodeGenome(genome)
    print("phenome =", phenome)
    passed = passed and (phenome == [0.0, 1.0])

    print()
    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()


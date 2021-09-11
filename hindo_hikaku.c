//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

const long long max_size = 2000;         // max length of strings
const long long N = 1;                   // number of closest words
const long long max_w = 50;              // max length of vocabulary entries

float quantize(float num, int bitlevel) {

  if (bitlevel == 0) {
    // Special bitlevel 0 => full precision
    return num;
  }
  
  // Extract sign
  float retval = 0;
  float sign = num < 0 ? -1 : 1;
  num *= sign;
  
  // Boundaries: 0
  if (bitlevel == 1) {
    return sign / 3;
  }
  
  // Determine boundary and discrete activation value (2 bits)
  // Boundaries: 0, .5
  if (bitlevel == 2) {
    if (num >= 0 && num <= .5) retval = .25; 
    else retval = .75;
  }

  // Determine boundary and discrete activation value (4 bits = 16 values)
  // Boundaries: 0, .1, .2, .3, .4, .5, .6, .7, .8
  //real boundaries[] = {0, .25, .5, .75, 1, 1.25, 1.5, 1.75};
  if (bitlevel >= 4) {
    int segmentation = pow(2, bitlevel-1);
    int casted = (num * segmentation) + (float).5;
    casted = casted > segmentation ? segmentation : casted;
    retval = casted / (float)segmentation;
  }

  return sign * retval;
}

int main(int argc, char **argv)
{
  FILE *f;
  FILE *fo;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size], file_output[max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
  int bitlevel = 0;
  float *M;
  char *vocab;
  int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  if (argc < 2) {
    printf("Usage: ./compute-accuracy <FILE> <bitlevel> <threshold>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000)\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  if (argc > 2) bitlevel = atoi(argv[2]);
  if (argc > 3) threshold = atoi(argv[3]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  if (threshold) if (words > threshold) words = threshold;
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (float *)malloc(words * size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
    return -1;
  }
  printf("Starting eval...\n");
  fflush(stdout);
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    for (a = 0; a < size; a++) M[a+b*size] = quantize(M[a+b*size], bitlevel);
    len = 0;
    fo = fopen("norm_file", "wb");
    for (a = 0; a < size; a++) {
        len += M[a + b * size] * M[a + b * size];
    }
    len = sqrt(len);
    fprintf(fo, "%f\n", len);
    fclose(fo);
  }
  fclose(f);
}

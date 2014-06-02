function [vector_normalized] = j_normalize(vector)

% shift to zero
vector_min = min(vector);
vector = vector - vector_min;

% normalize
vector_max = max(vector);
vector_normalized = vector/vector_max;

% transpose
vector_normalized = vector_normalized';
